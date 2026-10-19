# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio

import pytest
import pytest_asyncio
from pydantic import Field

from sensorkit.backend.base import KVError
from sensorkit.backend.event import Event
from sensorkit.core.state import EventSourcedState


class CounterEvent(Event):
    value: int
    details: dict[str, int] = Field(default_factory=dict)


class CounterState(EventSourcedState):
    counter: CounterEvent


@pytest.fixture
def state() -> CounterState:
    return CounterState(counter=CounterEvent(value=0))


@pytest_asyncio.fixture
async def emitted(kit, entity_impl):
    """Observe counter events from the registered entity."""
    return await kit.entity(entity_impl.entity).monitor_event(CounterEvent)


async def apply_update(state: CounterState, entity, entry: str, event: Event, **options):
    """Apply an event through either public update API."""
    match entry:
        case "update":
            return await state.update(entity, event, **options)
        case "updating":
            async with state.updating(entity) as update:
                return await update(event, **options)


@pytest.mark.asyncio
async def test_updating_applies_events(state, entity_impl, emitted):
    async with state.updating(entity_impl) as update:
        event = CounterEvent(value=state.counter.value + 1)
        await update(event)

    assert state.counter == event
    assert await anext(emitted) == event
    assert (await entity_impl.kv_get_model(CounterState)).counter == event


@pytest.mark.asyncio
async def test_updating_excludes_other_updates(state, entity_impl):
    async with state.updating(entity_impl) as update:
        waiting = asyncio.create_task(state.update(entity_impl, CounterEvent(value=2)))
        await asyncio.sleep(0)

        assert not waiting.done()
        await update(CounterEvent(value=1))
        assert state.counter.value == 1

    await waiting
    assert state.counter.value == 2


@pytest.mark.asyncio
async def test_updating_releases_on_error(state, entity_impl):
    with pytest.raises(RuntimeError, match="rejected"):
        async with state.updating(entity_impl):
            raise RuntimeError("rejected")

    async with asyncio.timeout(1.0):
        await state.update(entity_impl, CounterEvent(value=1))

    assert state.counter.value == 1


@pytest.mark.asyncio
async def test_updating_releases_on_cancel(state, entity_impl):
    entered = asyncio.Event()

    async def hold():
        async with state.updating(entity_impl):
            entered.set()
            await asyncio.Event().wait()

    holder = asyncio.create_task(hold())
    await entered.wait()
    holder.cancel()

    with pytest.raises(asyncio.CancelledError):
        await holder

    async with asyncio.timeout(1.0):
        await state.update(entity_impl, CounterEvent(value=1))

    assert state.counter.value == 1


@pytest.mark.asyncio
async def test_update_callable_expires_with_scope(state, entity_impl):
    async with state.updating(entity_impl) as update:
        pass

    with pytest.raises(RuntimeError):
        await update(CounterEvent(value=1))

    assert state.counter.value == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["update", "updating"])
async def test_update_without_publishing_state(state, entity_impl, emitted, entry):
    event = CounterEvent(value=1)
    await apply_update(state, entity_impl, entry, event, publish_state=False)

    assert state.counter == event
    assert await anext(emitted) == event

    with pytest.raises(KVError):
        await entity_impl.kv_get_model(CounterState)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["update", "updating"])
async def test_update_returns_snapshot(state, entity_impl, entry):
    assert await apply_update(state, entity_impl, entry, CounterEvent(value=1)) is None

    snapshot = await apply_update(
        state, entity_impl, entry, CounterEvent(value=2), return_snapshot=True
    )
    await state.update(entity_impl, CounterEvent(value=3))

    assert snapshot.counter.value == 2
    assert state.counter.value == 3


@pytest.mark.asyncio
async def test_scoped_snapshot_with_queued_writer(state, entity_impl):
    queued = asyncio.Event()

    async def write_later():
        queued.set()

        async with state.updating(entity_impl) as update:
            await update(CounterEvent(value=2))

    async with state.updating(entity_impl) as update:
        writer = asyncio.create_task(write_later())
        await queued.wait()
        snapshot = await update(CounterEvent(value=1), return_snapshot=True)

        # The snapshot's update scope must be independent of the held source scope.
        async with asyncio.timeout(1.0), snapshot.updating(entity_impl):
            pass

        snapshot.counter.details["count"] = 9
        assert state.counter.details == {}

    async with asyncio.timeout(1.0):
        await writer

    assert snapshot.counter.details == {"count": 9}
    assert state.counter.value == 2


@pytest.mark.asyncio
async def test_update_snapshot_with_queued_writer(state, entity_impl):
    async with state.updating(entity_impl):
        snapshotting = asyncio.create_task(
            state.update(entity_impl, CounterEvent(value=1), return_snapshot=True)
        )
        queued = asyncio.create_task(state.update(entity_impl, CounterEvent(value=2)))
        await asyncio.sleep(0)

    async with asyncio.timeout(1.0):
        snapshot = await snapshotting
        await queued

    assert snapshot.counter.value == 1
    assert state.counter.value == 2
