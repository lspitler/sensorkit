# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal, override

import pytest
import pytest_asyncio
import uuid_utils.compat as uuid
from pydantic import BaseModel

from sensorkit.backend.base import Backend, BackendError, Subject
from sensorkit.backend.event import Event
from sensorkit.backend.fake import FakeBackendImpl
from sensorkit.backend.request import CallError
from sensorkit.common.keyword import KeywordDict, declare_keyword
from sensorkit.core.client import SensorKit
from sensorkit.core.controller import (
    ControllerClient,
    ControllerDevice,
    ControllerEnableState,
    ControllerOperatingState,
    ControllerState,
    InternalControllerState,
    TaskExecutionState,
    TaskFinishInfo,
)
from sensorkit.core.device import DeviceClient
from sensorkit.core.impl.controller import ControllerImpl
from sensorkit.core.task import (
    CollectTask,
    InitTask,
    RecoverTask,
    ShutdownTask,
    StandbyTask,
    TaskExecution,
)

OPERATE = InternalControllerState.OPERATE
STANDBY = InternalControllerState.STANDBY
SHUTDOWN = InternalControllerState.SHUTDOWN
ERROR = InternalControllerState.ERROR
UNKNOWN = InternalControllerState.UNKNOWN
TRANSITION = InternalControllerState.TRANSITION


@dataclass
class PublishFault:
    """A rule that blocks or fails one matching stream publication."""

    match: Callable[[dict[str, Any]], bool]
    fail: bool = False
    reached: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)


class GatedBackendImpl(FakeBackendImpl):
    """Fake backend with gates and failures for stream publication."""

    def __init__(self):
        super().__init__()
        self.faults: list[PublishFault] = []

    def arm(self, match: Callable[[dict[str, Any]], bool], *, fail=False) -> PublishFault:
        """Block or fail the next matching publication."""
        fault = PublishFault(match, fail)
        self.faults.append(fault)
        return fault

    def take_fault(self, payload: bytes) -> PublishFault | None:
        """Consume the first fault matching the publication payload."""
        try:
            data = json.loads(payload)
        except ValueError:
            return None

        if not isinstance(data, dict):
            return None

        for fault in self.faults:
            if fault.match(data):
                self.faults.remove(fault)
                return fault

        return None

    @override
    async def stream_publish(self, target: Subject, payload: bytes):
        if fault := self.take_fault(payload):
            fault.reached.set()

            if fault.fail:
                raise BackendError("injected publication failure")

            await fault.release.wait()

        await super().stream_publish(target, payload)


@pytest_asyncio.fixture
async def gated_backend():
    return GatedBackendImpl()


@pytest_asyncio.fixture
async def gated_kit(gated_backend):
    return SensorKit(Backend(impl=gated_backend))


def starting_execution(data: dict[str, Any]) -> bool:
    return data.get("event_model") == "TaskExecutionState" and data.get("executing") is True


def finishing_execution(data: dict[str, Any]) -> bool:
    return data.get("event_model") == "TaskExecutionState" and data.get("finished") is not None


def entering_transition(data: dict[str, Any]) -> bool:
    return (
        data.get("event_model") == "ControllerOperatingState" and data["current"] == "transition"
    )


def succeeding_call(data: dict[str, Any]) -> bool:
    return data.get("call_state") == "success"


def reporting(state: InternalControllerState) -> Callable[[dict[str, Any]], bool]:
    def match(data: dict[str, Any]) -> bool:
        return data.get("event_model") == "ControllerOperatingState" and data["current"] == state

    return match


@contextlib.asynccontextmanager
async def watch_state(client: ControllerClient):
    """Open a controller snapshot subscription and consume its initial value."""
    async with contextlib.aclosing(client.kv_monitor_model(ControllerState)) as states:
        await anext(states)
        yield states


async def collect_until(
    states: AsyncIterator[ControllerState],
    predicate: Callable[[ControllerState], bool],
) -> list[ControllerState]:
    """Collect snapshots through the first match."""
    seen = []

    async for state in states:
        seen.append(state)

        if predicate(state):
            return seen

    raise AssertionError("state stream ended")


def execution_id(state: ControllerState) -> uuid.UUID | None:
    execution = state.execution_state.execution
    return execution.task_id if execution else None


def is_finished(state: ControllerState, task_id: uuid.UUID) -> bool:
    return execution_id(state) == task_id and state.execution_state.finished is not None


def operating_history(
    states: list[ControllerState],
    *task_ids: uuid.UUID,
) -> list[tuple[uuid.UUID | None, InternalControllerState]]:
    """Return execution IDs and operating states for the selected tasks."""
    return [
        (execution_id(state), state.operating_state.current)
        for state in states
        if execution_id(state) in task_ids
    ]


@pytest.mark.asyncio
async def test_controller(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    disabled = asyncio.Event()
    enabled = asyncio.Event()
    done = asyncio.Event()

    @controller.on_disable
    async def disable():
        disabled.set()

    @controller.on_enable
    async def on_enable():
        enabled.set()

    @controller.task_handler(InitTask)
    async def run_task(task: InitTask):
        assert ControllerImpl.current.get() is controller
        assert isinstance(task, InitTask)
        done.set()

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        await cli.disable()
        await disabled.wait()
        await cli.enable()
        await enabled.wait()
        await cli.execute_task(InitTask(task_id=uuid.uuid1(), controller_id="mycontroller"))
        await done.wait()


@pytest.mark.asyncio
async def test_controller_derived_task(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    class MyCollectTask(CollectTask):
        task_type: Literal["my_collect"] = "my_collect"
        my_field: int

    remaining = {MyCollectTask, CollectTask}

    @controller.task_handler(MyCollectTask)
    async def run_custom_collect(task: MyCollectTask):
        assert type(task) is MyCollectTask
        remaining.discard(MyCollectTask)

    @controller.task_handler(CollectTask)
    async def run_generic_collect(task: CollectTask):
        assert type(task) is CollectTask
        remaining.discard(CollectTask)

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        await cli.enable()
        await cli.execute_task(CollectTask(task_id=uuid.uuid1(), controller_id="mycontroller"))
        await cli.execute_task(
            MyCollectTask(task_id=uuid.uuid1(), controller_id="mycontroller", my_field=42)
        )
        assert not remaining


@pytest.mark.asyncio
async def test_controller_interrupt(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()
    can_finish = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_task(_: InitTask):
        started.set()
        await can_finish.wait()

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        exec_call = cli.execute_task(InitTask(task_id=uuid.uuid1(), controller_id="mycontroller"))
        await exec_call.invoke()
        await started.wait()
        started.clear()

        with pytest.raises(CallError):
            await cli.execute_task(
                InitTask(task_id=uuid.uuid1(), controller_id="mycontroller"),
                interrupt=False,
            )

        assert not started.is_set()

        interrupting_call = cli.execute_task(
            InitTask(task_id=uuid.uuid1(), controller_id="mycontroller"),
            interrupt=True,
        )
        await interrupting_call.invoke()
        await started.wait()

        with pytest.raises(CallError):
            await exec_call

        can_finish.set()
        await interrupting_call


@pytest.mark.asyncio
async def test_controller_abort(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    ready = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_task(_: InitTask):
        ready.set()
        await asyncio.sleep(5)

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        exec_call = cli.execute_task(InitTask())
        # The controller mints the task_id; learn it from the initial response.
        exec_response = await exec_call.invoke()
        task_id = exec_response.execution.task_id
        await ready.wait()

        abort_call = cli.abort_task()
        abort_response = await abort_call.invoke()

        assert abort_response.aborting
        assert abort_response.task_id == task_id

        await abort_call.wait()
        assert abort_call.result() is None

        with pytest.raises(CallError):
            await exec_call.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("succeed", InternalControllerState.SHUTDOWN),
        ("fail", InternalControllerState.ERROR),
        ("abort", InternalControllerState.UNKNOWN),
    ],
)
async def test_lifecycle_task_reports_arrival_only_on_success(kit, outcome, expected):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    @controller.task_handler(ShutdownTask)
    async def run_shutdown(_: ShutdownTask):
        started.set()

        match outcome:
            case "fail":
                raise RuntimeError("enclosure fault")
            case "abort":
                await asyncio.sleep(5)

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        await cli.execute_task(InitTask())

        exec_call = cli.execute_task(ShutdownTask())
        task_id = (await exec_call.invoke()).execution.task_id
        await started.wait()

        match outcome:
            case "succeed":
                await exec_call.wait()
            case "fail":
                with pytest.raises(CallError):
                    await exec_call.wait()
            case "abort":
                await cli.abort_task()

                with pytest.raises(CallError):
                    await exec_call.wait()

        state = await cli.kv_get_model(ControllerState)

    assert state.operating_state.current == expected
    assert state.operating_state.previous == TRANSITION
    assert state.operating_state.target is None
    assert is_finished(state, task_id)
    assert state.execution_state.finished.aborted == (outcome == "abort")
    assert bool(state.execution_state.finished.error) == (outcome == "fail")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("task_type", "target"),
    [(InitTask, OPERATE), (StandbyTask, STANDBY), (ShutdownTask, SHUTDOWN)],
    ids=("init", "standby", "shutdown"),
)
async def test_lifecycle_task_reports_transition(kit, task_type, target):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()
    can_finish = asyncio.Event()

    @controller.task_handler(task_type)
    async def run_task(_):
        started.set()
        await can_finish.wait()

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        exec_call = cli.execute_task(task_type())
        task_id = (await exec_call.invoke()).execution.task_id
        await started.wait()

        running = await cli.kv_get_model(ControllerState)
        can_finish.set()
        await exec_call
        settled = await cli.kv_get_model(ControllerState)

    assert running.operating_state.current == TRANSITION
    assert running.operating_state.target == target
    assert running.execution_state.executing
    assert execution_id(running) == task_id

    assert settled.operating_state.current == target
    assert settled.operating_state.previous == TRANSITION
    assert settled.operating_state.target is None
    assert is_finished(settled, task_id)


@pytest.mark.asyncio
async def test_fast_lifecycle_task_reports_transition(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    cli = kit.controller("mycontroller")

    async with asyncio.timeout(1.0), watch_state(cli) as states:
        exec_call = cli.execute_task(InitTask())
        task_id = (await exec_call.invoke()).execution.task_id
        await exec_call
        seen = await collect_until(states, lambda state: is_finished(state, task_id))

    assert operating_history(seen, task_id) == [(task_id, TRANSITION), (task_id, OPERATE)]


@pytest.mark.asyncio
async def test_interrupting_task_settles_prior_transition_first(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        started.set()
        await asyncio.Event().wait()

    @controller.task_handler(ShutdownTask)
    async def run_shutdown(_: ShutdownTask):
        pass

    cli = kit.controller("mycontroller")

    async with asyncio.timeout(1.0), watch_state(cli) as states:
        first = cli.execute_task(InitTask())
        first_id = (await first.invoke()).execution.task_id
        await started.wait()

        second = cli.execute_task(ShutdownTask(), interrupt=True)
        second_id = (await second.invoke()).execution.task_id
        await second

        with pytest.raises(CallError):
            await first.wait()

        seen = await collect_until(states, lambda state: is_finished(state, second_id))

    assert operating_history(seen, first_id, second_id) == [
        (first_id, TRANSITION),
        (first_id, UNKNOWN),
        (second_id, TRANSITION),
        (second_id, SHUTDOWN),
    ]

    interrupted = next(state for state in seen if is_finished(state, first_id))
    assert interrupted.execution_state.finished.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["succeed", "fail", "abort"])
async def test_non_lifecycle_task_preserves_reported_state(kit, outcome):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()
    can_finish = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    @controller.task_handler(CollectTask)
    async def run_collect(_: CollectTask):
        started.set()

        match outcome:
            case "succeed":
                await can_finish.wait()
            case "fail":
                await can_finish.wait()
                raise RuntimeError("camera fault")
            case "abort":
                await asyncio.Event().wait()

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        await cli.execute_task(InitTask())

        exec_call = cli.execute_task(CollectTask())
        task_id = (await exec_call.invoke()).execution.task_id
        await started.wait()

        await controller.set_internal_state(ERROR)
        can_finish.set()

        match outcome:
            case "succeed":
                await exec_call.wait()
            case "fail":
                with pytest.raises(CallError):
                    await exec_call.wait()
            case "abort":
                await cli.abort_task()

                with pytest.raises(CallError):
                    await exec_call.wait()

        state = await cli.kv_get_model(ControllerState)

    assert state.operating_state.current == ERROR
    assert state.operating_state.previous == OPERATE
    assert state.operating_state.target is None
    assert is_finished(state, task_id)
    assert state.execution_state.finished.aborted == (outcome == "abort")


@pytest.mark.asyncio
async def test_state_report_rejected_during_lifecycle_task(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()
    can_finish = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        started.set()
        await can_finish.wait()

    with pytest.raises(ValueError):
        await controller.set_internal_state(TRANSITION)

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        exec_call = cli.execute_task(InitTask())
        await exec_call.invoke()
        await started.wait()

        with pytest.raises(ValueError):
            await controller.set_internal_state(STANDBY)

        running = await cli.kv_get_model(ControllerState)
        can_finish.set()
        await exec_call

        await controller.set_internal_state(STANDBY)
        reported = await cli.kv_get_model(ControllerState)

    assert running.operating_state.current == TRANSITION
    assert reported.operating_state.current == STANDBY
    assert reported.operating_state.previous == OPERATE


def seeded_state(
    operating: ControllerOperatingState,
    execution_state: TaskExecutionState,
) -> ControllerState:
    return ControllerState(
        enable_state=ControllerEnableState(enabled=True),
        operating_state=operating,
        execution_state=execution_state,
    )


def seeded_execution() -> TaskExecution:
    return TaskExecution(task=ShutdownTask(), task_id=uuid.uuid7(), controller_id="mycontroller")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operating", "executing", "aborting", "finished"),
    [
        (ControllerOperatingState(current=TRANSITION, target=SHUTDOWN), True, False, None),
        (ControllerOperatingState(current=TRANSITION, target=SHUTDOWN), True, True, None),
        (
            ControllerOperatingState(current=OPERATE, target=SHUTDOWN, context=KeywordDict()),
            True,
            False,
            None,
        ),
        (
            ControllerOperatingState(current=TRANSITION, target=SHUTDOWN),
            False,
            False,
            TaskFinishInfo(error="lost"),
        ),
    ],
    ids=("executing", "aborting", "legacy_target", "finished"),
)
async def test_startup_recovers_interrupted_transition(
    kit, operating, executing, aborting, finished
):
    execution = seeded_execution()
    cli = kit.controller("mycontroller")
    observed: list[ControllerState] = []

    await cli.kv_put_model(
        seeded_state(
            operating,
            TaskExecutionState(
                executing=executing,
                aborting=aborting,
                finished=finished,
                execution=execution,
            ),
        )
    )

    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        impl = ControllerImpl.for_service_context(sc, "mycontroller")

        @impl.on_attach
        async def observe():
            observed.append(await cli.kv_get_model(ControllerState))

        await sc.register_impl(impl)

    [state] = observed
    assert state.operating_state.current == UNKNOWN
    assert state.operating_state.previous == operating.current
    assert state.operating_state.target is None
    assert is_finished(state, execution.task_id)

    if finished is None:
        assert state.execution_state.finished.aborted
    else:
        assert state.execution_state.finished == finished


@pytest.mark.asyncio
async def test_startup_keeps_settled_state(kit):
    cli = kit.controller("mycontroller")
    seeded = seeded_state(
        ControllerOperatingState(current=SHUTDOWN, previous=TRANSITION, context=KeywordDict()),
        TaskExecutionState(finished=TaskFinishInfo(), execution=seeded_execution()),
    )

    await cli.kv_put_model(seeded)

    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        await sc.register_controller("mycontroller")
        state = await cli.kv_get_model(ControllerState)

    assert state.model_dump() == seeded.model_dump()


@pytest.mark.asyncio
async def test_state_report_waits_for_transition_start(gated_kit, gated_backend):
    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()
    can_finish = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        started.set()
        await can_finish.wait()

    async with asyncio.timeout(1.0):
        cli = gated_kit.controller("mycontroller")

        # An earlier report holds the update lock while it publishes.
        fault = gated_backend.arm(reporting(STANDBY))
        earlier = asyncio.create_task(controller.set_internal_state(STANDBY))
        await fault.reached.wait()

        # Queue the lifecycle start behind the held report.
        exec_call = cli.execute_task(InitTask())
        await exec_call.invoke()

        later = asyncio.create_task(controller.set_internal_state(ERROR))
        await asyncio.sleep(0)
        fault.release.set()
        await earlier

        with pytest.raises(ValueError):
            await later

        await started.wait()
        running = await cli.kv_get_model(ControllerState)
        can_finish.set()
        await exec_call

    assert running.operating_state.current == TRANSITION
    assert running.operating_state.target == OPERATE


@pytest.mark.asyncio
async def test_abort_after_settlement_is_rejected(gated_kit, gated_backend):
    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    async with asyncio.timeout(1.0):
        cli = gated_kit.controller("mycontroller")

        # Holding the success response keeps the settled execution's request in flight.
        fault = gated_backend.arm(succeeding_call)
        exec_call = cli.execute_task(InitTask())
        task_id = (await exec_call.invoke()).execution.task_id
        await fault.reached.wait()

        with pytest.raises(CallError):
            await cli.abort_task().invoke()

        fault.release.set()
        await exec_call
        state = await cli.kv_get_model(ControllerState)

    assert state.operating_state.current == OPERATE
    assert is_finished(state, task_id)
    assert not state.execution_state.aborting
    assert not state.execution_state.finished.aborted


@pytest.mark.asyncio
async def test_interrupt_during_settlement_completes_it(gated_kit, gated_backend):
    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        started.set()
        await asyncio.Event().wait()

    @controller.task_handler(ShutdownTask)
    async def run_shutdown(_: ShutdownTask):
        pass

    cli = gated_kit.controller("mycontroller")

    async with asyncio.timeout(1.0), watch_state(cli) as states:
        first = cli.execute_task(InitTask())
        first_id = (await first.invoke()).execution.task_id
        await started.wait()

        fault = gated_backend.arm(finishing_execution)
        abort_call = cli.abort_task()
        await abort_call.invoke()
        await fault.reached.wait()

        # The interrupt cancels the first execution again while it publishes its final state.
        second = cli.execute_task(ShutdownTask(), interrupt=True)
        second_id = (await second.invoke()).execution.task_id
        fault.release.set()
        await second
        await abort_call.wait()

        with pytest.raises(CallError):
            await first.wait()

        seen = await collect_until(states, lambda state: is_finished(state, second_id))

    assert operating_history(seen, first_id, second_id) == [
        (first_id, TRANSITION),
        (first_id, TRANSITION),
        (first_id, UNKNOWN),
        (second_id, TRANSITION),
        (second_id, SHUTDOWN),
    ]

    interrupted = next(state for state in seen if is_finished(state, first_id))
    assert interrupted.execution_state.finished.aborted


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing", "expected", "recorded"),
    [(starting_execution, OPERATE, "init"), (entering_transition, ERROR, "standby")],
    ids=("execution", "transition"),
)
async def test_failed_start_publication_skips_handler(
    gated_kit, gated_backend, failing, expected, recorded
):
    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    ran = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    @controller.task_handler(StandbyTask)
    async def run_standby(_: StandbyTask):
        ran.set()

    async with asyncio.timeout(1.0):
        cli = gated_kit.controller("mycontroller")
        await cli.execute_task(InitTask(), context=KeywordDict({"mode": "init"}))

        gated_backend.arm(failing, fail=True)
        exec_call = cli.execute_task(StandbyTask(), context=KeywordDict({"mode": "standby"}))
        task_id = (await exec_call.invoke()).execution.task_id

        with pytest.raises(CallError):
            await exec_call.wait()

        state = await cli.kv_get_model(ControllerState)

    assert not ran.is_set()
    assert state.operating_state.current == expected
    assert state.operating_state.target is None
    assert state.operating_state.context == {"mode": recorded}
    assert is_finished(state, task_id)
    assert state.execution_state.finished.error


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failing", "expected", "previous"),
    [(starting_execution, OPERATE, TRANSITION), (reporting(UNKNOWN), UNKNOWN, OPERATE)],
    ids=("execution", "invalidation"),
)
async def test_failed_recovery_start_skips_handler(
    gated_kit, gated_backend, failing, expected, previous
):
    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    ran = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    @controller.task_handler(RecoverTask)
    async def run_recover(_: RecoverTask):
        ran.set()

    async with asyncio.timeout(1.0):
        cli = gated_kit.controller("mycontroller")
        await cli.execute_task(InitTask(), context=KeywordDict({"mode": "init"}))
        emitted: list[Event] = []

        async with contextlib.aclosing(await cli.monitor_all_events()) as events:
            gated_backend.arm(failing, fail=True)
            exec_call = cli.execute_task(RecoverTask())
            task_id = (await exec_call.invoke()).execution.task_id

            with pytest.raises(CallError):
                await exec_call.wait()

            state = await cli.kv_get_model(ControllerState)

            async for event in events:
                emitted.append(event)

                if isinstance(event, ControllerOperatingState):
                    break

    # Settlement emits the retained state after the execution finishes and before the snapshot.
    *_, finished, settled = emitted
    assert isinstance(finished, TaskExecutionState)
    assert finished.execution.task_id == task_id
    assert finished.finished is not None
    assert settled.event_id == state.operating_state.event_id
    assert settled == state.operating_state

    assert not ran.is_set()
    assert state.operating_state.current == expected
    assert state.operating_state.previous == previous
    assert state.operating_state.target is None
    assert state.operating_state.context == {"mode": "init"}
    assert is_finished(state, task_id)
    assert state.execution_state.finished.error
    assert not state.execution_state.finished.aborted


@pytest.mark.asyncio
async def test_failed_final_publication_fails_request(gated_kit, gated_backend):
    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    @controller.task_handler(InitTask)
    async def run_init(_: InitTask):
        pass

    async with asyncio.timeout(1.0):
        cli = gated_kit.controller("mycontroller")
        gated_backend.arm(finishing_execution, fail=True)
        exec_call = cli.execute_task(InitTask())

        with pytest.raises(CallError):
            await exec_call

        stale = await cli.kv_get_model(ControllerState)

        # No running task owns the stale transition, so a report replaces it.
        await controller.set_internal_state(STANDBY)
        reported = await cli.kv_get_model(ControllerState)

    assert stale.operating_state.current == TRANSITION
    assert stale.execution_state.executing
    assert reported.operating_state.current == STANDBY
    assert reported.operating_state.target is None


@pytest.mark.asyncio
async def test_failed_startup_recovery_fails_registration(gated_kit, gated_backend):
    cli = gated_kit.controller("mycontroller")
    seeded = seeded_state(
        ControllerOperatingState(current=TRANSITION, target=SHUTDOWN),
        TaskExecutionState(executing=True, execution=seeded_execution()),
    )

    await cli.kv_put_model(seeded)

    async with asyncio.timeout(1.0):
        sc = await gated_kit.register_service("testservice", "0.1.0")
        gated_backend.arm(finishing_execution, fail=True)

        with pytest.raises(BackendError):
            await sc.register_controller("mycontroller")

        state = await cli.kv_get_model(ControllerState)

    assert state.model_dump() == seeded.model_dump()


@pytest.mark.asyncio
@pytest.mark.parametrize("with_id", [True, False], ids=("with_id", "without_id"))
async def test_wait_for_task(kit, with_id):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")

    started = asyncio.Event()
    can_finish = asyncio.Event()

    @controller.task_handler(InitTask)
    async def run_task(_: InitTask):
        started.set()
        await can_finish.wait()

    async with asyncio.timeout(1.0):
        cli = kit.controller("mycontroller")
        await cli.enable()

        # No task is running yet: waiting for an arbitrary id returns None immediately.
        probe_id = uuid.uuid7() if with_id else None
        event = await cli.wait_for_task(task_id=probe_id)
        assert event is None if with_id else not event.executing

        exec_call = cli.execute_task(InitTask())
        # The controller mints the task_id; learn it from the initial response.
        response = await exec_call.invoke()
        task_id = response.execution.task_id
        await started.wait()

        want_id = task_id if with_id else None
        wait_fut = asyncio.create_task(cli.wait_for_task(task_id=want_id))
        await asyncio.sleep(0)
        assert not wait_fut.done()

        can_finish.set()
        event = await wait_fut
        assert event is not None
        assert event.finished is not None
        assert not event.finished.aborted
        assert not event.executing

        if with_id:
            assert event.execution.task_id == task_id

        await exec_call


@pytest.mark.asyncio
async def test_controller_use_device(kit):
    @declare_keyword
    class TestKeyword(BaseModel):
        value: int

    async with asyncio.timeout(2.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")
        # We need an actual device registered in the backend for subscriptions to work
        device_impl = await sc.register_device("camera")

    # 1. Register a device for use within the controller.
    # Returns a client instance for interacting with the registered device.
    camera_client = controller.use_device("camera")
    assert isinstance(camera_client, DeviceClient)
    assert camera_client.entity.path == ("camera",)

    # 2. Access the registered device via get_device(name)
    # Returns a ControllerDevice instance.
    camera_device = controller.get_device("camera")
    assert isinstance(camera_device, ControllerDevice)
    assert camera_device.client is camera_client

    # 3. KeyError: If attempting to access a device via get_device(name) that hasn't been registered.
    with pytest.raises(KeyError):
        controller.get_device("unknown")

    # 4. all_devices() returns all registered devices.
    devices = list(controller.all_devices())
    assert len(devices) == 1
    assert devices[0] is camera_device

    # 5. Register device and subscribe to specific keyword types.
    # Subscriptions remain inactive until start_device_subscriptions() is called.
    controller.use_device("camera", subscribe=[TestKeyword])

    # Check that it's still the same device
    assert controller.get_device("camera") is camera_device

    # 6. Verify inactivity before start_device_subscriptions()
    await device_impl.publish(TestKeyword(value=10))
    await asyncio.sleep(0.05)
    assert camera_device.get(TestKeyword) is None

    await controller.start_device_subscriptions()

    # 7. Access cached subscription values from the ControllerDevice instance.
    # Publish a value from the device implementation
    await device_impl.publish(TestKeyword(value=42))

    # Wait for the update to propagate through the subscription
    for _ in range(10):
        if camera_device.get(TestKeyword) is not None:
            break
        await asyncio.sleep(0.01)

    pointing = camera_device.get(TestKeyword)
    assert pointing is not None
    assert pointing.value == 42

    # Also test __getitem__
    assert camera_device[TestKeyword].value == 42

    # 8. Test multiple use_device calls accumulate subscriptions
    @declare_keyword
    class AnotherKeyword(BaseModel):
        other_value: int

    controller.use_device("camera", subscribe=[AnotherKeyword])
    await device_impl.publish(AnotherKeyword(other_value=100))

    # Wait for the update to propagate through the subscription
    for _ in range(10):
        if camera_device.get(AnotherKeyword) is not None:
            break
        await asyncio.sleep(0.01)

    assert camera_device.get(AnotherKeyword).other_value == 100
    # Original keyword still works
    assert camera_device.get(TestKeyword).value == 42

    # 9. stop_device_subscriptions()
    await controller.stop_device_subscriptions()


# Legacy state migration: versions before the Task/TaskExecution split stored the executing
# task as a flat ControllerTask on `execution_state.execution`. A `mode="before"` validator on
# TaskExecutionState upgrades that shape so old state from a NATS broker still loads.
_LEGACY_TASK = {
    "task_type": "init",
    "task_id": "0192c3f4-5678-7abc-8def-0123456789ab",
    "controller_id": "legacy-controller",
    "context": None,
    "end_time": "2026-01-01T00:00:00Z",
}
_LEGACY_TASK_ID = uuid.UUID(_LEGACY_TASK["task_id"])
_LEGACY_EXPIRY = datetime(2026, 1, 1, tzinfo=UTC)


def test_legacy_task_execution_state_migrates_flat_task():
    """A legacy flat task on a TaskExecutionState is wrapped into a TaskExecution envelope."""
    state = TaskExecutionState.model_validate(
        {"executing": True, "task": dict(_LEGACY_TASK)}
    )

    assert isinstance(state.execution, TaskExecution)
    assert isinstance(state.execution.task, InitTask)
    assert state.execution.task_id == _LEGACY_TASK_ID
    assert state.execution.controller_id == "legacy-controller"
    # The legacy deadline becomes the envelope expiry; identity is lifted off the task.
    assert state.execution.expiry_time == _LEGACY_EXPIRY


def test_legacy_controller_state_loads_from_kv_json():
    """A full ControllerState snapshot persisted in the old shape deserialises without error."""
    legacy = {
        "enable_state": {"enabled": True},
        "operating_state": {"current": "operate"},
        "execution_state": {"executing": True, "task": dict(_LEGACY_TASK)},
    }

    state = ControllerState.model_validate_json(json.dumps(legacy))

    assert isinstance(state.execution_state.execution, TaskExecution)
    assert isinstance(state.execution_state.execution.task, InitTask)
    assert state.execution_state.execution.task_id == _LEGACY_TASK_ID
    assert state.execution_state.execution.expiry_time == _LEGACY_EXPIRY


def test_legacy_task_execution_event_replays_through_registry():
    """A legacy event on the durable stream is upgraded when narrowed via the Event registry."""
    payload = {
        "event_model": "TaskExecutionState",
        "executing": True,
        "task": dict(_LEGACY_TASK),
    }

    event = Event.model_validate_json(json.dumps(payload))

    assert isinstance(event, TaskExecutionState)
    assert isinstance(event.execution, TaskExecution)
    assert isinstance(event.execution.task, InitTask)
    assert event.execution.task_id == _LEGACY_TASK_ID


def test_current_task_execution_state_roundtrips_unchanged():
    """Current-shape state passes through the migration validator as a no-op."""
    execution = TaskExecution(
        task=InitTask(), task_id=uuid.uuid7(), controller_id="ctl"
    )
    state = TaskExecutionState(executing=True, execution=execution)

    reloaded = TaskExecutionState.model_validate(state.model_dump())

    assert isinstance(reloaded.execution, TaskExecution)
    assert isinstance(reloaded.execution.task, InitTask)
    assert reloaded.execution.task_id == execution.task_id


def test_pre_rename_task_envelope_migrates_to_execution():
    """State persisted after the split but before `task` was renamed loads onto `execution`."""
    execution = TaskExecution(
        task=InitTask(), task_id=uuid.uuid7(), controller_id="ctl"
    )
    legacy = {"executing": True, "task": execution.model_dump()}

    state = TaskExecutionState.model_validate(legacy)

    assert isinstance(state.execution, TaskExecution)
    assert isinstance(state.execution.task, InitTask)
    assert state.execution.task_id == execution.task_id
