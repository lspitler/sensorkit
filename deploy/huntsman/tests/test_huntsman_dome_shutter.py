"""Musca shutter open/close/abort against a fake transport (no serial, no NATS).

Run from /opt/sensorkit:
    .venv/bin/python -m pytest -p no:cacheprovider deploy/huntsman/tests -q
"""

from __future__ import annotations

import asyncio
import sys

import pytest

hd = sys.modules["huntsman_dome_under_test"]  # loaded once by conftest.py


class FakeMusca:
    """Stands in for MuscaTransport, driving shutter state as the Musca would.

    Shutter_open runs for `open_s`, then reaches Open, or PartOpen if `stall`
    (the firmware gives up). Shutter_abort stops motion at PartOpen. As the
    TCL monitor suggests, Shutter_close is ignored while Opening.
    """

    def __init__(self, shutter: hd._MuscaShutter, *, open_s=0.3, close_s=0.1, stall=False):
        self.shutter = shutter
        self.open_s, self.close_s, self.stall = open_s, close_s, stall
        self.connected = True
        self.sent: list[str] = []
        self._motion: asyncio.Task | None = None

    async def send(self, cmd: str) -> bool:
        self.sent.append(cmd)
        match cmd:
            case hd.OPEN_CMD:
                self._move("Opening", self.open_s, "PartOpen" if self.stall else "Open")
            case hd.CLOSE_CMD if self.shutter.shutter != "Opening":
                self._move("Closing", self.close_s, "Closed")
            case hd.ABORT_CMD:
                self._stop()
                self._set("PartOpen")
        return True

    def _set(self, state: str):
        self.shutter._apply_status(("Shutter", state))

    def _stop(self):
        if self._motion is not None:
            self._motion.cancel()

    def _move(self, moving: str, duration: float, end: str):
        self._stop()

        async def run():
            # Musca reports motion a beat after the command, like the real 5 s poll.
            await asyncio.sleep(0.02)
            self._set(moving)
            await asyncio.sleep(duration)
            self._set(end)

        self._motion = asyncio.create_task(run())


def make(**fake_kw) -> tuple[hd._MuscaShutter, FakeMusca]:
    config = hd.HuntsmanDomeConfig(shutter_timeout=1.0)
    shutter = hd._MuscaShutter(config)
    shutter.battery_voltage = 13.0
    shutter._apply_status(("Shutter", "Closed"))
    shutter.transport = FakeMusca(shutter, **fake_kw)
    return shutter, shutter.transport


@pytest.fixture(autouse=True)
def fast_resend(monkeypatch):
    """open() waits 5 s before checking whether to resend; shrink that to 50 ms."""
    real_sleep = asyncio.sleep

    async def sleep(delay, *args, **kwargs):
        return await real_sleep(0.05 if delay == 5.0 else delay, *args, **kwargs)

    monkeypatch.setattr(hd.asyncio, "sleep", sleep)


async def until(predicate, timeout=1.0):
    async with asyncio.timeout(timeout):
        while not predicate():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_open_succeeds_and_keeps_heartbeat():
    shutter, fake = make()

    await shutter.open()

    assert shutter.shutter == "Open"
    assert shutter._keep_open
    assert not shutter._open_commanded
    assert fake.sent == [hd.OPEN_CMD]


@pytest.mark.asyncio
async def test_stalled_open_times_out_with_reason():
    shutter, _ = make(stall=True, open_s=0.1)

    with pytest.raises(
        TimeoutError, match=r"did not reach Open within 1s \(last state: PartOpen\)"
    ):
        await shutter.open()

    assert not shutter._open_commanded


@pytest.mark.asyncio
async def test_close_preempts_stalled_open():
    """Last night: a close waited ~66 s behind a stalled open. It must not wait at all."""
    shutter, fake = make(stall=True, open_s=10.0)
    opening = asyncio.create_task(shutter.open())
    await until(lambda: shutter.shutter == "Opening")

    loop = asyncio.get_running_loop()
    start = loop.time()
    closing = asyncio.create_task(shutter.close())
    await until(lambda: hd.CLOSE_CMD in fake.sent)
    waited = loop.time() - start
    await closing

    assert shutter.shutter == "Closed"
    assert waited < 0.1, f"close waited {waited:.2f}s behind the open"
    # Abort first: the Musca would ignore a close while Opening.
    assert fake.sent == [hd.OPEN_CMD, hd.ABORT_CMD, hd.CLOSE_CMD]
    with pytest.raises(hd.ShutterOpenSuperseded):
        await opening
    assert not shutter._keep_open


@pytest.mark.asyncio
async def test_close_right_after_open_sent_still_aborts():
    """Status lags the command, so `shutter` can still read Closed while it moves."""
    shutter, fake = make(open_s=10.0)
    opening = asyncio.create_task(shutter.open())
    await until(lambda: hd.OPEN_CMD in fake.sent)
    assert shutter.shutter == "Closed"  # Musca has not reported Opening yet

    await shutter.close()

    assert fake.sent == [hd.OPEN_CMD, hd.ABORT_CMD, hd.CLOSE_CMD]
    assert shutter.shutter == "Closed"
    with pytest.raises(hd.ShutterOpenSuperseded):
        await opening


@pytest.mark.asyncio
async def test_close_cancels_open_still_waiting_for_lock():
    shutter, fake = make(open_s=10.0)
    first = asyncio.create_task(shutter.open())
    await until(lambda: shutter.shutter == "Opening")
    queued = asyncio.create_task(shutter.open())  # blocked on _op_lock
    await asyncio.sleep(0.05)

    await shutter.close()

    for task in (first, queued):
        with pytest.raises(hd.ShutterOpenSuperseded):
            await task
    # The queued open never sent anything once the lock freed.
    assert fake.sent.count(hd.OPEN_CMD) == 1
    assert shutter.shutter == "Closed"


@pytest.mark.asyncio
async def test_open_after_close_is_not_superseded():
    shutter, _ = make()
    await shutter.close()

    await shutter.open()

    assert shutter.shutter == "Open"


@pytest.mark.asyncio
async def test_abort_stops_open_in_flight():
    shutter, fake = make(open_s=10.0)
    opening = asyncio.create_task(shutter.open())
    await until(lambda: shutter.shutter == "Opening")

    async with asyncio.timeout(0.5):  # dispatch waits only ABORT_TIMEOUT_S
        await shutter.abort()

    assert fake.sent == [hd.OPEN_CMD, hd.ABORT_CMD]
    assert shutter.shutter == "PartOpen"
    assert not shutter._keep_open
    with pytest.raises(hd.ShutterOpenSuperseded):
        await opening


@pytest.mark.asyncio
async def test_abort_after_caller_cancelled_open_still_stops_motion():
    """The agent cancels its call, then sends Abort: the open task is gone by then."""
    shutter, fake = make(open_s=10.0)
    opening = asyncio.create_task(shutter.open())
    await until(lambda: shutter.shutter == "Opening")
    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening
    assert not shutter._open_tasks

    await shutter.abort()

    assert fake.sent == [hd.OPEN_CMD, hd.ABORT_CMD]
    assert shutter.shutter == "PartOpen"


@pytest.mark.asyncio
async def test_abort_leaves_open_shutter_and_heartbeat_alone():
    """Abort follows any interrupted call (e.g. a slew) and must not drop the dome."""
    shutter, fake = make()
    await shutter.open()

    await shutter.abort()

    assert shutter._keep_open
    assert shutter.shutter == "Open"
    assert fake.sent == [hd.OPEN_CMD]


@pytest.mark.asyncio
async def test_abort_never_interrupts_close():
    shutter, fake = make(close_s=0.3)
    shutter._apply_status(("Shutter", "Open"))
    closing = asyncio.create_task(shutter.close())
    await until(lambda: shutter.shutter == "Closing")

    async with asyncio.timeout(0.5):
        await shutter.abort()
    await closing

    assert fake.sent == [hd.CLOSE_CMD]
    assert shutter.shutter == "Closed"


@pytest.mark.asyncio
async def test_cancelling_open_caller_cancels_the_work():
    shutter, fake = make(stall=True, open_s=10.0)
    opening = asyncio.create_task(shutter.open())
    await until(lambda: shutter.shutter == "Opening")

    opening.cancel()
    with pytest.raises(asyncio.CancelledError):
        await opening

    assert not shutter._op_lock.locked()
    assert not shutter._open_tasks


# ── Through SensorKit's command plumbing (in-memory backend) ──────────


async def _shutter_device(kit, shutter: hd._MuscaShutter):
    """Register the dome's shutter handlers on a test device.

    Only the Musca side is wired: dome_abort's rotation stop would reach for
    TheSkyX, which may be live on this machine.
    """
    from sensorkit.api import Abort
    from sensorkit.std.enclosure import CloseEnclosure, OpenEnclosure

    service = await kit.register_service("test_dome_service", "0.1.0")
    dev = await service.register_device("TestDome")

    @dev.command_handler(OpenEnclosure)
    async def on_open(cmd):
        await shutter.open()

    @dev.command_handler(CloseEnclosure)
    async def on_close(cmd):
        await shutter.close()

    @dev.command_handler(Abort)
    async def on_abort(cmd):
        await shutter.abort()

    return service, kit.device("TestDome")


@pytest.mark.asyncio
async def test_agent_cancel_then_abort_releases_shutter(kit):
    """Last night's sequence end to end: cancelled open call, then Abort, then close.

    Cancelling the caller does not retract the request, so the open handler keeps
    running until Abort reaches it. Close then proceeds without waiting.
    """
    from sensorkit.api import Abort
    from sensorkit.std.enclosure import CloseEnclosure, OpenEnclosure

    shutter, fake = make(stall=True, open_s=10.0)
    service, client = await _shutter_device(kit, shutter)
    try:
        call = asyncio.create_task(client.command(OpenEnclosure()).run_to_completion())
        await until(lambda: shutter.shutter == "Opening")

        call.cancel()  # what the agent's controller does on interruption
        with pytest.raises(asyncio.CancelledError):
            await call
        await asyncio.sleep(0.1)
        assert shutter._op_lock.locked(), "cancelling the caller alone must not reach the device"

        async with asyncio.timeout(1.0):  # well inside dispatch's ABORT_TIMEOUT_S
            await client.command(Abort())
        await until(lambda: not shutter._op_lock.locked(), timeout=0.5)

        async with asyncio.timeout(1.0):
            await client.command(CloseEnclosure())

        assert shutter.shutter == "Closed"
        assert fake.sent == [hd.OPEN_CMD, hd.ABORT_CMD, hd.CLOSE_CMD]
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_superseded_open_call_fails_with_reason(kit):
    """The open's caller gets a failure that says why, rather than a hang."""
    from sensorkit.backend.request import CallError
    from sensorkit.std.enclosure import CloseEnclosure, OpenEnclosure

    shutter, _ = make(stall=True, open_s=10.0)
    service, client = await _shutter_device(kit, shutter)
    try:
        call = asyncio.create_task(client.command(OpenEnclosure()).run_to_completion())
        await until(lambda: shutter.shutter == "Opening")

        async with asyncio.timeout(1.0):
            await client.command(CloseEnclosure())

        with pytest.raises(CallError, match="superseded"):
            async with asyncio.timeout(1.0):
                await call
    finally:
        await service.shutdown()
