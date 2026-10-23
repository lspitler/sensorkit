# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import json
from datetime import datetime, timedelta

import pytest
import pytest_asyncio
import uuid_utils.compat as uuid

from sensorkit.backend.lease import Lease
from sensorkit.backend.request import CallError, declare_request
from sensorkit.core.controller import (
    AbortResponseMessage,
    ExecuteRequestMessage,
    ExecuteResponseMessage,
    abort_task_request,
    execute_task_request,
)
from sensorkit.core.executor import TaskOutcomeUnknown
from sensorkit.core.impl.controller import ControllerImpl
from sensorkit.core.impl.entity import EntityImpl
from sensorkit.core.impl.program import ProgramImpl
from sensorkit.core.program import ProgramOffering, ProgramState, ProgramTaskingState
from sensorkit.core.task import CollectTask, TaskExecution


class RecoverableController(ControllerImpl):
    """Controller whose abort endpoint can temporarily fail."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.allow_abort = asyncio.Event()
        self.allow_abort.set()
        self.abort_failed = asyncio.Event()
        self.abort_requests: list[uuid.UUID | None] = []

    async def _abort_request(self, requested, call):
        self.abort_requests.append(requested.task_id)

        if not self.allow_abort.is_set():
            self.abort_failed.set()
            raise RuntimeError("Controller connection lost")

        await super()._abort_request(requested, call)


async def tasking_ended(client) -> bool:
    return not (await client.kv_get_model(ProgramState)).active_state.active


@pytest_asyncio.fixture
async def controller_endpoint(kit, service_context):
    endpoint = await service_context.register_impl(
        EntityImpl.for_service_context(service_context, "endpoint"), acquire_lease=False
    )
    kv = kit.backend.key_value(endpoint.entity)

    async with await Lease.acquire(kv, "EntityLease", 60, None) as lease:
        yield endpoint, lease


@pytest_asyncio.fixture
async def leased_controller(kit, service_context):
    controller = await service_context.register_impl(
        RecoverableController.for_service_context(service_context, "controller"),
        acquire_lease=False,
    )
    kv = kit.backend.key_value(controller.entity)

    async with await Lease.acquire(kv, "EntityLease", 60, None) as lease:
        yield controller, lease


@pytest.mark.asyncio
async def test_program_offering(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        program = await sc.register_program("myprogram")

    ready = asyncio.Event()
    now = datetime.now().replace(microsecond=0)
    offer = (now, now + timedelta(hours=1))
    program_client = kit.program("myprogram")

    async def monitor_offering():
        stream = await program_client.monitor(ProgramOffering)
        ready.set()

        async for _, data in stream:
            offered = data.offer_windows
            assert len(offered) == 1
            assert offered[0].begin == offer[0]
            assert offered[0].end == offer[1]
            break

    task = asyncio.create_task(monitor_offering())

    async with asyncio.timeout(2.0):
        await program_client.enable("any")
        await ready.wait()

        program.clear_offers()
        program.add_offer(*offer)
        await program.publish_offers()
        await task


@pytest.mark.asyncio
async def test_program_tasking(kit):
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")
        program = await sc.register_program("myprogram")

    task_obj = CollectTask(task_id=uuid.uuid1(), controller_id="mycontroller")
    task_ran = False
    enabled = asyncio.Event()
    disabled = asyncio.Event()
    done = asyncio.Event()

    @program.on_enable
    async def program_enable():
        assert ProgramImpl.current.get() is program
        enabled.set()

    @program.on_disable
    async def program_disable():
        assert ProgramImpl.current.get() is program
        disabled.set()

    @program.task_factory
    async def program_tasker():
        assert ProgramImpl.current.get() is program

        if not done.is_set():
            assert not task_ran
            await (yield task_obj)
            assert task_ran
            done.set()

    @controller.task_handler(CollectTask)
    async def run_task(task: CollectTask):
        assert ControllerImpl.current.get() is controller
        assert task == task_obj
        nonlocal task_ran
        task_ran = True

    ready = asyncio.Event()
    activated = asyncio.Event()
    stopping = asyncio.Event()
    deactivated = asyncio.Event()
    program_client = kit.program("myprogram")

    async def state_monitor():
        events = await program_client.tasking_change_events()
        ready.set()

        async for event in events:
            if event.active:
                if event.stopping:
                    assert not stopping.is_set()
                    stopping.set()
                else:
                    assert not activated.is_set()
                    activated.set()
            else:
                assert not deactivated.is_set()
                deactivated.set()

    _t = asyncio.create_task(state_monitor())

    async with asyncio.timeout(2.0):
        await kit.controller("mycontroller").enable()
        await ready.wait()
        await program_client.enable("mycontroller")
        await enabled.wait()
        await program_client.start_tasking()
        await activated.wait()
        await done.wait()
        await program_client.stop_tasking()
        await stopping.wait()
        await deactivated.wait()
        await program_client.disable()
        await disabled.wait()


@pytest.mark.asyncio
async def test_program_publishes_tasking_state(kit):
    """While tasking, the program publishes a ProgramTaskingState carrying the live execution,
    then clears it (executing_task=None) once the task completes."""
    async with asyncio.timeout(1.0):
        sc = await kit.register_service("testservice", "0.1.0")
        controller = await sc.register_controller("mycontroller")
        program = await sc.register_program("myprogram")

    task_obj = CollectTask()
    done = asyncio.Event()

    @program.task_factory
    async def program_tasker():
        if not done.is_set():
            await (yield task_obj)
            done.set()

    @controller.task_handler(CollectTask)
    async def run_task(task: CollectTask):
        pass

    ready = asyncio.Event()
    cleared = asyncio.Event()
    program_client = kit.program("myprogram")
    executing: asyncio.Future[ProgramTaskingState] = asyncio.get_running_loop().create_future()

    async def monitor():
        stream = await program_client.monitor_event(ProgramTaskingState)
        ready.set()

        async for event in stream:
            if event.executing_task is not None:
                if not executing.done():
                    executing.set_result(event)
            elif executing.done():
                # The clear must follow an executing state, never precede it.
                cleared.set()
                break

    monitor_task = asyncio.create_task(monitor())

    async with asyncio.timeout(2.0):
        await kit.controller("mycontroller").enable()
        await ready.wait()
        await program_client.enable("mycontroller")
        await program_client.start_tasking()

        event = await executing
        execution = event.executing_task
        # The published state carries the full TaskExecution envelope, not just an id.
        assert execution.task == task_obj
        assert execution.task_id is not None
        assert execution.controller_id == "mycontroller"

        await done.wait()
        await cleared.wait()
        await program_client.stop_tasking()

    await monitor_task


@pytest.mark.asyncio
async def test_program_reuses_task_instance(kit, service_context):
    controller = await service_context.register_controller("mycontroller")
    program = await service_context.register_program("myprogram")

    task_obj = CollectTask()
    executed: list[uuid.UUID] = []
    done = asyncio.Event()

    @program.task_factory
    async def program_tasker():
        if not done.is_set():
            await (yield task_obj)

            if len(executed) == 2:
                done.set()

    async def run_task(execution: TaskExecution[CollectTask]):
        assert execution.task == task_obj
        executed.append(execution.task_id)

    controller.add_task_handler(CollectTask, execution_callback=run_task)

    program_client = kit.program("myprogram")

    async with asyncio.timeout(2.0):
        await kit.controller("mycontroller").enable()
        await program_client.enable("mycontroller")
        await program_client.start_tasking()
        await done.wait()
        await program_client.stop_tasking()

    assert len(set(executed)) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_program_abort_waits_for_controller_cleanup(kit, service_context, accepted):
    """Aborting tasking returns only after the controller has ended the running task."""
    controller = await service_context.register_controller("mycontroller")
    program = await service_context.register_program("myprogram")
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()

    @controller.task_handler(CollectTask)
    async def collect(task):
        entered.set()

        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    @program.task_factory
    async def factory():
        yield CollectTask()

    program_client = kit.program("myprogram")
    await kit.controller("mycontroller").enable()
    await program_client.enable("mycontroller")
    abort = None

    try:
        async with asyncio.timeout(5.0):
            await program_client.start_tasking()
            await entered.wait()

            if accepted:
                state = await program_client.kv_get_model(ProgramState)

                while state.tasking_state.executing_task is None:
                    await asyncio.sleep(0.01)
                    state = await program_client.kv_get_model(ProgramState)

            abort = asyncio.create_task(program_client.abort_tasking())
            await cleaning.wait()

        done, _ = await asyncio.wait((abort,), timeout=0.2)

        assert not done
    finally:
        release.set()

        async with asyncio.timeout(5.0):
            if abort is not None:
                await abort
            else:
                await program_client.abort_tasking()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["rejected", "remote", "unexpected"])
async def test_dispatch_failure_ends_tasking(
    kit, service_context, controller_endpoint, until, failure
):
    """A failed dispatch ends tasking, and another tasking session can follow."""
    entity_impl, _ = controller_endpoint
    requests = 0
    finished = asyncio.Event()

    if failure == "unexpected":
        request = declare_request(
            "execute_task", message=ExecuteRequestMessage, response=ExecuteResponseMessage
        )

        async def execute(msg):
            nonlocal requests
            requests += 1
            return ExecuteResponseMessage()
    else:
        request = execute_task_request

        async def execute(msg, call):
            nonlocal requests
            requests += 1

            if failure == "rejected":
                call.reject(reason="Unavailable")
            else:
                raise RuntimeError("Admission failed")

    await entity_impl.handle_request(request, execute)
    program = await service_context.register_program("dispatch-program")

    @program.task_factory
    async def factory():
        try:
            yield CollectTask()
        finally:
            finished.set()

    client = kit.program("dispatch-program")
    await client.enable(str(entity_impl.entity))

    async with asyncio.timeout(5.0):
        await client.start_tasking()
        await finished.wait()

        assert requests == 1

        if failure == "rejected":
            await client.stop_tasking()
            await client.start_tasking()

            while (await client.kv_get_model(ProgramState)).active_state.active:
                await asyncio.sleep(0)

            assert requests == 2
        else:
            # Without acceptance, tasking ends in error rather than waiting on the controller.
            await until(lambda: tasking_ended(client))
            state = await client.kv_get_model(ProgramState)
            assert state.active_state.origin == "error"

            await client.start_tasking()
            await until(lambda: requests == 2)


@pytest.mark.asyncio
async def test_abort_before_first_execution_event(kit, service_context, controller_endpoint):
    """An abort before the first execution event waits for the delayed end event."""
    endpoint, _ = controller_endpoint
    execution = TaskExecution(
        task=CollectTask(), task_id=uuid.uuid7(), controller_id=str(endpoint.entity)
    )
    admitted = asyncio.Event()
    release = asyncio.Event()
    errors: list[BaseException] = []

    async def execute(msg, call):
        call.accept(response=ExecuteResponseMessage(execution=execution))
        admitted.set()
        await release.wait()
        await call.fail("Aborted")

    async def abort_task(msg, call):
        assert msg.task_id == execution.task_id
        call.accept(response=AbortResponseMessage(aborting=True, task_id=msg.task_id))
        await call.succeed(result=None)

    await endpoint.handle_request(execute_task_request, execute)
    await endpoint.handle_request(abort_task_request, abort_task)
    program = await service_context.register_program("admission-abort-program")

    @program.task_factory
    async def factory():
        try:
            yield CollectTask()
        except BaseException as error:
            errors.append(error)

    client = kit.program(program.entity)
    await client.enable(str(endpoint.entity))
    abort = None

    try:
        async with asyncio.timeout(5.0):
            await client.start_tasking()
            loop = program._task_loop
            await admitted.wait()
            abort = asyncio.create_task(client.abort_tasking())

            # The abort waits for the execution's end event.
            done, _ = await asyncio.wait((abort,), timeout=0.1)
            assert not done

            release.set()
            await abort

            assert len(errors) == 1
            assert isinstance(errors[0], asyncio.CancelledError)
            assert loop.finalized
    finally:
        release.set()

        async with asyncio.timeout(5.0):
            if abort is not None:
                await abort

            await client.abort_tasking()


@pytest.mark.asyncio
async def test_abort_interrupts_dispatch_error_cleanup(kit, service_context, controller_endpoint):
    """A user abort interrupts factory cleanup after an unresolved dispatch."""
    endpoint, _ = controller_endpoint
    execution = TaskExecution(
        task=CollectTask(), task_id=uuid.uuid7(), controller_id=str(endpoint.entity)
    )
    cleaning = asyncio.Event()
    interrupted = asyncio.Event()
    release = asyncio.Event()
    request = declare_request(
        "execute_task", message=ExecuteRequestMessage, response=ExecuteResponseMessage
    )

    async def execute(msg):
        return ExecuteResponseMessage(execution=execution)

    await endpoint.handle_request(request, execute)
    program = await service_context.register_program("error-cleanup-program")

    @program.task_factory
    async def factory():
        try:
            yield CollectTask()
        except TaskOutcomeUnknown:
            cleaning.set()

            try:
                await release.wait()
            except asyncio.CancelledError:
                interrupted.set()
                raise

    client = kit.program(program.entity)
    await client.enable(str(endpoint.entity))

    try:
        async with asyncio.timeout(5.0):
            await client.start_tasking()
            await cleaning.wait()
            loop = program._task_loop

            # The dispatch was never confirmed, so the abort cannot report the work ended.
            with pytest.raises(CallError):
                await client.abort_tasking()

            assert interrupted.is_set()
            assert loop.finalized
            assert loop.unresolved
    finally:
        release.set()

        async with asyncio.timeout(5.0):
            await client.abort_tasking()


@pytest.mark.asyncio
async def test_unanswered_dispatch_ends_tasking(kit, service_context, leased_controller, until):
    """A dispatch reply timeout ends tasking and allows another tasking session to start."""
    controller_impl, _ = leased_controller
    program = await service_context.register_program("cleanup-program")
    release = asyncio.Event()

    @controller_impl.task_handler(CollectTask)
    async def collect(task):
        await release.wait()

    @program.task_factory
    async def factory():
        yield CollectTask()

    client = kit.program("cleanup-program")
    controller = kit.controller(controller_impl.entity)
    await client.enable(str(controller_impl.entity))

    try:
        async with asyncio.timeout(30.0):
            # Delay admission until the dispatch reply times out.
            async with controller_impl._state.updating(controller_impl):
                await client.start_tasking()
                loop = program._task_loop
                await until(lambda: tasking_ended(client), timeout=25.0)
                state = await client.kv_get_model(ProgramState)

                assert state.active_state.origin == "error"
                assert loop.finalized
                assert loop.unresolved
    finally:
        release.set()

        with contextlib.suppress(CallError):
            async with asyncio.timeout(5.0):
                await controller.abort_task()


@pytest.mark.asyncio
@pytest.mark.parametrize("confirmation", ["abort", "completion", "successor"])
async def test_cleanup_retries_and_allows_restart(
    kit, service_context, leased_controller, confirmation
):
    """Loop finalization waits for the task's end event, with or without an accepted abort."""
    controller, _ = leased_controller
    program = await service_context.register_program("retry-program")
    entered = asyncio.Event()
    release = asyncio.Event()

    @controller.task_handler(CollectTask)
    async def collect(task):
        entered.set()
        await release.wait()

    @program.task_factory
    async def factory():
        await (yield CollectTask())

    client = kit.program(program.entity)
    await kit.controller(controller.entity).enable()
    await client.enable(str(controller.entity))
    graceful = abort = retry = successor = None

    try:
        async with asyncio.timeout(5.0):
            await client.start_tasking()
            await entered.wait()
            loop = program._task_loop
            assert not loop.finalized
            controller.allow_abort.clear()
            graceful = asyncio.create_task(client.stop_tasking())

            while not (await client.kv_get_model(ProgramState)).active_state.stopping:
                await asyncio.sleep(0)

            with pytest.raises(CallError):
                await client.start_tasking()

            abort = asyncio.create_task(client.abort_tasking())
            await controller.abort_failed.wait()

            assert program._task_loop is loop
            assert not loop.finalized

            with pytest.raises(CallError):
                await client.start_tasking()

            match confirmation:
                case "abort":
                    # Retry the abort request that the controller previously failed to handle.
                    controller.allow_abort.set()
                    retry = asyncio.create_task(client.abort_tasking())
                case "completion":
                    release.set()
                case "successor":
                    entered.clear()
                    successor = kit.controller(controller.entity).execute_task(
                        CollectTask(), interrupt=True
                    )
                    await successor.invoke()
                    await entered.wait()

            await asyncio.gather(*(stop for stop in (graceful, abort, retry) if stop is not None))
            assert loop.finalized
            assert program._task_loop is None

            if successor is not None:
                assert not successor.done()
                release.set()
                await successor

            entered.clear()
            release.clear()
            await client.start_tasking()
            await entered.wait()
            assert program._task_loop is not loop
    finally:
        controller.allow_abort.set()
        release.set()

        async with asyncio.timeout(5.0):
            await client.abort_tasking()

            if graceful is not None:
                await graceful

            if abort is not None:
                await abort

            if retry is not None:
                await retry

            if successor is not None:
                with contextlib.suppress(CallError):
                    await successor


@pytest.mark.asyncio
async def test_controller_lease_absence_preserves_error_origin(
    kit, service_context, leased_controller
):
    """A factory error remains the tasking stop reason when controller cleanup fails."""
    controller, lease = leased_controller
    program = await service_context.register_program("error-origin-program")
    fail = asyncio.Event()

    @program.task_factory
    async def factory():
        await fail.wait()
        raise RuntimeError("Task factory failed")

    client = kit.program(program.entity)
    await client.enable(str(controller.entity))
    await client.start_tasking()
    loop = program._task_loop
    finalizer = program._task_finalizer

    try:
        async with asyncio.timeout(5.0):
            await lease.expire()
            fail.set()
            await asyncio.shield(finalizer)

            state = await client.kv_get_model(ProgramState)
            assert not state.active_state.active
            assert state.active_state.origin == "error"
            assert loop.finalized
            assert not loop.stop_requested
    finally:
        fail.set()

        async with asyncio.timeout(5.0):
            await client.abort_tasking()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["factory", "execution"])
async def test_abort_does_not_repeat_for_released_work(
    kit, service_context, leased_controller, phase
):
    """Producer teardown does not repeatedly abort work that already ended."""
    controller, _ = leased_controller
    program = await service_context.register_program("released-work-program")
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    emitted = False

    @controller.task_handler(CollectTask)
    async def collect(task):
        if phase == "execution":
            await asyncio.Event().wait()

    @program.task_factory
    async def factory():
        nonlocal emitted

        if phase == "factory" and emitted:
            entered.set()

            try:
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await release.wait()

            return

        emitted = True
        execution = yield CollectTask()

        if phase == "execution":
            entered.set()

            try:
                await execution
            finally:
                cleaning.set()
                await release.wait()
        else:
            await execution

    client = kit.program(program.entity)
    await kit.controller(controller.entity).enable()
    await client.enable(str(controller.entity))
    abort = None

    try:
        async with asyncio.timeout(5.0):
            await client.start_tasking()
            await entered.wait()
            loop = program._task_loop
            abort = asyncio.create_task(client.abort_tasking())
            await cleaning.wait()
            done, _ = await asyncio.wait((abort,), timeout=0.1)

            assert not done
            assert len(controller.abort_requests) == (1 if phase == "execution" else 0)

            release.set()
            await abort
            assert loop.finalized
    finally:
        release.set()

        async with asyncio.timeout(5.0):
            if abort is not None:
                await abort

            await client.abort_tasking()


@pytest.mark.asyncio
async def test_target_change_waits_for_cleanup(kit, service_context, leased_controller):
    """Changing controllers waits for the current controller task's cleanup to finish."""
    original, _ = leased_controller
    replacement = await service_context.register_controller("replacement")
    program = await service_context.register_program("switch-program")
    entered = asyncio.Event()
    cleaning = asyncio.Event()
    release = asyncio.Event()
    switched = asyncio.Event()

    @original.task_handler(CollectTask)
    async def old_collect(task):
        entered.set()

        try:
            await asyncio.Event().wait()
        finally:
            cleaning.set()
            await release.wait()

    @replacement.task_handler(CollectTask)
    async def new_collect(task):
        switched.set()

    @program.task_factory
    async def factory():
        if not switched.is_set():
            yield CollectTask()

    client = kit.program(program.entity)
    await kit.controller(original.entity).enable()
    await kit.controller(replacement.entity).enable()
    await client.enable(str(original.entity))
    switch = None

    try:
        async with asyncio.timeout(5.0):
            await client.start_tasking()
            await entered.wait()
            loop = program._task_loop
            switch = asyncio.create_task(client.enable(str(replacement.entity)))
            await cleaning.wait()

            assert not loop.finalized
            assert not switch.done()
            assert not switched.is_set()
            assert (await client.kv_get_model(ProgramState)).enable_state.controller == str(
                original.entity
            )

            with pytest.raises(CallError):
                await client.start_tasking()

            release.set()
            await switch
            assert loop.finalized
            await client.start_tasking()
            await switched.wait()
            await client.stop_tasking()
    finally:
        release.set()

        async with asyncio.timeout(5.0):
            if switch is not None:
                await switch

            await client.abort_tasking()


@pytest.mark.asyncio
@pytest.mark.parametrize("previous", ["idle", "tasking"])
async def test_delayed_stop_cannot_stop_replacement_loop(
    kit, service_context, leased_controller, previous
):
    """A delayed stop cannot affect a tasking loop started after the request."""
    controller, _ = leased_controller
    program = await service_context.register_program("delayed-stop-program")
    entered = asyncio.Event()

    @controller.task_handler(CollectTask)
    async def collect(task):
        entered.set()
        await asyncio.Event().wait()

    @program.task_factory
    async def factory():
        await (yield CollectTask())

    client = kit.program(program.entity)
    await kit.controller(controller.entity).enable()
    await client.enable(str(controller.entity))
    stop = None

    try:
        async with asyncio.timeout(5.0):
            if previous == "tasking":
                await client.start_tasking()
                await entered.wait()

            stop = program._stop_loop(program._task_loop, abort=True)
            await client.abort_tasking()
            entered.clear()
            await client.start_tasking()
            await entered.wait()
            replacement = program._task_loop

            await stop
            stop = None

            assert program._task_loop is replacement
            assert replacement.running
            state = await client.kv_get_model(ProgramState)
            assert state.active_state.active and not state.active_state.stopping
    finally:
        if stop is not None:
            stop.close()

        async with asyncio.timeout(5.0):
            await client.abort_tasking()


@pytest.mark.asyncio
async def test_enable_hook_gates_start_without_blocking_abort(kit, service_context):
    """An enable hook blocks new tasking while abort requests remain available."""
    controller = await service_context.register_controller("hook-controller")
    program = await service_context.register_program("hook-program")
    entered = asyncio.Event()
    release = asyncio.Event()
    task_entered = asyncio.Event()

    @program.on_enable
    async def enable_hook():
        entered.set()
        await release.wait()

    @controller.task_handler(CollectTask)
    async def collect(task):
        task_entered.set()
        await asyncio.Event().wait()

    @program.task_factory
    async def factory():
        await (yield CollectTask())

    client = kit.program(program.entity)
    await kit.controller(controller.entity).enable()
    enable = asyncio.create_task(client.enable(str(controller.entity)))

    try:
        async with asyncio.timeout(5.0):
            await entered.wait()

            with pytest.raises(CallError):
                await client.start_tasking()

            await client.abort_tasking()
            assert not enable.done()
            assert not task_entered.is_set()

            release.set()
            await enable
            await client.start_tasking()
            await task_entered.wait()
    finally:
        release.set()

        async with asyncio.timeout(5.0):
            await enable
            await client.abort_tasking()


# Legacy state migration: versions before the Task/TaskExecution split stored the program's
# executing task under a `tasking_status` key (a `ProgramTaskingStatus` carrying only
# `executing_task_id`). `mode="before"` validators on ProgramState and ProgramTaskingState upgrade
# that shape so old state from a NATS broker still loads.
def test_legacy_tasking_status_migrates_to_tasking_state():
    """A snapshot with the renamed `tasking_status` key loads as an idle `tasking_state`."""
    legacy = {
        "enable_state": {"enabled": True, "controller": "mycontroller"},
        "active_state": {"active": False, "origin": "init"},
        "tasking_status": {"executing_task_id": str(uuid.uuid4())},
    }

    state = ProgramState.model_validate(legacy)

    # A bare legacy id cannot rehydrate a TaskExecution envelope, so the program loads as idle.
    assert isinstance(state.tasking_state, ProgramTaskingState)
    assert state.tasking_state.executing_task is None


def test_legacy_program_state_loads_from_kv_json():
    """A full legacy ProgramState snapshot deserialises from JSON without error."""
    legacy = {
        "enable_state": {"enabled": True, "controller": "mycontroller"},
        "active_state": {"active": False, "origin": "init"},
        "tasking_status": {"executing_task_id": None},
    }

    state = ProgramState.model_validate_json(json.dumps(legacy))

    assert state.tasking_state.executing_task is None


def test_program_state_missing_tasking_key_defaults_to_idle():
    """A snapshot predating the tasking field entirely is treated as idle rather than failing."""
    legacy = {
        "enable_state": {"enabled": False},
        "active_state": {"active": False, "origin": "init"},
    }

    state = ProgramState.model_validate(legacy)

    assert state.tasking_state.executing_task is None


def test_current_program_state_roundtrips_unchanged():
    """Current-shape state passes through the migration validators as a no-op."""
    state = ProgramState.model_validate(
        {
            "enable_state": {"enabled": True, "controller": "mycontroller"},
            "active_state": {"active": False, "origin": "init"},
            "tasking_state": {"executing_task": None},
        }
    )

    reloaded = ProgramState.model_validate(state.model_dump())

    assert isinstance(reloaded.tasking_state, ProgramTaskingState)
    assert reloaded.tasking_state.executing_task is None
