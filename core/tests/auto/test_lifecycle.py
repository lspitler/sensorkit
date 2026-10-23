# SPDX-License-Identifier: Apache-2.0
"""System tests for controller lifecycle transitions and program tasking.

Cover service registration, lifecycle enablement, initialization, program tasking,
and shutdown using the fake backend.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid

import pytest

from sensorkit.auto.lifecycle import ControllerLifecycle, LifecycleStep
from sensorkit.auto.procedures import StandbyProcedure
from sensorkit.backend.request import CallError
from sensorkit.core.client import SensorKit
from sensorkit.core.controller import ControllerState, InternalControllerState
from sensorkit.core.task import (
    CollectTask,
    InitTask,
    ShutdownTask,
    StandbyTask,
)


async def reports(kit, state: InternalControllerState) -> bool:
    """Return whether ctrl1 reports the requested state with no transition pending."""
    operating = (await kit.controller("ctrl1").kv_get_model(ControllerState)).operating_state
    return operating.current == state and operating.target is None


async def cleanup(lifecycle: ControllerLifecycle, sc):
    """Cancel the lifecycle background task and shut down the service."""
    lifecycle._main_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await lifecycle._main_task
    await sc.shutdown()


@pytest.mark.asyncio
async def test_operate_full_cycle(kit, until):
    """OPERATE demand initializes the controller and starts tasking before SHUTDOWN demand."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program = await sc.register_program("prog1")

    init_ran = asyncio.Event()
    task_ran = asyncio.Event()
    shutdown_ran = asyncio.Event()
    done = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        init_ran.set()

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        task_ran.set()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_ran.set()

    @program.task_factory
    async def factory():
        if not done.is_set():
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            done.set()

    # Enable controller and program.
    await kit.controller("ctrl1").enable()
    prog_client = kit.program("prog1")
    await prog_client.enable("ctrl1")

    # Create and drive the lifecycle.
    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_client)

    async with asyncio.timeout(10.0):
        await init_ran.wait()
        await task_ran.wait()

    # Transition to SHUTDOWN.
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)

    async with asyncio.timeout(15.0):
        await shutdown_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.SHUTDOWN))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_operate_multiple_tasks(kit):
    """Program's task_factory yields multiple tasks in sequence."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program = await sc.register_program("prog1")

    init_ran = asyncio.Event()
    task_count = 0
    all_done = asyncio.Event()
    expected_tasks = 3

    @controller.task_handler(InitTask)
    async def handle_init(task):
        init_ran.set()

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        nonlocal task_count
        task_count += 1

    @program.task_factory
    async def factory():
        nonlocal task_count
        if task_count < expected_tasks:
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            if task_count >= expected_tasks:
                all_done.set()

    await kit.controller("ctrl1").enable()
    prog_client = kit.program("prog1")
    await prog_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_client)

    async with asyncio.timeout(15.0):
        await init_ran.wait()
        await all_done.wait()

    assert task_count == expected_tasks
    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_operate_then_shutdown_interrupts_tasking(kit, until):
    """Changing demand to SHUTDOWN while tasking interrupts the program."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program = await sc.register_program("prog1")

    init_ran = asyncio.Event()
    task_started = asyncio.Event()
    task_can_finish = asyncio.Event()
    shutdown_ran = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        init_ran.set()

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        task_started.set()
        # Wait until signaled to finish (simulates a running task that cooperates with stop).
        await task_can_finish.wait()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_ran.set()

    @program.task_factory
    async def factory():
        yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")

    await kit.controller("ctrl1").enable()
    prog_client = kit.program("prog1")
    await prog_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_client)

    async with asyncio.timeout(10.0):
        await init_ran.wait()
        await task_started.wait()

    # Change demand while the task is still running.
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)
    # Allow the task to finish so stop_tasking() can complete.
    task_can_finish.set()

    async with asyncio.timeout(15.0):
        await shutdown_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.SHUTDOWN))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_standby_cycle(kit, until):
    """STANDBY demand runs the controller's standby task."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")

    standby_ran = asyncio.Event()

    @controller.task_handler(StandbyTask)
    async def handle_standby(task):
        standby_ran.set()

    await kit.controller("ctrl1").enable()

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()
    lifecycle.set_demand_state(InternalControllerState.STANDBY)

    async with asyncio.timeout(10.0):
        await standby_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.STANDBY))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_interrupt_program_aborts_running_task(kit, until):
    """A replacement demand with interrupt=True aborts the current program's running task."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program_a = await sc.register_program("prog_a")
    program_b = await sc.register_program("prog_b")

    task_a_started = asyncio.Event()
    task_a_aborted = asyncio.Event()
    task_b_ran = asyncio.Event()
    done_b = asyncio.Event()
    shutdown_ran = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        pass

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        if not task_a_started.is_set():
            task_a_started.set()
            try:
                # Wait for the controller to abort this task.
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                task_a_aborted.set()
                raise
        else:
            task_b_ran.set()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_ran.set()

    @program_a.task_factory
    async def factory_a():
        yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")

    @program_b.task_factory
    async def factory_b():
        if not done_b.is_set():
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            done_b.set()

    await kit.controller("ctrl1").enable()
    prog_a_client = kit.program("prog_a")
    await prog_a_client.enable("ctrl1")
    prog_b_client = kit.program("prog_b")
    await prog_b_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()

    # Start program A and leave its controller task running.
    lifecycle.set_demand_state(
        InternalControllerState.OPERATE, program=prog_a_client
    )

    async with asyncio.timeout(10.0):
        await task_a_started.wait()

    # The replacement demand requests an abort of program A's running task.
    lifecycle.set_demand_state(
        InternalControllerState.OPERATE, program=prog_b_client, interrupt=True
    )

    async with asyncio.timeout(10.0):
        await task_a_aborted.wait()
        await task_b_ran.wait()

    # Transition to SHUTDOWN for clean teardown.
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)

    async with asyncio.timeout(15.0):
        await shutdown_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.SHUTDOWN))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_non_interrupt_program_waits_for_running_task(kit, until):
    """Without interrupt=True, switching programs waits for the current task to finish gracefully."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program_a = await sc.register_program("prog_a")
    program_b = await sc.register_program("prog_b")

    task_a_started = asyncio.Event()
    task_a_can_finish = asyncio.Event()
    task_a_finished = asyncio.Event()
    task_b_ran = asyncio.Event()
    done_b = asyncio.Event()
    shutdown_ran = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        pass

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        if not task_a_started.is_set():
            task_a_started.set()
            # A graceful stop waits for this task to be released.
            await task_a_can_finish.wait()
            task_a_finished.set()
        else:
            task_b_ran.set()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_ran.set()

    @program_a.task_factory
    async def factory_a():
        yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")

    @program_b.task_factory
    async def factory_b():
        if not done_b.is_set():
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            done_b.set()

    await kit.controller("ctrl1").enable()
    prog_a_client = kit.program("prog_a")
    await prog_a_client.enable("ctrl1")
    prog_b_client = kit.program("prog_b")
    await prog_b_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()

    # Start operating with program A without interrupt (the default).
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_a_client)

    async with asyncio.timeout(10.0):
        await task_a_started.wait()

    # Switching to program B without interrupt waits for A's task to finish.
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_b_client)

    # Allow time for the demand change to be processed while A's task remains running.
    await asyncio.sleep(0.2)
    assert not task_a_finished.is_set(), "task should still be running (not aborted)"

    # Allow the task to complete naturally.
    task_a_can_finish.set()

    async with asyncio.timeout(10.0):
        await task_a_finished.wait()
        await task_b_ran.wait()

    # Transition to SHUTDOWN for clean teardown.
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)

    async with asyncio.timeout(15.0):
        await shutdown_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.SHUTDOWN))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_quick_completion_with_long_offer_holds_tasking_loop(kit, until):
    """An empty task factory leaves its program active until demand changes.

    The lifecycle keeps retrying the demanded program's factory after its last task completes.
    Changing demand to another program starts that program's tasking.
    """
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program_a = await sc.register_program("prog_a")
    program_b = await sc.register_program("prog_b")

    task_a_ran = asyncio.Event()
    task_b_ran = asyncio.Event()
    done_a = asyncio.Event()
    done_b = asyncio.Event()
    shutdown_ran = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        pass

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        if not task_a_ran.is_set():
            task_a_ran.set()
        else:
            task_b_ran.set()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_ran.set()

    @program_a.task_factory
    async def factory_a():
        if not done_a.is_set():
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            done_a.set()
        # Subsequent factory calls yield no task.

    @program_b.task_factory
    async def factory_b():
        if not done_b.is_set():
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            done_b.set()

    await kit.controller("ctrl1").enable()
    prog_a_client = kit.program("prog_a")
    await prog_a_client.enable("ctrl1")
    prog_b_client = kit.program("prog_b")
    await prog_b_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()

    # Start operating with the high-priority program. Its single task completes immediately.
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_a_client)

    async with asyncio.timeout(10.0):
        await task_a_ran.wait()

    # The high-priority task has completed. Give the tasking loop time to cycle back
    # and discover there are no more tasks.
    await asyncio.sleep(1.0)

    # The lifecycle is still in the TASKING step -- the loop is retrying with NoTaskAvailable.
    assert lifecycle.step == LifecycleStep.TASKING
    assert not task_b_ran.is_set(), (
        "low-priority program should not run while lifecycle is occupied with high-priority"
    )

    # In the real system, the operator would re-evaluate the schedule and switch programs.
    # Simulate that by changing the demand to the low-priority program.
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_b_client)

    async with asyncio.timeout(15.0):
        await task_b_ran.wait()

    # Transition to SHUTDOWN for clean teardown.
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)

    async with asyncio.timeout(15.0):
        await shutdown_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.SHUTDOWN))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_shutdown_task_completes_despite_demand_change(kit, until):
    """Changing demand to OPERATE waits for the running shutdown task to finish."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program = await sc.register_program("prog1")

    init_ran = asyncio.Event()
    task_ran = asyncio.Event()
    shutdown_started = asyncio.Event()
    shutdown_can_finish = asyncio.Event()
    shutdown_finished = asyncio.Event()
    done = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        init_ran.set()

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        task_ran.set()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_started.set()
        await shutdown_can_finish.wait()
        shutdown_finished.set()

    @program.task_factory
    async def factory():
        if not done.is_set():
            yield CollectTask(task_id=uuid.uuid1(), controller_id="ctrl1")
            done.set()

    await kit.controller("ctrl1").enable()
    prog_client = kit.program("prog1")
    await prog_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_client)

    async with asyncio.timeout(10.0):
        await init_ran.wait()
        await task_ran.wait()

    # Demand SHUTDOWN and wait for its handler to block on the gate.
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)

    async with asyncio.timeout(10.0):
        await shutdown_started.wait()

    # While the shutdown task is still running, demand OPERATE again.
    lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_client)

    # Allow time for graceful cleanup to begin waiting for the shutdown task.
    await asyncio.sleep(0.3)

    # Graceful cleanup leaves the shutdown handler waiting on its gate.
    assert not shutdown_finished.is_set(), "shutdown should still be waiting (shielded from cancel)"

    # Release the shutdown handler.
    shutdown_can_finish.set()

    # The shutdown task should complete, and then the lifecycle should proceed to OPERATE.
    async with asyncio.timeout(10.0):
        await shutdown_finished.wait()

    # The pending OPERATE demand initializes the controller again.
    init_ran.clear()
    async with asyncio.timeout(15.0):
        await init_ran.wait()
        await until(lambda: reports(kit, InternalControllerState.OPERATE))

    await cleanup(lifecycle, sc)


@pytest.mark.asyncio
async def test_stop_aborts_running_task(kit):
    """Stopping the lifecycle aborts in-flight tasking before the service goes away."""
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    program = await sc.register_program("prog1")

    task_started = asyncio.Event()
    task_aborted = asyncio.Event()
    task_can_finish = asyncio.Event()

    @controller.task_handler(InitTask)
    async def handle_init(task):
        pass

    @controller.task_handler(CollectTask)
    async def handle_collect(task):
        task_started.set()

        try:
            await task_can_finish.wait()
        except asyncio.CancelledError:
            task_aborted.set()
            raise

    @program.task_factory
    async def factory():
        yield CollectTask()

    await kit.controller("ctrl1").enable()
    prog_client = kit.program("prog1")
    await prog_client.enable("ctrl1")

    lifecycle = ControllerLifecycle()

    try:
        async with asyncio.TaskGroup() as tg:
            lifecycle.start(kit.controller("ctrl1"), task_group=tg)
            lifecycle.enable()
            lifecycle.set_demand_state(InternalControllerState.OPERATE, program=prog_client)

            async with asyncio.timeout(10.0):
                await task_started.wait()

            async with asyncio.timeout(10.0):
                await lifecycle.stop()

        # The service is still up, so only the lifecycle's abort can reach the handler.
        async with asyncio.timeout(10.0):
            await task_aborted.wait()
    finally:
        task_can_finish.set()
        await sc.shutdown()


@pytest.mark.asyncio
async def test_cancelled_procedure_restarts_current_demand(kit):
    """Procedure cancellation leaves supervision running for the current demand."""
    starts = 0
    restarted = asyncio.Event()
    release = asyncio.Event()

    class CancellingStandby(StandbyProcedure):
        async def state_logic(self):
            nonlocal starts
            starts += 1

            if starts == 1:
                raise asyncio.CancelledError()

            restarted.set()
            await release.wait()

    class CancellingLifecycle(ControllerLifecycle):
        procedures = {
            **ControllerLifecycle.procedures,
            InternalControllerState.STANDBY: CancellingStandby,
        }

    sc = await kit.register_service("ctrl-svc", "0.1.0")
    lifecycle = CancellingLifecycle()

    try:
        await sc.register_controller("ctrl1")
        await kit.controller("ctrl1").enable()

        async with asyncio.TaskGroup() as tg:
            lifecycle.start(kit.controller("ctrl1"), task_group=tg)

            try:
                lifecycle.enable()
                lifecycle.set_demand_state(InternalControllerState.STANDBY)

                async with asyncio.timeout(5.0):
                    await restarted.wait()

                assert starts == 2
            finally:
                async with asyncio.timeout(5.0):
                    await lifecycle.stop()
    finally:
        await sc.shutdown()


@pytest.mark.asyncio
async def test_failed_shutdown_is_not_skipped_after_restart(backend):
    """A shutdown that failed before a restart runs again under SHUTDOWN demand."""
    kit = SensorKit(backend)
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")

    @controller.task_handler(ShutdownTask)
    async def fail_shutdown(task):
        raise RuntimeError("enclosure fault")

    try:
        await kit.controller("ctrl1").enable()

        with pytest.raises(CallError):
            await kit.controller("ctrl1").execute_task(ShutdownTask())
    finally:
        await sc.shutdown()

    # Restart on the same backend, so the controller recovers its persisted state.
    kit = SensorKit(backend)
    sc = await kit.register_service("ctrl-svc", "0.1.0")
    controller = await sc.register_controller("ctrl1")
    shutdown_ran = asyncio.Event()

    @controller.task_handler(ShutdownTask)
    async def handle_shutdown(task):
        shutdown_ran.set()

    lifecycle = ControllerLifecycle()
    lifecycle.start(kit.controller("ctrl1"))
    lifecycle.enable()
    lifecycle.set_demand_state(InternalControllerState.SHUTDOWN)

    try:
        async with asyncio.timeout(5.0):
            await shutdown_ran.wait()
    finally:
        await cleanup(lifecycle, sc)
