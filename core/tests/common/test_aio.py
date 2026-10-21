# SPDX-License-Identifier: Apache-2.0
import asyncio
import gc

import pytest

from sensorkit.common.aio import (
    AsyncLoop,
    AsyncObserver,
    AsyncValueLatch,
    defer_cancellation,
    interruptible_wait,
)


async def released_operation(started: asyncio.Event, release: asyncio.Event, error=None):
    """Wait for release, then return a result or raise the given error."""
    started.set()
    await release.wait()

    if error is not None:
        raise error

    return "done"


@pytest.mark.asyncio
async def test_defer_cancellation_returns_result():
    async def compute(value: int):
        return value

    future = asyncio.get_running_loop().create_future()
    future.set_result(3)

    async with asyncio.timeout(1):
        assert await defer_cancellation(compute(1)) == 1
        assert await defer_cancellation(asyncio.create_task(compute(2))) == 2
        assert await defer_cancellation(future) == 3


@pytest.mark.asyncio
async def test_defer_cancellation_raises_operation_failure():
    async def fail():
        raise RuntimeError("operation failed")

    async with asyncio.timeout(1):
        with pytest.raises(RuntimeError, match="operation failed"):
            await defer_cancellation(fail())


@pytest.mark.asyncio
async def test_defer_cancellation_waits_through_repeated_cancellation():
    started = asyncio.Event()
    release = asyncio.Event()
    operation = asyncio.create_task(released_operation(started, release))
    caller = asyncio.create_task(defer_cancellation(operation))
    await started.wait()

    for _ in range(2):
        caller.cancel()
        await asyncio.sleep(0)

    # Neither the operation nor its caller ends before release.
    assert not operation.done()
    assert not caller.done()

    release.set()

    async with asyncio.timeout(1):
        with pytest.raises(asyncio.CancelledError):
            await caller

    assert operation.result() == "done"


@pytest.mark.asyncio
async def test_defer_cancellation_prefers_operation_failure():
    started = asyncio.Event()
    release = asyncio.Event()
    error = RuntimeError("operation failed")
    caller = asyncio.create_task(defer_cancellation(released_operation(started, release, error)))
    await started.wait()

    caller.cancel()
    await asyncio.sleep(0)
    release.set()

    async with asyncio.timeout(1):
        with pytest.raises(RuntimeError, match="operation failed"):
            await caller


@pytest.mark.asyncio
async def test_defer_cancellation_propagates_operation_cancellation():
    started = asyncio.Event()
    release = asyncio.Event()
    operation = asyncio.create_task(released_operation(started, release))
    caller = asyncio.create_task(defer_cancellation(operation))
    await started.wait()

    operation.cancel()

    async with asyncio.timeout(1):
        with pytest.raises(asyncio.CancelledError):
            await caller

    assert operation.cancelled()


@pytest.mark.asyncio
async def test_interruptible_wait_leaves_results_to_caller():
    loop = asyncio.get_running_loop()
    succeeded = loop.create_future()
    failed = loop.create_future()
    canceled = loop.create_future()
    loop.call_soon(succeeded.set_result, "done")
    loop.call_soon(failed.set_exception, ValueError("operation failed"))
    loop.call_soon(canceled.cancel)

    async with asyncio.timeout(1):
        assert await interruptible_wait((succeeded, failed, canceled)) == set()

    assert succeeded.result() == "done"

    with pytest.raises(ValueError, match="operation failed"):
        failed.result()

    assert canceled.cancelled()
    assert await interruptible_wait(()) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("already_set", [False, True])
async def test_interruptible_wait_preserves_work_on_interrupt(already_set):
    started = asyncio.Event()
    release = asyncio.Event()
    interrupt = asyncio.Event()
    operation = asyncio.create_task(released_operation(started, release))

    if already_set:
        interrupt.set()

    caller = asyncio.create_task(interruptible_wait((operation,), interrupt=interrupt))
    await started.wait()
    await asyncio.sleep(0)
    interrupt.set()

    async with asyncio.timeout(1):
        assert await caller == {operation}

    assert not operation.done()
    release.set()
    assert await operation == "done"


@pytest.mark.asyncio
async def test_interruptible_wait_preserves_work_on_timeout():
    started = asyncio.Event()
    release = asyncio.Event()
    operation = asyncio.create_task(released_operation(started, release))

    async with asyncio.timeout(1):
        assert await interruptible_wait((operation,), timeout=0.01) == {operation}

    assert not operation.done()
    release.set()
    assert await operation == "done"


@pytest.mark.asyncio
@pytest.mark.parametrize("with_interrupt", [False, True])
async def test_interruptible_wait_preserves_work_on_caller_cancellation(with_interrupt):
    started = asyncio.Event()
    release = asyncio.Event()
    interrupt = asyncio.Event() if with_interrupt else None
    operation = asyncio.create_task(released_operation(started, release))
    caller = asyncio.create_task(interruptible_wait((operation,), interrupt=interrupt))
    await started.wait()
    await asyncio.sleep(0)
    caller.cancel()

    async with asyncio.timeout(1):
        with pytest.raises(asyncio.CancelledError):
            await caller

    assert not operation.done()
    release.set()
    assert await operation == "done"


@pytest.mark.asyncio
async def test_async_value_latch():
    latch = AsyncValueLatch(0)

    assert latch.value == 0
    assert latch.pending_value == 0
    assert not latch.pending_change()

    # Stage a change
    changed = latch.update(1)
    assert changed is True
    assert latch.pending_change()
    assert latch.pending_value == 1

    # Apply the change
    applied = latch.apply()
    assert applied == 1
    assert latch.value == 1
    assert not latch.pending_change()


@pytest.mark.asyncio
async def test_async_value_latch_pending_wait():
    latch = AsyncValueLatch("a")

    # No-op update when same value and only_if_different=True (default)
    assert not latch.update("a")
    assert not latch.pending_change()

    # Waiter should block until an update occurs
    started = asyncio.Event()
    resumed = asyncio.Event()

    async def waiter():
        started.set()
        await latch.wait_until_pending()
        resumed.set()

    task = asyncio.create_task(waiter())
    await started.wait()

    # Still waiting
    await asyncio.sleep(0)
    assert not resumed.is_set()

    # Trigger a pending change with a new value
    assert latch.update("b")
    await asyncio.wait_for(resumed.wait(), timeout=1)

    # Apply the change
    assert latch.apply() == "b"
    assert latch.value == "b"

    # Updating with same value does nothing when only_if_different=True
    assert latch.update("b") is False
    assert not latch.pending_change()

    # Forcing update with only_if_different=False should set pending
    assert latch.update("b", only_if_different=False) is True
    assert latch.pending_change()

    # Cleanup
    if not task.done():
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        assert task.done()


@pytest.mark.asyncio
async def test_async_observer():
    obs = AsyncObserver(0)
    q1 = obs.subscribe(initial_value=True)
    q2 = obs.subscribe(initial_value=False)

    async with asyncio.timeout(1):
        assert obs.value == 0
        assert await q1.get() == 0
        q1.task_done()
        assert q2.empty()

        obs.notify(1)
        assert obs.value == 1
        assert await q1.get() == 1
        q1.task_done()
        assert await q2.get() == 1
        q2.task_done()

        # Unsubscribe one queue; it should not receive further notifications.
        obs.unsubscribe(q2)
        assert obs.subscriber_count == 1

        obs.notify(2)
        assert await q1.get() == 2
        q1.task_done()

        with pytest.raises(asyncio.QueueShutDown):
            await q2.get()

        # Test unset initial value.
        obs = AsyncObserver()

        with pytest.raises(RuntimeError):
            _ = obs.value

        assert obs.subscribe(initial_value=True).empty()

        obs.notify(42)
        assert obs.value == 42


@pytest.mark.asyncio
async def test_async_observer_generator():
    obs = AsyncObserver("x")
    queue = asyncio.Queue()
    ready = asyncio.Event()

    async def reader(n: int):
        async for v in obs.consume(initial_value=True):
            ready.set()
            queue.put_nowait(v)
            n -= 1

            if n <= 0:
                break

    # Create a task to read two notifications.
    task = asyncio.create_task(reader(n=2))
    await ready.wait()

    # Reach in and get a reference to our generator's underlying queue.
    underlying_queue = next(iter(obs._observers))

    async with asyncio.timeout(1):
        # The initial value should already be there.
        assert await queue.get() == "x"

        # Add a second value, which will cause the reader to exit its generator.
        obs.notify("y")
        assert await queue.get() == "y"

        # Wait for the reader task to end. Cleanup runs when the event loop finalizes the
        # abandoned generator, which takes an iteration or two.
        await task

        while obs.subscriber_count:
            await asyncio.sleep(0)

        # Make sure everything cleaned up.
        assert obs.subscriber_count == 0
        await underlying_queue.join()

        with pytest.raises(asyncio.QueueShutDown):
            await underlying_queue.get()


@pytest.mark.asyncio
async def test_async_observer_retains_unreferenced_subscriber():
    obs = AsyncObserver(0)

    # Nothing outside the observer references this queue. It must stay subscribed regardless,
    # so that a consumer which goes away without unsubscribing leaks visibly instead of
    # silently ceasing to receive values.
    obs.subscribe()
    gc.collect()

    assert obs.subscriber_count == 1


@pytest.mark.asyncio
async def test_async_observer_subscription_scope():
    obs = AsyncObserver(0)

    with obs.subscription() as queue:
        assert obs.subscriber_count == 1

        obs.notify(1)
        assert await queue.get() == 1

    assert obs.subscriber_count == 0

    with pytest.raises(asyncio.QueueShutDown):
        await queue.get()

    # An exception leaving the block must still unsubscribe.
    with pytest.raises(RuntimeError), obs.subscription():
        raise RuntimeError("boom")

    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_rejects_unbounded_subscriber():
    obs = AsyncObserver(0)

    # asyncio.Queue treats a non-positive maxsize as unbounded.
    for maxsize in (0, -1):
        with pytest.raises(ValueError):
            obs.subscribe(maxsize=maxsize)


@pytest.mark.asyncio
async def test_async_observer_drops_oldest_when_full():
    obs = AsyncObserver(0)
    queue = obs.subscribe(maxsize=2)

    # The third value overflows the queue.
    for value in (1, 2, 3):
        obs.notify(value)

    assert obs.dropped == 1

    # The oldest value gave way, leaving the subscriber converged on the latest.
    assert await queue.get() == 2
    queue.task_done()
    assert await queue.get() == 3
    queue.task_done()
    assert queue.empty()

    # A dropped value must not be left outstanding, which would strand join() forever.
    async with asyncio.timeout(1):
        await queue.join()


@pytest.mark.asyncio
async def test_async_observer_single_value_queue_keeps_latest():
    obs = AsyncObserver(0)
    queue = obs.subscribe(initial_value=True, maxsize=1)

    for value in (1, 2, 3):
        obs.notify(value)

    # Replacement is intended, but still counted.
    assert obs.dropped == 3
    assert queue.get_nowait() == 3
    assert queue.empty()


@pytest.mark.asyncio
async def test_async_observer_rejects_unknown_overflow_policy():
    obs = AsyncObserver(0)

    with pytest.raises(ValueError):
        obs.subscribe(overflow="newest")

    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_overflow_closes_only_that_subscriber():
    obs = AsyncObserver(0)
    closing = obs.subscribe(maxsize=2, overflow="close")
    dropping = obs.subscribe(maxsize=2)
    roomy = obs.subscribe()

    for value in (1, 2, 3):
        obs.notify(value)

    # Both queued values and the incoming one are lost to the closed subscriber.
    assert obs.dropped == 3 + 1
    assert obs.subscriber_count == 2
    assert not obs.closed
    assert obs.value == 3

    # Pending values are discarded rather than left as incomplete history.
    with pytest.raises(asyncio.QueueShutDown):
        closing.get_nowait()

    assert [dropping.get_nowait() for _ in range(2)] == [2, 3]
    assert [roomy.get_nowait() for _ in range(3)] == [1, 2, 3]

    # The observer stays open for the remaining subscribers.
    obs.notify(4)
    assert dropping.get_nowait() == 4
    assert roomy.get_nowait() == 4


@pytest.mark.asyncio
async def test_async_observer_overflow_counts_initial_value():
    obs = AsyncObserver(0)
    queue = obs.subscribe(initial_value=True, maxsize=1, overflow="close")

    obs.notify(1)

    assert obs.dropped == 2
    assert obs.subscriber_count == 0

    with pytest.raises(asyncio.QueueShutDown):
        queue.get_nowait()


@pytest.mark.asyncio
async def test_async_observer_overflow_wakes_blocked_reader():
    obs = AsyncObserver(0)
    queue = obs.subscribe(maxsize=1, overflow="close")
    reader = asyncio.create_task(queue.get())

    # Let the reader block, then wake it with a value and overflow before it resumes.
    await asyncio.sleep(0)
    obs.notify(1)
    obs.notify(2)

    async with asyncio.timeout(1):
        with pytest.raises(asyncio.QueueShutDown):
            await reader

    assert obs.dropped == 2


@pytest.mark.asyncio
async def test_async_observer_overflow_join_waits_for_delivered_values():
    obs = AsyncObserver(0)
    queue = obs.subscribe(maxsize=2, overflow="close")

    obs.notify(1)
    assert queue.get_nowait() == 1

    for value in (2, 3, 4):
        obs.notify(value)

    assert obs.subscriber_count == 0
    joining = asyncio.create_task(queue.join())

    # The delivered value is still outstanding, while discarded ones are not.
    await asyncio.sleep(0)
    assert not joining.done()

    queue.task_done()

    async with asyncio.timeout(1):
        await joining


@pytest.mark.asyncio
async def test_async_observer_canceled_read_keeps_subscription():
    obs = AsyncObserver(0)

    with obs.subscription(maxsize=1, overflow="close") as queue:
        reader = asyncio.create_task(queue.get())
        await asyncio.sleep(0)
        reader.cancel()

        with pytest.raises(asyncio.CancelledError):
            await reader

        assert obs.subscriber_count == 1

        obs.notify(1)
        assert queue.get_nowait() == 1


@pytest.mark.asyncio
async def test_async_observer_unsubscribe_after_overflow():
    obs = AsyncObserver(0)

    with obs.subscription(maxsize=1, overflow="close") as queue:
        obs.notify(1)
        obs.notify(2)

    assert obs.subscriber_count == 0

    with pytest.raises(asyncio.QueueShutDown):
        queue.get_nowait()


@pytest.mark.asyncio
async def test_async_observer_close_wakes_blocked_readers():
    obs = AsyncObserver(0)
    queues = [obs.subscribe(), obs.subscribe(overflow="close")]
    readers = [asyncio.create_task(queue.get()) for queue in queues]

    await asyncio.sleep(0)
    obs.close()

    async with asyncio.timeout(1):
        for reader in readers:
            with pytest.raises(asyncio.QueueShutDown):
                await reader


@pytest.mark.asyncio
async def test_async_observer_close_discards_pending_values():
    obs = AsyncObserver(0)

    with obs.subscription() as queue:
        obs.notify(1)
        obs.notify(2)
        assert queue.get_nowait() == 1

        obs.close()

        assert obs.subscriber_count == 0
        assert obs.dropped == 0

        with pytest.raises(asyncio.QueueShutDown):
            queue.get_nowait()

        # Only the delivered value remains outstanding.
        joining = asyncio.create_task(queue.join())
        await asyncio.sleep(0)
        assert not joining.done()

        queue.task_done()

        async with asyncio.timeout(1):
            await joining

    # Leaving the subscription after close is safe.
    obs.unsubscribe(queue)
    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_closed_lifetime():
    obs = AsyncObserver(0)
    assert not obs.closed

    obs.subscribe(maxsize=1, overflow="close")
    obs.notify(1)
    obs.notify(2)

    # An isolated overflow leaves the observer open.
    assert obs.subscriber_count == 0
    assert not obs.closed

    obs.close()
    assert obs.closed

    obs.close()
    assert obs.closed

    # The last value stays readable, but the observer cannot be reused.
    assert obs.value == 2

    with pytest.raises(RuntimeError):
        obs.subscribe()

    with pytest.raises(RuntimeError):
        obs.notify(3)

    assert obs.value == 2


@pytest.mark.asyncio
async def test_async_observer_consume_ends_on_close():
    obs = AsyncObserver(0)
    seen = []

    async def reader():
        async for value in obs.consume(initial_value=True):
            seen.append(value)

            if value == 1:
                obs.close()

    task = asyncio.create_task(reader())
    await asyncio.sleep(0)
    obs.notify(1)

    async with asyncio.timeout(1):
        await task

    assert seen == [0, 1]
    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_consume_raises_on_overflow():
    obs = AsyncObserver(0)
    values = obs.consume(maxsize=1, overflow="close")
    reading = asyncio.ensure_future(anext(values))

    await asyncio.sleep(0)
    obs.notify(1)
    obs.notify(2)

    async with asyncio.timeout(1):
        with pytest.raises(asyncio.QueueShutDown):
            await reading

    assert not obs.closed
    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_consume_raises_on_close():
    obs = AsyncObserver(0)
    values = obs.consume(initial_value=True, overflow="close")

    async with asyncio.timeout(1):
        assert await anext(values) == 0

        obs.close()

        with pytest.raises(asyncio.QueueShutDown):
            await anext(values)

    assert obs.closed
    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_consume_unsubscribes_on_cancellation():
    obs = AsyncObserver(0)
    values = obs.consume(initial_value=True)

    async with asyncio.timeout(1):
        assert await anext(values) == 0

        # Cancellation releases the subscription.
        with pytest.raises(asyncio.CancelledError):
            await values.athrow(asyncio.CancelledError())

    assert obs.subscriber_count == 0


@pytest.mark.asyncio
async def test_async_observer_notify_does_not_suspend():
    obs = AsyncObserver(0)
    queue = obs.subscribe()
    seen = []

    async def reader():
        while True:
            seen.append(await queue.get())

    task = asyncio.create_task(reader())

    # Let the reader reach its first get() and block there.
    await asyncio.sleep(0)

    for value in (1, 2, 3):
        obs.notify(value)

    # No subscriber could have run between the notifications.
    assert not seen
    assert queue.qsize() == 3

    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_async_loop_start_stop():
    ticks = 0

    async def body():
        nonlocal ticks
        ticks += 1

    loop = AsyncLoop(body, interval=0.01)
    assert not loop.active

    assert loop.start() is loop
    assert loop.active

    async with asyncio.timeout(1):
        while ticks < 3:
            await asyncio.sleep(0.01)

    await loop.stop()
    assert not loop.active

    # No further iterations once stopped.
    settled = ticks
    await asyncio.sleep(0.05)
    assert ticks == settled


@pytest.mark.asyncio
async def test_async_loop_start_is_idempotent():
    ticks = 0

    async def body():
        nonlocal ticks
        ticks += 1

    loop = AsyncLoop(body, interval=0.01)

    # Starting repeatedly must not leave orphaned tasks behind, which a single stop
    # would fail to cancel.
    loop.start()
    loop.start()
    loop.start()

    async with asyncio.timeout(1):
        while ticks < 2:
            await asyncio.sleep(0.01)

    await loop.stop()

    settled = ticks
    await asyncio.sleep(0.05)
    assert ticks == settled


@pytest.mark.asyncio
async def test_async_loop_restart_after_stop():
    ticks = 0

    async def body():
        nonlocal ticks
        ticks += 1

    loop = AsyncLoop(body, interval=0.01)
    loop.start()

    async with asyncio.timeout(1):
        while ticks < 1:
            await asyncio.sleep(0.01)

    await loop.stop()
    stopped_at = ticks

    loop.start()
    assert loop.active

    async with asyncio.timeout(1):
        while ticks == stopped_at:
            await asyncio.sleep(0.01)

    await loop.stop()


@pytest.mark.asyncio
async def test_async_loop_continues_after_error():
    attempts = 0

    async def body():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("boom")

    loop = AsyncLoop(body, interval=0.01)
    loop.start()

    async with asyncio.timeout(1):
        while attempts < 3:
            await asyncio.sleep(0.01)

    assert loop.active
    await loop.stop()


@pytest.mark.asyncio
async def test_async_loop_stop_without_start():
    async def body():
        await asyncio.sleep(0)

    loop = AsyncLoop(body, interval=0.01)

    # Stopping is safe before a first start, and again after a stop.
    await loop.stop()
    loop.start()
    await loop.stop()
    await loop.stop()

    assert not loop.active
