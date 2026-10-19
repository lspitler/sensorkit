# SPDX-License-Identifier: Apache-2.0
"""Async utilities: scoped futures, periodic loops, value latches, and observer queues."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import AsyncContextManager, ClassVar, Literal, Self, get_args, overload

from sensorkit.common.logging import limited_logger

type OverflowPolicy = Literal["drop_oldest", "close"]
"""Overflow handling for observer queues."""


def scoped_waiter[T](aw: Awaitable[T]) -> AsyncContextManager[asyncio.Future[T]]:
    """Create an async context manager wrapping an awaitable in a scoped `asyncio.Future`.

    The future is cancelled if it has not completed before the `async with` block
    exits or if an exception propagates out of the block.

    Important:
        Any exceptions raised by the awaitable are silently discarded. This makes
        `scoped_waiter` suitable only for "waiter" type tasks where exceptions either
        won't occur or are not meaningful to handle (e.g. background monitoring tasks,
        optional notifications, or advisory operations).

    Args:
        aw: The awaitable object to be wrapped in an `asyncio.Future`.

    Returns:
        An async context manager that yields the `asyncio.Future` for the awaitable.
    """
    return _scoped_waiter(aw)


@overload
def cleanup_future(fut: asyncio.Task): ...

@overload
def cleanup_future(fut: asyncio.Future): ...

def cleanup_future(fut: asyncio.Future | asyncio.Task):
    """Safely clean up a Future or an already-completed Task.

    Args:
        fut: The asyncio.Future to clean up.
    """
    if not fut.done():
        fut.cancel()
    elif not fut.cancelled():
        fut.exception()


async def defer_cancellation[T](aw: Awaitable[T]) -> T:
    """Finish an awaitable before propagating caller cancellation.

    Repeated caller cancellation leaves the operation running and waits for it to finish.
    Operation failures, including cancellation of the operation itself, take precedence
    over caller cancellation.

    Args:
        aw: The operation, as a coroutine, task, or future.

    Returns:
        The operation's result if the caller was not canceled while waiting.

    Raises:
        asyncio.CancelledError: If the caller or the operation was canceled.
    """
    fut = asyncio.ensure_future(aw)
    cancelled = False

    while not fut.done():
        try:
            await asyncio.shield(fut)
        except asyncio.CancelledError:
            cancelled = True

    result = fut.result()

    if cancelled:
        raise asyncio.CancelledError()

    return result


@contextlib.asynccontextmanager
async def _scoped_waiter(aw: Awaitable) -> AsyncGenerator[asyncio.Task]:
    fut = asyncio.ensure_future(aw)

    try:
        yield fut
    finally:
        if not fut.done():
            fut.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await fut
        else:
            with contextlib.suppress(asyncio.CancelledError):
                fut.exception()


class AsyncValueLatch[T]:
    """Stores a value and stages pending changes."""

    def __init__(self, initial_value: T):
        self.value: T = initial_value
        self.pending_value: T = initial_value
        self._pending = asyncio.Event()

    def update(self, value: T, only_if_different=True):
        """Flag a pending value change and set that pending value."""
        if only_if_different and value == self.pending_value:
            return False

        self.pending_value = value
        self._pending.set()
        return True

    def pending_change(self):
        """Return True if there is a pending value change."""
        return self._pending.is_set()

    async def wait_until_pending(self):
        """Wait until there is a pending value change."""
        await self._pending.wait()

    def apply(self):
        """Apply the pending value if there is one and return the current value."""
        if self._pending.is_set():
            self._pending.clear()
            self.value = self.pending_value

        return self.value


class AsyncObserver[T]:
    """Fan out notified values to independent, bounded subscriber queues.

    Each subscriber receives values notified after it subscribes, in order. With
    `initial_value=True`, it first receives the current value, if one has been set.
    Each subscriber chooses what happens when its queue is full. By default its oldest
    pending value is discarded.

    Closing the observer shuts down every subscriber queue and ends notification.
    """

    NOT_SET: ClassVar[object] = object()
    DEFAULT_MAXSIZE: ClassVar[int] = 1024

    def __init__(self, initial_value: T = NOT_SET):
        self._observers: dict[asyncio.Queue[T], OverflowPolicy] = {}
        self._value = initial_value
        self._dropped = 0
        self._closed = False

    def subscribe(
        self,
        *,
        initial_value: bool = False,
        maxsize: int = DEFAULT_MAXSIZE,
        overflow: OverflowPolicy = "drop_oldest",
    ):
        """Create and return a new observer queue.

        Args:
            initial_value: Whether to seed the queue with the current value, if one is set.
            maxsize: Bound on values queued for this subscriber.
            overflow: What to do when the queue is full. `drop_oldest` discards the oldest
                pending value. `close` unsubscribes the queue and shuts it down immediately,
                discarding its pending values.

        Raises:
            RuntimeError: If the observer is closed.
            ValueError: If `maxsize` is not positive, which would leave the queue unbounded,
                or if `overflow` is not a known policy.
        """
        if self._closed:
            raise RuntimeError("Cannot subscribe to a closed observer")

        if maxsize < 1:
            raise ValueError(f"maxsize must be positive, got {maxsize}")

        if overflow not in get_args(OverflowPolicy.__value__):
            raise ValueError(f"unknown overflow policy {overflow!r}")

        queue: asyncio.Queue[T] = asyncio.Queue(maxsize)
        self._observers[queue] = overflow

        if initial_value and self._value is not self.NOT_SET:
            queue.put_nowait(self._value)

        return queue

    def unsubscribe(self, queue: asyncio.Queue[T]):
        """Remove an observer queue, leaving its pending values readable."""
        self._observers.pop(queue, None)
        queue.shutdown()

    @contextlib.contextmanager
    def subscription(
        self,
        *,
        initial_value: bool = False,
        maxsize: int = DEFAULT_MAXSIZE,
        overflow: OverflowPolicy = "drop_oldest",
    ):
        """Context manager that provides a subscriber queue and unsubscribes it on exit."""
        queue = self.subscribe(initial_value=initial_value, maxsize=maxsize, overflow=overflow)

        try:
            yield queue
        finally:
            self.unsubscribe(queue)

    def notify(self, value: T):
        """Update the current value and notify all observers.

        Delivery never blocks on a subscriber. A subscriber whose queue is full is handled
        by its overflow policy.

        Raises:
            RuntimeError: If the observer is closed.
        """
        if self._closed:
            raise RuntimeError("Cannot notify a closed observer")

        self._value = value

        # Closing an overflowing queue removes its subscription.
        for queue, overflow in list(self._observers.items()):
            try:
                queue.put_nowait(value)
            except asyncio.QueueFull:
                match overflow:
                    case "drop_oldest":
                        queue.get_nowait()
                        queue.task_done()
                        queue.put_nowait(value)
                        self._dropped += 1

                        limited_logger().warning(
                            f"observer queue full at {queue.maxsize}; dropping the oldest value. "
                            "A subscriber is not keeping up, or leaked without unsubscribing."
                        )
                    case "close":
                        del self._observers[queue]
                        self._dropped += queue.qsize() + 1
                        queue.shutdown(immediate=True)

                        limited_logger().warning(
                            f"observer queue full at {queue.maxsize}; closing the subscriber. "
                            "It is not keeping up, or leaked without unsubscribing."
                        )

    def close(self):
        """Permanently close the observer.

        Discard pending values and wake subscribers with `asyncio.QueueShutDown`.
        The current value stays readable. Closing again has no effect.
        """
        self._closed = True
        queues = list(self._observers)
        self._observers.clear()

        for queue in queues:
            queue.shutdown(immediate=True)

    @property
    def value(self) -> T:
        """Get the current value."""
        if self._value is self.NOT_SET:
            raise RuntimeError("Observed value must be set prior to access")

        return self._value

    @property
    def subscriber_count(self) -> int:
        """The number of queues currently subscribed."""
        return len(self._observers)

    @property
    def dropped(self) -> int:
        """Values lost to overflow, including pending values discarded on overflow closure."""
        return self._dropped

    @property
    def closed(self) -> bool:
        """Whether the observer has been closed."""
        return self._closed

    async def consume(
        self,
        *,
        initial_value: bool = False,
        maxsize: int = DEFAULT_MAXSIZE,
        overflow: OverflowPolicy = "drop_oldest",
    ):
        """Yield successive values as an async generator, unsubscribing automatically on exit.

        By default, iteration ends when the observer closes. With `overflow="close"`,
        queue shutdown raises `asyncio.QueueShutDown`. The `closed` property reports
        whether the observer itself has closed.
        """
        queue = self.subscribe(initial_value=initial_value, maxsize=maxsize, overflow=overflow)

        try:
            while True:
                item = await queue.get()

                try:
                    yield item
                finally:
                    queue.task_done()
        except asyncio.QueueShutDown:
            if overflow == "close":
                raise
        finally:
            self.unsubscribe(queue)


class AsyncLoop:
    """Runs a function repeatedly on a fixed interval.

    The interval is a plain attribute, so assigning to it retunes a running loop from
    its next sleep onward.
    """

    def __init__(
        self,
        func: Callable[[], Awaitable[None] | None],
        *,
        interval: float,
        log: bool = False,
        label: str | None = None,
    ):
        """Initialize the loop without starting it.

        Args:
            func: Called once per iteration. Takes no arguments.
            interval: Seconds to sleep between iterations.
            log: Whether to log exceptions raised by `func`.
            label: Optional label for logging.
        """
        self.func = func
        self.interval = interval
        self._log = log
        self._label = label or getattr(func, "__qualname__", str(func))
        self._task: asyncio.Task | None = None

    @property
    def active(self) -> bool:
        """Report whether the loop is currently running."""
        return self._task is not None and not self._task.done()

    def start(self) -> Self:
        """Start the loop, returning self.

        Does nothing if the loop is already running, so callers that cannot easily
        tell may start it unconditionally.
        """
        if not self.active:
            self._task = asyncio.create_task(self._run())

        return self

    async def stop(self):
        """Cancel the loop and wait for it to unwind.

        Safe to call on a loop that was never started, and on one already stopped.
        """
        if self._task is None:
            return

        task = self._task
        task.cancel()
        self._task = None

        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def _run(self):
        while True:
            try:
                if coro := self.func():
                    await coro
            except Exception as e:
                if self._log:
                    from loguru import logger

                    msg = f": {e}" if str(e) else ""
                    logger.exception(f"{type(e).__name__} in {self._label} loop{msg}")

            await asyncio.sleep(self.interval)
