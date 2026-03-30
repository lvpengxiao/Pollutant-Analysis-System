from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Callable


logger = logging.getLogger(__name__)


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[Callable[..., Any]]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        if callback not in self._subscribers[event_type]:
            self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Callable[..., Any]) -> None:
        callbacks = self._subscribers.get(event_type, [])
        if callback in callbacks:
            callbacks.remove(callback)

    def publish(self, event_type: str, *args: Any, **kwargs: Any) -> None:
        for callback in list(self._subscribers.get(event_type, [])):
            try:
                callback(*args, **kwargs)
            except Exception:
                logger.exception("Event handler failed for %s", event_type)
