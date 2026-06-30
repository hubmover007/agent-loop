"""Circuit Breaker — per-provider failure tracking.

States:
  CLOSED    → normal operation
  OPEN      → provider blocked after N consecutive failures
  HALF_OPEN → probing: allow limited requests, if succeed → CLOSED, if fail → OPEN
"""

from __future__ import annotations

import asyncio
import logging
import time
from enum import Enum

logger = logging.getLogger(__name__)


class CBState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Per-provider circuit breaker."""

    def __init__(self, provider_id: str, failure_threshold: int = 5,
                 recovery_timeout_s: float = 60.0, half_open_requests: int = 2):
        self.provider_id = provider_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self.half_open_requests = half_open_requests

        self._state = CBState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CBState:
        return self._state

    def is_available(self) -> bool:
        """Check if provider can accept requests right now."""
        if self._state == CBState.CLOSED:
            return True
        if self._state == CBState.OPEN:
            # Check recovery timeout
            if self._opened_at and (time.monotonic() - self._opened_at) >= self.recovery_timeout_s:
                return True  # will transition to HALF_OPEN on next acquire
            return False
        if self._state == CBState.HALF_OPEN:
            return self._half_open_in_flight < self.half_open_requests
        return False

    async def acquire(self) -> bool:
        """Try to acquire a request slot. Returns False if breaker is open."""
        async with self._lock:
            if self._state == CBState.CLOSED:
                return True

            if self._state == CBState.OPEN:
                elapsed = time.monotonic() - (self._opened_at or 0)
                if elapsed >= self.recovery_timeout_s:
                    self._state = CBState.HALF_OPEN
                    self._half_open_in_flight = 0
                    logger.info("CircuitBreaker[%s]: OPEN → HALF_OPEN", self.provider_id)
                    self._half_open_in_flight += 1
                    return True
                return False

            if self._state == CBState.HALF_OPEN:
                if self._half_open_in_flight < self.half_open_requests:
                    self._half_open_in_flight += 1
                    return True
                return False

        return False

    async def record_success(self) -> None:
        """Record successful call — resets failure count, closes breaker."""
        async with self._lock:
            self._consecutive_failures = 0
            if self._state in (CBState.HALF_OPEN, CBState.OPEN):
                logger.info("CircuitBreaker[%s]: → CLOSED (recovered)", self.provider_id)
            self._state = CBState.CLOSED
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    async def record_failure(self) -> None:
        """Record failed call — may open breaker."""
        async with self._lock:
            self._consecutive_failures += 1
            if self._state == CBState.HALF_OPEN:
                # Probe failed — back to OPEN
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
                logger.warning("CircuitBreaker[%s]: HALF_OPEN → OPEN (probe failed)", self.provider_id)
            elif self._consecutive_failures >= self.failure_threshold:
                self._state = CBState.OPEN
                self._opened_at = time.monotonic()
                logger.warning(
                    "CircuitBreaker[%s]: CLOSED → OPEN (%d consecutive failures)",
                    self.provider_id, self._consecutive_failures
                )

    def status(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "opened_at": self._opened_at,
        }
