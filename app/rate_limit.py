"""Shared rate limiter for credential endpoints.

In-memory, so limits are per process. That is enough to stop a single client
hammering login from one worker; a multi-worker deployment wanting a global
limit should point slowapi at Redis.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
