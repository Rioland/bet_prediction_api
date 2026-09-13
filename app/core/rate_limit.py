"""Shared rate limiter.

Lives outside ``app.main`` so routers can apply limits without a circular import.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
