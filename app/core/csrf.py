import hmac

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Login/refresh establish the session, so they cannot present a CSRF token yet.
EXEMPT_PATHS = {"/admin/auth/login", "/admin/auth/refresh", "/admin/auth/logout"}


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        unsafe_method = request.method in {"POST", "PUT", "PATCH", "DELETE"}

        # The admin cookie authenticates every route, not just /admin/*, so CSRF
        # protection has to cover every route it can authenticate.
        if unsafe_method and request.url.path not in EXEMPT_PATHS:
            auth_header = request.headers.get("authorization", "")
            # Bearer-authenticated requests are not CSRF-able; cookies are.
            if not auth_header and request.cookies.get("admin_access_token"):
                csrf_cookie = request.cookies.get("admin_csrf_token")
                csrf_header = request.headers.get("x-csrf-token")
                if not csrf_cookie or not csrf_header or not _matches(csrf_cookie, csrf_header):
                    return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})
        return await call_next(request)


def _matches(cookie_value: str, header_value: str) -> bool:
    return hmac.compare_digest(cookie_value, header_value)
