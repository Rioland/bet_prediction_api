from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class CSRFMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        is_admin_path = request.url.path.startswith("/admin/")
        unsafe_method = request.method in {"POST", "PUT", "PATCH", "DELETE"}
        exempt_paths = {"/admin/auth/login", "/admin/auth/refresh"}

        if is_admin_path and unsafe_method and request.url.path not in exempt_paths:
            auth_header = request.headers.get("authorization", "")
            # CSRF protection is required for cookie-authenticated requests.
            if not auth_header and request.cookies.get("admin_access_token"):
                csrf_cookie = request.cookies.get("admin_csrf_token")
                csrf_header = request.headers.get("x-csrf-token")
                if not csrf_cookie or not csrf_header or csrf_cookie != csrf_header:
                    return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})
        return await call_next(request)
