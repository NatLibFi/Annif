"""ASGI middleware for enforcing maximum request body size.

This prevents OOM crashes by checking the Content-Length header at the ASGI
level, before Starlette's Request.body() method buffers the entire payload
into memory.
"""

from __future__ import annotations

import logging
from typing import Callable

from starlette.types import Receive, Scope, Send

logger = logging.getLogger("annif")


class BodyLimitMiddleware:
    """ASGI middleware that rejects requests with oversized bodies BEFORE they are
    parsed into memory.

    Checks Content-Length at the ASGI level and sends 413 immediately if the
    limit is exceeded, preventing Starlette from buffering the full body.
    """

    def __init__(self, app: Callable, max_content_length: int) -> None:
        self.app = app
        self.max_content_length = max_content_length

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_length_header = headers.get(b"content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
            except ValueError:
                content_length = 0

            if content_length > self.max_content_length:
                path = scope.get("path", "")
                logger.warning(
                    "[BODY-LIMIT] Rejecting request: path=%s "
                    "content_length=%s bytes "
                    "(exceeds limit=%s)",
                    path,
                    content_length,
                    self.max_content_length,
                )
                # Send 413 error response IMMEDIATELY without consuming the body.
                # The server can send a response before the full body arrives.
                # The client will receive the 413 and stop sending (or the
                # connection will be closed).
                limit_str = str(self.max_content_length).encode()
                error_body = (
                    b'{"detail":"Request entity too large. '
                    b'Maximum allowed size is '
                    + limit_str
                    + b' bytes."}'
                )
                error_headers = [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(error_body)).encode()),
                ]
                await send(
                    {
                        "type": "http.response.start",
                        "status": 413,
                        "headers": error_headers,
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": error_body,
                    }
                )
                return

        await self.app(scope, receive, send)
