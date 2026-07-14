"""Small production browser adapter for the PPC-2 request boundary."""
from __future__ import annotations

from pathlib import Path
from types import ModuleType

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

SCRIPT_PATH = "/platform-request.js"
SCRIPT = r'''(() => {
  const rawFetch = window.fetch.bind(window);
  const cookieValue = (name) => {
    const prefix = name + "=";
    for (const part of document.cookie.split(";")) {
      const item = part.trim();
      if (item.startsWith(prefix)) return decodeURIComponent(item.slice(prefix.length));
    }
    return "";
  };
  window.fetch = (input, init = {}) => {
    const method = String(init.method || "GET").toUpperCase();
    if (!["GET", "HEAD", "OPTIONS"].includes(method)) {
      const token = cookieValue("vektoryum_request");
      if (token) {
        const headers = new Headers(init.headers || {});
        headers.set("X-Vektoryum-Request", token);
        init = { ...init, headers };
      }
    }
    return rawFetch(input, init);
  };
})();
'''


def _remove_root_route(app: FastAPI) -> None:
    kept = []
    for route in app.router.routes:
        path = getattr(route, "path", None)
        methods = set(getattr(route, "methods", set()) or set())
        if path == "/" and "GET" in methods:
            continue
        kept.append(route)
    app.router.routes[:] = kept


def install_platform_frontend(main_module: ModuleType, app: FastAPI | None = None) -> None:
    target = app or main_module.app
    if getattr(target.state, "platform_frontend_installed", False):
        return
    _remove_root_route(target)

    async def request_script() -> Response:
        return Response(
            SCRIPT,
            media_type="application/javascript",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    async def index():
        static_dir = Path(main_module.__file__).parent / "static"
        index_path = static_dir / "index.html"
        if not index_path.exists():
            return JSONResponse(
                {"status": "ok", "service": "vektoryum-api", "modes": main_module.ALLOWED_MODES}
            )
        text = index_path.read_text(encoding="utf-8")
        marker = "</head>"
        tag = f'<script src="{SCRIPT_PATH}"></script>'
        if tag not in text:
            text = text.replace(marker, tag + marker, 1)
        return HTMLResponse(text)

    target.add_api_route(SCRIPT_PATH, request_script, methods=["GET"], include_in_schema=False)
    target.add_api_route("/", index, methods=["GET"], summary="Web arayüzü", include_in_schema=False)
    target.state.platform_frontend_installed = True


__all__ = ["SCRIPT", "SCRIPT_PATH", "install_platform_frontend"]
