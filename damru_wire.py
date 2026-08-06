"""
damru_wire.py
=============
Step 5 (integration) -- ONE-CALL wiring of the Damru Beast modules into app.py.

In app.py, just before the Gradio mount, add:

    try:
        from damru_wire import wire_damru
        BELT = wire_damru(api, globals())
    except Exception as e:
        BELT = None
        print("wiring skipped:", e)

wire_damru() (all additive, all guarded):
  1. Builds the Step-1 Tool Belt using providers taken from app.py's OWN
     functions (_llm_complete, prayas_retrieve, web_search, image_router,
     cortex_answer, trellis image_to_glb, DTwin) + Step-2 Live Internet.
  2. Mounts the Step-2 /internet router and the Step-3 /live router.
  3. Adds /tools (list), /tools/run (route+gather), /chat_smart (SSE 5-stage stream).
  4. Returns the belt. Any missing piece simply disables that one capability.

Built by Shiva AI for Damru.
"""

from __future__ import annotations

import io
import re
import json
import builtins
import contextlib
import logging
from typing import Any, Callable, Dict, Optional

log = logging.getLogger("damru.wire")

__all__ = ["wire_damru"]

_SAFE_BUILTINS = ("abs", "min", "max", "sum", "len", "range", "round", "print",
                  "pow", "sorted", "enumerate", "zip", "map", "filter", "int",
                  "float", "str", "list", "dict", "set", "tuple", "bool", "any",
                  "all", "divmod", "chr", "ord", "reversed", "format", "repr")


def _make_code_exec() -> Callable:
    """Restricted Python runner for the belt's code_exec tool (captures stdout)."""
    bset = vars(builtins)
    safe = {k: bset[k] for k in _SAFE_BUILTINS if k in bset}

    def _exec(code: Optional[str] = None, query: Optional[str] = None, **_) -> str:
        src = code or query or ""
        m = re.search(r"```(?:python|py)?\s*([\s\S]*?)```", src)
        if m:
            src = m.group(1)
        g: Dict[str, Any] = {"__builtins__": safe}
        try:
            import math as _math
            g["math"] = _math
        except Exception:
            pass
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(src, g, {})
            return buf.getvalue().strip() or "(ran ok, no stdout)"
        except Exception as e:
            return f"code error: {e}"
    return _exec


def _make_math_solver() -> Callable:
    def _solve(query: Optional[str] = None, expr: Optional[str] = None, **_) -> str:
        q = (expr or query or "").strip()
        if not q:
            return ""
        try:
            import sympy as sp
            from sympy.parsing.sympy_parser import parse_expr
            if "=" in q:
                left, right = q.split("=", 1)
                return str(sp.solve(sp.Eq(parse_expr(left), parse_expr(right))))
            return str(sp.simplify(parse_expr(q)))
        except Exception:
            try:
                return str(eval(q, {"__builtins__": {}}, {}))   # arithmetic fallback
            except Exception as e:
                return f"math error: {e}"
    return _solve


def _add_endpoints(api, belt, run_pipeline, llm) -> None:
    from fastapi import Body
    from fastapi.responses import StreamingResponse

    @api.get("/tools")
    def tools_list():
        return {"ok": True, "names": belt.names(), "card": belt.tool_card()}

    @api.post("/tools/run")
    def tools_run(body: dict = Body(...)):
        q = body.get("query") or body.get("message") or ""
        results = belt.smart_gather(q)
        return {"ok": True, "query": q, "plan": belt.plan(q),
                "results": [r.to_dict() for r in results]}

    @api.post("/chat_smart")
    def chat_smart(body: dict = Body(...)):
        q = body.get("message") or body.get("query") or ""
        use_llm = bool(body.get("llm_plan", True))

        def _gen():
            for ev in run_pipeline(q, belt=belt,
                                   llm_complete=(llm if use_llm else None)):
                yield "data: " + json.dumps(ev, default=str, ensure_ascii=False) + "\n\n"
        return StreamingResponse(_gen(), media_type="text/event-stream")


def wire_damru(api, g: Dict[str, Any]):
    """Wire Steps 1-3 into FastAPI `api`. `g` must be app.py's globals()."""
    from damru_tools import get_belt, Providers, run_pipeline

    llm             = g.get("_llm_complete")
    prayas_retrieve = g.get("prayas_retrieve")
    app_web_search  = g.get("web_search")
    cortex_answer   = g.get("cortex_answer")
    image_router    = g.get("image_router")
    DTWIN           = g.get("DTWIN")
    EMO             = g.get("EMO")

    prov: Dict[str, Callable] = {}

    # Step 2 : Live Internet -> web_search / deep_read / browse / watch_video
    net = None
    try:
        from damru_internet import get_internet, internet_providers
        net = get_internet(llm)
        prov.update(internet_providers(llm_complete=llm, internet=net))
    except Exception as e:
        log.info("internet providers skipped: %s", e)
        if app_web_search:
            prov["web_search"] = lambda query, **_: app_web_search(query, 4)

    # knowledge + rag  <- PRAYAS
    if prayas_retrieve:
        prov["knowledge"]  = lambda query, **_: prayas_retrieve(query)
        prov["rag_search"] = lambda query, **_: prayas_retrieve(query)

    # image + 3d
    if image_router:
        prov["image_gen"] = lambda prompt=None, query=None, **_: image_router(prompt or query or "")

        def _make3d(prompt=None, image_url=None, query=None, **_):
            from trellis_bridge import image_to_glb
            img = image_url or (image_router(prompt or query or "").get("imageUrl", ""))
            return image_to_glb(img) if img else None
        prov["make_3d"] = _make3d

    # code + math
    prov["code_exec"]  = _make_code_exec()
    prov["math_solve"] = _make_math_solver()

    # memory  <- DTwin
    if DTWIN is not None:
        def _memory(query=None, user_id="anonymous", response="", **_):
            try:
                DTWIN.observe(user_id=user_id, query=query or "", response=response,
                              intent="memory", emotion="", lang="en", meta={})
                return f"noted for {user_id}"
            except Exception as e:
                return f"memory error: {e}"
        prov["memory"] = _memory

    prov["llm_complete"] = llm

    belt = get_belt(Providers(**prov))
    log.info("belt wired -> %s", belt.names())

    # Step 2 router : /internet
    try:
        from damru_internet import build_internet_router
        api.include_router(build_internet_router(llm_complete=llm, internet=net))
    except Exception as e:
        log.info("internet router skipped: %s", e)

    # Step 3 router : /live
    try:
        from damru_live import build_live_router
        brain = None
        if cortex_answer:
            def brain(msg, user_id="anonymous", history=None):
                try:
                    return cortex_answer(msg, history=history, user_id=user_id)[0]
                except Exception:
                    return llm([{"role": "user", "content": msg}]) if llm else ""
        api.include_router(build_live_router(brain=brain, llm_complete=llm, emotion=EMO))
    except Exception as e:
        log.info("live router skipped: %s", e)

    # extra endpoints : /tools /tools/run /chat_smart
    try:
        _add_endpoints(api, belt, run_pipeline, llm)
    except Exception as e:
        log.info("extra endpoints skipped: %s", e)

    return belt


if __name__ == "__main__":
    # offline self-test: mock app.py globals; routers/endpoints skip (no fastapi in sandbox)
    print("code_exec 6*7 ->", _make_code_exec()(code="print(6*7)"))
    print("math 2+2*3   ->", _make_math_solver()(query="2+2*3"))

    def _llm(msgs, max_tokens=None):
        return "mock reply"

    def _prayas(q, k=5):
        return [{"text": "PRAYAS demo tile", "url": "", "score": 0.9, "topic": "demo"}]

    def _img(p, *a, **k):
        return {"ok": True, "imageUrl": "https://img/" + str(p).replace(" ", "_")}

    g = {"_llm_complete": _llm, "prayas_retrieve": _prayas,
         "web_search": lambda q, l=3: [], "image_router": _img,
         "cortex_answer": lambda m, history=None, user_id="a": ("cortex: " + m, {}),
         "DTWIN": None, "EMO": None}

    class _FakeAPI:
        def __init__(self):
            self.included = 0
            self.routes = []

        def include_router(self, r):
            self.included += 1

        def get(self, *a, **k):
            self.routes.append(("GET",) + a)
            return lambda f: f

        def post(self, *a, **k):
            self.routes.append(("POST",) + a)
            return lambda f: f

    api = _FakeAPI()
    belt = wire_damru(api, g)
    print("belt names ->", belt.names())
    print("routers included ->", api.included, "| extra routes ->", api.routes)
    print("plan 'compute 5*9' ->", [p["name"] for p in belt.plan("compute 5*9")])
    print("plan 'latest isro news' ->", [p["name"] for p in belt.plan("latest isro news")])
    print("OK damru_wire")
