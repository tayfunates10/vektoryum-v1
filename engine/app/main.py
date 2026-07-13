"""Vektoryum API - FastAPI giriş noktası.

Akış:
1. Analiz (analyzer)
2. Mod seçimi + uyarılar
3. Profil bazlı ön işleme (preprocess)
4. Çoklu aday üretimi (vector_engines)
5. Geometri temizleme (geometry_cleanup)
6. Skorlama (scoring)
7. Profil bazlı en iyi aday seçimi
8. Export: SVG / PDF / EPS / DXF (exporters)
9. Kalite raporu (quality)

Dayanıklılık: CairoSVG/Inkscape/Potrace/AutoTrace yoksa sistem çökmez; ilgili
adım atlanır ve hata raporlanır.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import logging
import os
import secrets
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image

from app.exporters import export_all
from app.input_guard import InputError, validate_and_load
from app.pipeline import WorkerFailure, run_pipeline
from app.quality import basic_svg_quality_check
from app.settings import get_settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vektoryum API", version="2.0.0")

ALLOWED_MODES = [
    "auto", "geometric_logo", "minimal_ai", "logo_color",
    "flat_logo", "single_color", "lineart", "centerline", "photo_poster",
]

# Geriye dönük uyumluluk için ikinci ad (README'de geçer)
ALLOWED_TRACE_MODES = ALLOWED_MODES

# Yol'lar env ile kalıcı bir konuma (ör. HF paid Persistent Storage /data)
# yönlendirilebilir; kalıcı disk yoksa küçük JSON'lar HF Dataset'e senkronlanır
# (app/store.py). Varsayılan geçici yol -> yerel/test davranışı değişmez.
JOBS_ROOT = Path(os.environ.get("VEKTORYUM_JOBS_ROOT", str(Path(tempfile.gettempdir()) / "vector_jobs")))
DATA_ROOT = Path(os.environ.get("VEKTORYUM_DATA_ROOT", str(Path(tempfile.gettempdir()) / "vektoryum_data")))
USERS_FILE = DATA_ROOT / "users.json"
FEEDBACK_FILE = DATA_ROOT / "feedback.jsonl"   # kalıcı geri-bildirim/iş kaydı
SESSIONS: dict[str, dict[str, Any]] = {}

_MEDIA_TYPES = {
    "svg": "image/svg+xml",
    "pdf": "application/pdf",
    "eps": "application/postscript",
    "dxf": "image/vnd.dxf",
    "png": "image/png",
}



def _load_users() -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    if not USERS_FILE.exists():
        users: dict[str, Any] = {}
        admin_email = os.environ.get("VEKTORYUM_ADMIN_EMAIL", "admin@vektoryum.local").lower().strip()
        admin_password = os.environ.get("VEKTORYUM_ADMIN_PASSWORD", "admin123")
        users[admin_email] = {
            "email": admin_email,
            "name": "Vektoryum Yönetici",
            "role": "admin",
            "password": _hash_password(admin_password),
        }
        USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        return json.loads(USERS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _save_users(users: dict[str, Any]) -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")
    from app import store  # noqa: PLC0415
    store.persist(USERS_FILE, "users.json")  # kalıcı depoya senkronla (best-effort)


def _append_feedback(record: dict[str, Any]) -> None:
    """Geri-bildirim/iş özetini kalıcı JSONL'e ekler + kalıcı depoya senkronlar.

    Admin paneli bu dosyadan okur; HF Dataset senkronu sayesinde restart'ta
    kaybolmaz. Yerele yazma her zaman çalışır; senkron best-effort.
    """
    try:
        DATA_ROOT.mkdir(parents=True, exist_ok=True)
        with FEEDBACK_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        from app import store  # noqa: PLC0415
        store.persist(FEEDBACK_FILE, "feedback.jsonl")
    except Exception as e:  # noqa: BLE001
        logger.warning("geri-bildirim kaydı atlandı: %s", e)


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
    return base64.b64encode(salt + digest).decode("ascii")


def _verify_password(password: str, encoded: str) -> bool:
    try:
        raw = base64.b64decode(encoded.encode("ascii"))
        salt, old = raw[:16], raw[16:]
        new = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120_000)
        return hmac.compare_digest(old, new)
    except Exception:  # noqa: BLE001
        return False


def _safe_user(user: dict[str, Any]) -> dict[str, str]:
    return {"email": user.get("email", ""), "name": user.get("name", ""), "role": user.get("role", "user")}


def _current_user(session: str | None) -> dict[str, Any] | None:
    if not session:
        return None
    sess = SESSIONS.get(session)
    if not sess:
        return None
    return _load_users().get(sess.get("email"))


def _require_user(session: str | None) -> dict[str, Any]:
    user = _current_user(session)
    if not user:
        raise HTTPException(status_code=401, detail="Devam etmek için giriş yapın.")
    return user


def _require_admin(session: str | None) -> dict[str, Any]:
    user = _require_user(session)
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Bu alan yalnızca yöneticiler içindir.")
    return user

def _job_dir(job_id: str) -> Path:
    return JOBS_ROOT / job_id


def _max_input_side() -> int:
    """VEKTORYUM_MAX_INPUT_SIDE (px). 0/geçersiz = küçültme kapalı."""
    try:
        return max(0, int(os.environ.get("VEKTORYUM_MAX_INPUT_SIDE", "0") or "0"))
    except ValueError:
        return 0


# ---------------------------------------------------------------------------
# /api/vectorize
# ---------------------------------------------------------------------------
@app.post("/api/vectorize", summary="Raster görseli vektöre dönüştürür")
async def vectorize_image(
    file: UploadFile = File(...),
    trace_mode: str = Form("auto"),
    # NOT: cutouts (gerçek evenodd delikleri) varsayılan YAPILMADI — pyclipper
    # boolean'ı eğrileri poligonize eder; LEGO fixture'ında komut sayısı
    # 339 -> 49.942'ye şişti (ölçüldü). Delikler için doğru hedef, eğri
    # KORUYAN sayaç-birleştirme (üstteki zemin-renkli örtme path'ini alttaki
    # path'e evenodd alt-yol olarak gömme); backlog'dadır. cutouts seçenek
    # olarak duruyor.
    shape_stacking: str = Form("stacked"),
    edge_cleanup: str = Form("on"),
    session: str | None = Cookie(default=None),
):
    user = _require_user(session)
    if not isinstance(shape_stacking, str):
        shape_stacking = "stacked"  # doğrudan (test) çağrıda Form varsayılanı nesne gelir
    if not isinstance(edge_cleanup, str):
        edge_cleanup = "on"
    # VARSAYILAN AÇIK; yalnız açıkça kapatılırsa devre dışı (ölçüm korumalı geçiş)
    edge_cleanup_on = edge_cleanup.lower() not in ("off", "false", "0", "no")
    if trace_mode not in ALLOWED_MODES:
        raise HTTPException(status_code=400, detail=f"Geçersiz trace_mode. İzin verilenler: {ALLOWED_MODES}")
    if shape_stacking not in ("stacked", "cutouts"):
        raise HTTPException(status_code=400, detail="Geçersiz shape_stacking. İzin verilenler: ['stacked', 'cutouts']")
    # GÜVENLİ ALIM: magic/format (istemci Content-Type'a güvenilmez), byte/piksel
    # sınırı, decompression bomb, animated reddi, EXIF transpose, ICC/CMYK->sRGB.
    contents = await file.read()
    try:
        loaded = validate_and_load(contents, file.filename)
    except InputError as e:
        return JSONResponse(status_code=e.status,
                            content={"error": e.message, "code": e.code})
    image = loaded.image

    # iş klasörü
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    # Kullanıcı filename'i YOL olarak KULLANILMAZ; sunucu-verilen güvenli uzantı.
    original_path = job_dir / f"original{loaded.safe_suffix}"
    if loaded.normalized:
        # EXIF/ICC normalizasyonu pikselleri değiştirdi → normalize görüntüyü yaz
        save_img = image if image.mode in ("RGB", "L", "RGBA") else image.convert("RGB")
        save_img.save(original_path)
    else:
        original_path.write_bytes(contents)   # değişmeyen kaynak → ham baytlar

    # Bellek-kısıtlı barındırma (ör. Render free 512MB) için OPSİYONEL girdi
    # küçültme: VEKTORYUM_MAX_INPUT_SIDE ayarlıysa ve en uzun kenar bunu aşıyorsa
    # görsel küçültülür ve kaynak dosya da tutarlı olması için yeniden yazılır.
    # Vektör izleme zaten ~1600px'de çalıştığından kayıp ihmal edilebilir; çok
    # büyük görselde OOM/timeout (HTML hata sayfası -> istemcide JSON hatası)
    # önlenir. Varsayılan KAPALI (0) — yerel/kütüphane davranışı değişmez.
    max_side = _max_input_side()
    if max_side and max(image.size) > max_side:
        sc = max_side / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * sc)), max(1, round(image.height * sc))),
            Image.LANCZOS,
        )
        save_img = image if image.mode in ("RGB", "L") else image.convert("RGB")
        try:
            save_img.save(original_path)
        except Exception:  # noqa: BLE001
            image.convert("RGB").save(original_path.with_suffix(".png"))
            original_path = original_path.with_suffix(".png")
        logger.info("Girdi %s'e küçültüldü (VEKTORYUM_MAX_INPUT_SIDE=%d)", image.size, max_side)

    # 1-7. Çekirdek pipeline (analiz → ön işleme → aday → temizleme → skor → seçim)
    # CPU-AĞIR iş event loop'u BLOKE ETMESİN: threadpool'a taşınır; böylece ağır
    # bir istek işlenirken /livez ve /api/auth/me yanıt vermeye devam eder (canlı
    # hata: 900² dizisinden sonra /api/health 30 sn'de yanıt vermiyordu). Gerçek
    # CPU işi zaten alt-süreç havuzunda; parent-thread'i loop'tan ayırmak yeterli.
    try:
        pipe = await run_in_threadpool(
            run_pipeline, image, original_path, trace_mode, job_dir,
            edge_cleanup=edge_cleanup_on,
        )
    except WorkerFailure as e:
        # Paralel aday havuzu başarısız/zaman aşımı — KONTROLLÜ (inline rerun yok).
        logger.error("Worker havuzu hatası: %s", e)
        return JSONResponse(
            status_code=503,
            content={"error": "İşleme kapasitesi geçici olarak yetersiz, tekrar deneyin.",
                     "code": "worker_failure", "job_id": job_id},
            headers={"Retry-After": "10"},
        )
    except Exception as e:  # noqa: BLE001
        logger.error("Pipeline hatası: %s", e)
        raise HTTPException(status_code=500, detail=f"İşlem başarısız: {e}")

    analysis = pipe["analysis"]
    mode_used = pipe["mode_used"]
    mode_warning = pipe["mode_warning"]
    preprocess_report = pipe["preprocess_report"]
    results = pipe["results"]
    scored = pipe["scored"]

    if not scored:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Hiçbir vektör adayı üretilemedi.",
                "job_id": job_id,
                "mode_used": mode_used,
                "candidate_report": {
                    "candidates": [
                        {"name": r["name"], "success": False, "error": r.get("error")}
                        for r in results
                    ],
                },
            },
        )

    best = pipe["best"]
    raw_best = pipe["raw_best"]
    selection_reason = pipe["selection_reason"]

    # 7.5 Shape stacking dönüşümü (istenirse): stacked -> cut-outs. Kopya
    # üzerinde çalışılır; başarısız olursa stacked çıktı aynen kullanılır.
    export_source = Path(best["svg_path"])
    stacking_report = {"mode": "stacked"}
    if shape_stacking == "cutouts":
        from shutil import copyfile

        from app.cutouts import convert_svg_to_cutouts

        cut_svg = job_dir / f"{best['name']}_cutouts.svg"
        copyfile(export_source, cut_svg)
        result = convert_svg_to_cutouts(cut_svg)
        stacking_report = {"mode": "cutouts", **result}
        if result.get("status") in ("completed", "no_change"):
            export_source = cut_svg
        else:
            stacking_report["fallback"] = "stacked"

    # 8. Export ("temizlenmiş" PNG dahil; boyut = orijinal görsel boyutu)
    best_geo = best.get("cleanup_report", {}).get("report", {})
    outputs, output_errors = export_all(
        best_svg=export_source,
        job_dir=job_dir,
        job_id=job_id,
        candidate_id=f"{mode_used}:{best['name']}",
        png_size=(int(analysis.get("width", 0)) or None, int(analysis.get("height", 0)) or None),
    )

    # 9. Kalite raporu (yapı bütünlüğü dahil: kırık/eksik çizgi denetimi)
    quality_report = basic_svg_quality_check(
        score_details=best.get("score_details", {}),
        mode=mode_used,
        geometry_report=best_geo,
        total_score=best["total_score"],
        fidelity_score=best.get("fidelity_score"),
        structure_report=pipe.get("structure_report"),
    )

    download_links = {fmt: f"/api/download/{job_id}/{fmt}" for fmt in ("svg", "pdf", "eps", "dxf", "png")}

    final_report = {
        "job_id": job_id,
        "user": {"email": user.get("email"), "name": user.get("name")},
        "mode_used": mode_used,
        "mode_warning": mode_warning,
        "analysis": analysis,
        "preprocess": {"steps": preprocess_report.get("steps", []), "palette": preprocess_report.get("palette", [])},
        "candidate_report": {
            "best_candidate": best["name"],
            "best_score": best["total_score"],
            "raw_best_candidate": raw_best["name"],
            "raw_best_score": raw_best["total_score"],
            "selection_reason": selection_reason,
            "candidates": [
                {
                    "name": (c.get("name")),
                    "success": c.get("success", False),
                    "error": c.get("error"),
                    "engine": c.get("engine"),
                    "total_score": c.get("total_score"),
                    "color_score": c.get("color_score"),
                    "edge_score": c.get("edge_score"),
                    "detail_score": c.get("detail_score"),
                    "path_score": c.get("path_score"),
                    "warning_score": c.get("warning_score"),
                    "straight_edge_score": c.get("straight_edge_score"),
                    "corner_cleanliness_score": c.get("corner_cleanliness_score"),
                    "axis_alignment_score": c.get("axis_alignment_score"),
                    "geometry_score": c.get("geometry_score"),
                    "rendered_ok": c.get("rendered_ok"),
                    "fidelity_score": c.get("fidelity_score"),
                    "details": c.get("score_details"),
                }
                # başarısız adaylar da raporlanır
                for c in _merge_for_report(scored, results)
            ],
        },
        "quality_report": quality_report,
        "refine_info": pipe.get("refine_info"),
        "refit_info": pipe.get("refit_info"),
        "shape_stacking": stacking_report,
        "outputs": {fmt: Path(p).name for fmt, p in outputs.items()},
        "output_errors": output_errors,
        "download_links": download_links,
    }

    (job_dir / "report.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    # KALICI geri-bildirim kaydı: admin paneli bunu okur, restart'ta kaybolmaz.
    _append_feedback({
        "job_id": job_id,
        "ts": int(time.time()),
        "user": {"email": user.get("email"), "name": user.get("name")},
        "mode_used": mode_used,
        "status": quality_report.get("status"),
        "fidelity": best.get("fidelity_score"),
        "best_candidate": best.get("name"),
        "selection_reason": selection_reason,
        "warnings": quality_report.get("warnings", []),
        "download_links": download_links,
    })
    return JSONResponse(content=final_report)


def _merge_for_report(scored: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Skorlanan adaylar + başarısız adayları tek listede birleştirir.

    Refinement'ta üretilen adaylar ``results`` içinde olmayabilir; onları da
    sona ekleriz ki rapor (ve seçilen kazanan) eksik kalmasın.
    """
    scored_by_name = {c["name"]: c for c in scored}
    merged = []
    seen: set[str] = set()
    for r in results:
        seen.add(r["name"])
        merged.append(scored_by_name.get(r["name"], r))
    for c in scored:
        if c["name"] not in seen:
            merged.append(c)
    return merged


# ---------------------------------------------------------------------------
# /api/download/{job_id}/{file_type}
# ---------------------------------------------------------------------------
@app.get("/api/download/{job_id}/{file_type}", summary="Üretilen vektör dosyasını indir")
async def download_file(job_id: str, file_type: str):
    if file_type not in _MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya formatı.")

    # job_id güvenlik: sadece hex
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Geçersiz job_id.")

    file_path = _job_dir(job_id) / f"{job_id}.{file_type}"
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"'{file_type}' dosyası bu iş için üretilmedi (export başarısız olmuş olabilir).",
        )

    return FileResponse(
        file_path,
        media_type=_MEDIA_TYPES[file_type],
        filename=f"{job_id}.{file_type}",
    )


_STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", summary="Web arayüzü", include_in_schema=False)
async def index():
    """Kökte web arayüzü servis edilir; statik dosya yoksa JSON sağlık raporu
    döner (eski davranış — API-yalnız kurulumlar bozulmaz)."""
    index_html = _STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html, media_type="text/html")
    return JSONResponse({"status": "ok", "service": "vektoryum-api", "modes": ALLOWED_MODES})



@app.post("/api/auth/register", summary="Kullanıcı kaydı")
async def register(payload: dict[str, str], response: Response):
    email = (payload.get("email") or "").lower().strip()
    name = (payload.get("name") or "").strip()
    password = payload.get("password") or ""
    if not email or "@" not in email or len(password) < 6:
        raise HTTPException(status_code=400, detail="Geçerli e-posta ve en az 6 karakter şifre girin.")
    users = _load_users()
    if email in users:
        raise HTTPException(status_code=409, detail="Bu e-posta zaten kayıtlı.")
    users[email] = {"email": email, "name": name or email.split("@")[0], "role": "user", "password": _hash_password(password)}
    _save_users(users)
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {"email": email}
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 14)
    return {"user": _safe_user(users[email])}


@app.post("/api/auth/login", summary="Kullanıcı / yönetici girişi")
async def login(payload: dict[str, str], response: Response):
    email = (payload.get("email") or "").lower().strip()
    password = payload.get("password") or ""
    user = _load_users().get(email)
    if not user or not _verify_password(password, user.get("password", "")):
        raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı.")
    token = secrets.token_urlsafe(32)
    SESSIONS[token] = {"email": email}
    response.set_cookie("session", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 14)
    return {"user": _safe_user(user), "admin_url": "/admin" if user.get("role") == "admin" else None}


@app.post("/api/auth/logout", summary="Çıkış")
async def logout(response: Response, session: str | None = Cookie(default=None)):
    if session:
        SESSIONS.pop(session, None)
    response.delete_cookie("session")
    return {"ok": True}


@app.get("/api/auth/me", summary="Aktif kullanıcı")
async def me(session: str | None = Cookie(default=None)):
    user = _current_user(session)
    return {"user": _safe_user(user) if user else None}


@app.get("/api/admin/jobs", summary="Yönetici iş listesi")
async def admin_jobs(session: str | None = Cookie(default=None)):
    _require_admin(session)
    # KALICI geri-bildirim kaydından okunur (HF Dataset ile senkron -> restart'ta
    # kaybolmaz). Geçici /tmp iş klasörlerini taramaz; en yeni en üstte.
    jobs = []
    if FEEDBACK_FILE.exists():
        try:
            lines = FEEDBACK_FILE.read_text(encoding="utf-8").splitlines()
        except Exception:  # noqa: BLE001
            lines = []
        for ln in reversed(lines):
            ln = ln.strip()
            if not ln:
                continue
            try:
                d = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            jobs.append({
                "job_id": d.get("job_id"),
                "user": d.get("user"),
                "mode_used": d.get("mode_used"),
                "status": d.get("status"),
                "fidelity": d.get("fidelity"),
                "best_candidate": d.get("best_candidate"),
                "selection_reason": d.get("selection_reason"),
                "warnings": d.get("warnings", []),
                "downloads": d.get("download_links", {}),
            })
    return {"jobs": jobs}


@app.on_event("startup")
def _restore_persisted_state() -> None:
    """Açılışta kalıcı depodan (HF Dataset) users.json + feedback.jsonl indir."""
    try:
        from app import store  # noqa: PLC0415
        store.restore(DATA_ROOT, ["users.json", "feedback.jsonl"])
    except Exception as e:  # noqa: BLE001
        logger.warning("kalıcı durum geri yüklenemedi: %s", e)
    _load_users()  # restore sonrası admin kullanıcısını garanti et


@app.get("/admin", include_in_schema=False)
async def admin_page(session: str | None = Cookie(default=None)):
    _require_admin(session)
    return HTMLResponse('<!doctype html><html lang="tr"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Vektoryum Admin</title><style>body{margin:0;background:#0b1020;color:#eaf0ff;font:14px system-ui}.wrap{max-width:1180px;margin:auto;padding:28px}.top{display:flex;justify-content:space-between;align-items:center}.card{background:linear-gradient(180deg,#111a35,#0e1530);border:1px solid rgba(255,255,255,.12);border-radius:16px;padding:16px;margin:14px 0}.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.badge{padding:5px 10px;border-radius:999px;background:rgba(75,141,255,.16);color:#8db6ff;font-weight:700}.warn{color:#fbbf24}.ok{color:#34d399}a{color:#9cc2ff}.muted{color:#93a1c4}.btn{border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);color:#fff;border-radius:10px;padding:9px 12px;cursor:pointer}.downloads{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}pre{white-space:pre-wrap;color:#cbd5ff}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style></head><body><div class="wrap"><div class="top"><div><h1>Vektoryum Admin Paneli</h1><p class="muted">Beta çıktıları, kalite raporları ve otomatik hata inceleme kuyruğu.</p></div><button class="btn" onclick="logout()">Çıkış</button></div><div id="jobs"></div></div><script>async function logout(){await fetch(\'/api/auth/logout\',{method:\'POST\'});location.href=\'/\'}function row(j){const st=j.status===\'production_ready\'?\'ok\':\'warn\';const d=j.downloads||{};return `<div class="card"><div class="grid"><div><b>İş:</b> ${j.job_id}<br><b>Kullanıcı:</b> ${(j.user&&j.user.email)||\'-\'}<br><b>Mod:</b> ${j.mode_used||\'-\'} · <b>Aday:</b> ${j.best_candidate||\'-\'}<br><b>Seçim:</b> ${j.selection_reason||\'-\'}</div><div><span class="badge ${st}">${j.status||\'bilinmiyor\'}</span><p><b>Skor:</b> ${j.fidelity==null?\'-\':j.fidelity}</p><p class="muted">Uyarılar: ${(j.warnings||[]).join(\' · \')||\'-\'}</p></div></div><div class="downloads">${Object.entries(d).map(([k,v])=>`<a href="${v}" target="_blank">${k.toUpperCase()}</a>`).join(\'\')}</div><pre>Otomatik analiz önerisi: renk farkı, kenar uyumsuzluğu, eksik/fazla detay ve kalite uyarıları bu iş raporundan incelenir.</pre></div>`}async function load(){const r=await fetch(\'/api/admin/jobs\');if(!r.ok){location.href=\'/\';return}const data=await r.json();document.getElementById(\'jobs\').innerHTML=(data.jobs||[]).map(row).join(\'\')||\'<div class="card">Henüz iş yok.</div>\'}load()</script></body></html>')

@app.get("/livez", summary="Liveness (event loop yaşıyor mu)")
async def livez() -> dict[str, Any]:
    """HIZLI liveness: yalnız API event loop/process yaşadığını gösterir.

    Ağır bir vektörleştirme işlenirken bile (iş threadpool + alt-süreç havuzunda)
    bu uç anında yanıt vermelidir. Ağır I/O/DB kontrolü YAPMAZ."""
    return {"status": "alive", "service": "vektoryum-api"}


def _readiness() -> tuple[bool, dict[str, Any]]:
    """DB/disk yerine bu sürümde: yazılabilir artifact alanı + temel kontrol."""
    checks: dict[str, Any] = {}
    ok = True
    try:
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        probe = JOBS_ROOT / ".readyz_probe"
        probe.write_text("ok")
        probe.unlink(missing_ok=True)
        checks["artifact_writable"] = True
    except Exception as e:  # noqa: BLE001
        checks["artifact_writable"] = False
        checks["artifact_error"] = str(e)
        ok = False
    return ok, checks


@app.get("/readyz", summary="Readiness (istek almaya hazır mı)")
async def readyz() -> JSONResponse:
    """Yazılabilir artifact alanı vb. hazır değilse 503 döner."""
    ok, checks = await run_in_threadpool(_readiness)
    return JSONResponse(status_code=200 if ok else 503,
                        content={"status": "ready" if ok else "not_ready", "checks": checks})


@app.get("/api/health", summary="Sağlık kontrolü")
async def health() -> JSONResponse:
    """Geriye uyum: readiness sonucuna bağlı (hazır değilse 503)."""
    ok, checks = await run_in_threadpool(_readiness)
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"status": "ok" if ok else "degraded", "service": "vektoryum-api",
                 "modes": ALLOWED_MODES, "checks": checks})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
