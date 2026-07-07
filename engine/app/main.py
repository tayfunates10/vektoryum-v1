"""Vektoryum.ai API - FastAPI giriş noktası.

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
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import Cookie, FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from PIL import Image, ImageChops, ImageStat

from app.exporters import export_all
from app.pipeline import run_pipeline
from app.quality import basic_svg_quality_check

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Vektoryum.ai API", version="2.0.0")

ALLOWED_MODES = [
    "auto", "geometric_logo", "minimal_ai", "logo_color",
    "flat_logo", "single_color", "lineart", "centerline", "photo_poster",
]

# Geriye dönük uyumluluk için ikinci ad (README'de geçer)
ALLOWED_TRACE_MODES = ALLOWED_MODES

DATA_ROOT = Path(os.environ.get("VEKTORYUM_DATA_ROOT", str(Path(tempfile.gettempdir()) / "vektoryum_data")))
JOBS_ROOT = Path(os.environ.get("VEKTORYUM_JOBS_ROOT", str(DATA_ROOT / "jobs")))
USERS_FILE = DATA_ROOT / "users.json"
SESSIONS: dict[str, dict[str, Any]] = {}
_HF_RESTORE_ATTEMPTED = False

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
            "name": "Vektoryum.ai Yönetici",
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


def _validate_job_id(job_id: str) -> None:
    if not job_id.isalnum():
        raise HTTPException(status_code=400, detail="Geçersiz job_id.")


def _hf_persist_token() -> str:
    return (
        os.environ.get("VEKTORYUM_HF_TOKEN")
        or os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or ""
    ).strip()


def _default_hf_persist_repo() -> str:
    space_id = os.environ.get("SPACE_ID", "").strip()
    if "/" not in space_id:
        return ""
    owner, name = space_id.split("/", 1)
    return f"{owner}/{name}-jobs"


def _hf_persist_repo() -> str:
    return os.environ.get("VEKTORYUM_HF_PERSIST_REPO", _default_hf_persist_repo()).strip()


def _hf_persist_repo_type() -> str:
    return os.environ.get("VEKTORYUM_HF_PERSIST_REPO_TYPE", "dataset").strip() or "dataset"


def _hf_persistence_enabled() -> bool:
    return bool(_hf_persist_token() and _hf_persist_repo())


def _sync_job_to_hub(job_id: str) -> None:
    """Runtime restart'larına karşı job klasörünü isteğe bağlı HF Hub'a kopyalar."""
    if not _hf_persistence_enabled():
        return

    from huggingface_hub import HfApi

    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        return

    token = _hf_persist_token()
    repo_id = _hf_persist_repo()
    repo_type = _hf_persist_repo_type()
    try:
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
        api.upload_folder(
            folder_path=str(job_dir),
            path_in_repo=f"jobs/{job_id}",
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
            commit_message=f"Persist Vektoryum.ai job {job_id}",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF job persistence sync failed for %s: %s", job_id, exc)


def _restore_jobs_from_hub_once() -> None:
    """Local jobs boşsa isteğe bağlı HF Hub deposundan işleri geri yükler."""
    global _HF_RESTORE_ATTEMPTED
    if _HF_RESTORE_ATTEMPTED or not _hf_persistence_enabled():
        return
    _HF_RESTORE_ATTEMPTED = True
    if any(JOBS_ROOT.glob("*/report.json")):
        return

    from huggingface_hub import snapshot_download

    snapshot_dir = DATA_ROOT / "hf_jobs_snapshot"
    try:
        snapshot_download(
            repo_id=_hf_persist_repo(),
            repo_type=_hf_persist_repo_type(),
            token=_hf_persist_token(),
            allow_patterns="jobs/**",
            local_dir=str(snapshot_dir),
        )
        source_jobs = snapshot_dir / "jobs"
        if not source_jobs.exists():
            return
        JOBS_ROOT.mkdir(parents=True, exist_ok=True)
        for source in source_jobs.iterdir():
            if source.is_dir() and (source / "report.json").exists():
                shutil.copytree(source, JOBS_ROOT / source.name, dirs_exist_ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("HF job persistence restore failed: %s", exc)


def _find_original_file(job_dir: Path) -> Path | None:
    for path in job_dir.glob("original.*"):
        if path.is_file():
            return path
    return None


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
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Desteklenmeyen dosya türü.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.load()
    except Exception as e:  # noqa: BLE001
        logger.error("Görsel okuma hatası: %s", e)
        raise HTTPException(status_code=400, detail="Görsel dosyası bozuk veya okunamıyor.")

    # iş klasörü
    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    original_path = job_dir / f"original{suffix}"
    original_path.write_bytes(contents)

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
    try:
        pipe = run_pipeline(image, original_path, trace_mode, job_dir, edge_cleanup=edge_cleanup_on)
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
    _sync_job_to_hub(job_id)
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

    # job_id güvenlik: sadece alfanümerik
    _validate_job_id(job_id)

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



def _quantized_color_count(image: Image.Image, max_side: int = 512) -> int:
    img = image.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
    colors = {(r // 16, g // 16, b // 16) for r, g, b in img.getdata()}
    return len(colors)


def _build_feedback_analysis(job_id: str, report: dict[str, Any]) -> dict[str, Any] | None:
    job_dir = _job_dir(job_id)
    original = _find_original_file(job_dir)
    rendered = job_dir / f"{job_id}.png"
    if not original or not rendered.exists():
        return None
    try:
        orig_img = Image.open(original).convert("RGB")
        out_img = Image.open(rendered).convert("RGB")
        if orig_img.size != out_img.size:
            out_img = out_img.resize(orig_img.size)
        diff = Image.eval(ImageChops.difference(orig_img, out_img), lambda p: p)
        stat = ImageStat.Stat(diff)
        mean_rgb = [round(float(v), 3) for v in stat.mean]
        err = diff.convert("L").point(lambda p: 255 if p > 35 else 0)
        bbox = err.getbbox()
        err_pixels = err.histogram()[255]
        total = orig_img.width * orig_img.height
        high_error_ratio = round(err_pixels / float(total), 5) if total else 0.0
        orig_q = _quantized_color_count(orig_img)
        out_q = _quantized_color_count(out_img)
        q = report.get("quality_report", {})
        structure = q.get("structure_report") or {}
        component_delta = structure.get("component_delta")
        notes: list[str] = []
        primary_issue = "minor_visual_difference"
        severity = "low"
        if orig_q > max(out_q * 1.15, 64) and out_q < 512:
            primary_issue = "smooth_gradient_banding"
            severity = "medium"
            notes.append("Orijinaldeki yumuşak ton/gradient geçişleri çıktıdaki sınırlı düz renk bantlarına dönüşmüş.")
        if high_error_ratio > 0.02:
            severity = "high"
            notes.append("Fark haritasında geniş alana yayılan görünür renk/kenar farkı var.")
        if component_delta not in (None, 0):
            notes.append(f"Bileşen sayısı değişmiş görünüyor: component_delta={component_delta}.")
        if not notes:
            notes.append("Farklar düşük seviyede; ana çıktı genel olarak orijinale yakın.")
        return {
            "primary_issue": primary_issue,
            "severity": severity,
            "mean_abs_rgb": mean_rgb,
            "high_error_ratio": high_error_ratio,
            "error_bbox": list(bbox) if bbox else None,
            "original_quantized_colors": orig_q,
            "output_quantized_colors": out_q,
            "notes": notes,
        }
    except Exception as exc:  # noqa: BLE001
        return {"primary_issue": "analysis_failed", "severity": "unknown", "error": str(exc)}


@app.get("/api/admin/jobs", summary="Yönetici iş listesi")
async def admin_jobs(session: str | None = Cookie(default=None)):
    _require_admin(session)
    _restore_jobs_from_hub_once()
    jobs = []
    if JOBS_ROOT.exists():
        for report in sorted(JOBS_ROOT.glob("*/report.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(report.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            q = data.get("quality_report", {})
            cr = data.get("candidate_report", {})
            jobs.append({
                "job_id": data.get("job_id"),
                "user": data.get("user"),
                "mode_used": data.get("mode_used"),
                "status": q.get("status"),
                "fidelity": (q.get("metrics") or {}).get("fidelity_score") or cr.get("best_score"),
                "best_candidate": cr.get("best_candidate"),
                "selection_reason": cr.get("selection_reason"),
                "warnings": q.get("warnings", []),
                "downloads": data.get("download_links", {}),
                "detail_url": f"/api/admin/jobs/{data.get('job_id')}",
                "original_url": f"/api/admin/download/{data.get('job_id')}/original" if _find_original_file(report.parent) else None,
            })
    return {"jobs": jobs}


@app.get("/api/admin/jobs/{job_id}", summary="Yönetici iş detayı")
async def admin_job_detail(job_id: str, session: str | None = Cookie(default=None)):
    _require_admin(session)
    _validate_job_id(job_id)
    report_path = _job_dir(job_id) / "report.json"
    if not report_path.exists():
        raise HTTPException(status_code=404, detail="İş raporu bulunamadı.")
    data = json.loads(report_path.read_text(encoding="utf-8"))
    data["admin_links"] = {
        "original": f"/api/admin/download/{job_id}/original" if _find_original_file(report_path.parent) else None,
        "report": f"/api/admin/jobs/{job_id}",
    }
    data["feedback_analysis"] = _build_feedback_analysis(job_id, data)
    return data


@app.get("/api/admin/download/{job_id}/original", summary="Yönetici orijinal görsel indirme")
async def admin_download_original(job_id: str, session: str | None = Cookie(default=None)):
    _require_admin(session)
    _validate_job_id(job_id)
    original = _find_original_file(_job_dir(job_id))
    if not original:
        raise HTTPException(status_code=404, detail="Orijinal görsel bulunamadı.")
    return FileResponse(original, media_type="image/*", filename=original.name)


@app.get("/admin", include_in_schema=False)
async def admin_page(session: str | None = Cookie(default=None)):
    _require_admin(session)
    admin_html = _STATIC_DIR / "admin.html"
    if not admin_html.exists():
        raise HTTPException(status_code=404, detail="Admin arayüzü bulunamadı.")
    return FileResponse(admin_html, media_type="text/html")

@app.get("/api/health", summary="Sağlık kontrolü")
async def health() -> dict[str, Any]:
    return {"status": "ok", "service": "vektoryum-api", "modes": ALLOWED_MODES}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
