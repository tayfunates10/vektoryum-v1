"""Kalıcı depolama: HF Dataset senkronu (ücretsiz, dayanıklı).

HF Spaces konteyner dosya sistemi GEÇİCİdir; kullanıcılar + geri-bildirim kaydı
restart/redeploy'da silinir. Bu modül DATA_ROOT'taki küçük JSON dosyalarını
(users.json, feedback.jsonl) özel bir HF Dataset'e yazar ve açılışta geri yükler.

Yapılandırma (ortam değişkenleri):
  VEKTORYUM_DATASET = "kullanici/vektoryum-data"   (Space'in yazabildiği dataset)
  HF_TOKEN          = HF write token               (Space secret olarak eklenir)

Yapılandırılmamışsa ya da huggingface_hub kurulu değilse: modül SESSİZCE no-op
olur ve yalnız yerel dosyalar kullanılır (yerel/test davranışı ve regresyon
fixture'ları değişmez). Yükleme/indirme hep BEST-EFFORT'tur: hata olsa bile
istek akışı kesilmez (yerele yazma her zaman çalışır).
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

_DATASET = os.environ.get("VEKTORYUM_DATASET", "").strip()
_TOKEN = (
    os.environ.get("HF_TOKEN", "").strip()
    or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    or os.environ.get("HUGGINGFACE_TOKEN", "").strip()
)
_repo_ready = False


def _api():
    """Yapılandırılmışsa HfApi döner; değilse None (no-op)."""
    if not (_DATASET and _TOKEN):
        return None
    try:
        from huggingface_hub import HfApi  # noqa: PLC0415
        return HfApi(token=_TOKEN)
    except Exception:  # noqa: BLE001 (kütüphane yok -> yerel-yalnız)
        return None


def enabled() -> bool:
    return _api() is not None


def _ensure_repo(api) -> bool:
    global _repo_ready
    if _repo_ready:
        return True
    try:
        api.create_repo(_DATASET, repo_type="dataset", private=True, exist_ok=True)
        _repo_ready = True
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("HF Dataset hazırlanamadı (%s): %s", _DATASET, e)
        return False


def restore(data_root: Path, files: list[str]) -> None:
    """Açılışta dataset'ten verilen dosyaları DATA_ROOT'a indirir (best-effort)."""
    api = _api()
    if api is None or not _ensure_repo(api):
        return
    from huggingface_hub import hf_hub_download  # noqa: PLC0415
    data_root.mkdir(parents=True, exist_ok=True)
    for name in files:
        try:
            path = hf_hub_download(_DATASET, name, repo_type="dataset", token=_TOKEN)
            (data_root / name).write_bytes(Path(path).read_bytes())
            logger.info("HF Dataset'ten geri yüklendi: %s", name)
        except Exception:  # noqa: BLE001 (dosya henüz yok -> normal, ilk çalıştırma)
            pass


def persist(local_path: Path, name: str) -> None:
    """Dosyayı dataset'e yükler — ARKA PLAN, best-effort (isteği bloklamaz)."""
    api = _api()
    if api is None:
        return

    def _up() -> None:
        try:
            if not _ensure_repo(api):
                return
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=name,
                repo_id=_DATASET,
                repo_type="dataset",
                commit_message=f"update {name}",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("HF Dataset yükleme atlandı (%s): %s", name, e)

    threading.Thread(target=_up, daemon=True).start()
