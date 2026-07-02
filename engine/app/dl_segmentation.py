"""Opsiyonel derin öğrenme kenar/segmentasyon katmanı (HED, cv2.dnn).

Holistically-Nested Edge Detection (Xie & Tu, ICCV 2015) açık kaynak caffe
modelini OpenCV DNN ile CPU'da çalıştırır ve 0..1 aralığında ANLAMSAL kenar
haritası üretir. Sobel'den farkı: JPEG gürültüsü ve doku pürüzü kenar sayılmaz,
gerçek nesne/yazı sınırları güçlü yanıt verir — kuantizasyonun "düz bölge /
yapılı bölge" kararları bununla belirginleşir.

Projenin dayanıklılık felsefesi: model dosyaları yoksa her fonksiyon ``None``
döner ve çağıran taraf klasik (Sobel tabanlı) yola güvenle düşer; API çökmez,
yeni zorunlu bağımlılık yoktur (cv2.dnn OpenCV çekirdeğindedir).

Model dosyaları (``models/fetch_hed.py`` ile indirilebilir):
* ``models/deploy.prototxt``
* ``models/hed_pretrained_bsds.caffemodel``

Ortam değişkenleriyle farklı yol verilebilir::

    HED_PROTO_PATH=/path/deploy.prototxt
    HED_MODEL_PATH=/path/hed_pretrained_bsds.caffemodel
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import OrderedDict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_NET = None
_NET_TRIED = False

# küçük LRU: refinement aynı görseli birden çok kez ön işler; kenar haritası
# içerik anahtarıyla önbelleklenir ki HED her varyantta yeniden koşmasın
_CACHE: OrderedDict[str, np.ndarray] = OrderedDict()
_CACHE_MAX = 8


def _model_paths() -> tuple[Path, Path] | None:
    proto = Path(os.environ.get("HED_PROTO_PATH", _MODELS_DIR / "deploy.prototxt"))
    weights = Path(os.environ.get("HED_MODEL_PATH", _MODELS_DIR / "hed_pretrained_bsds.caffemodel"))
    if proto.exists() and weights.exists():
        return proto, weights
    return None


def _load_net():
    """HED ağını bir kez yükler; yoksa/yüklenemezse None (çökme yok)."""
    global _NET, _NET_TRIED
    if _NET_TRIED:
        return _NET
    _NET_TRIED = True
    paths = _model_paths()
    if paths is None:
        logger.info("HED modeli bulunamadı; derin kenar haritası devre dışı (klasik yol).")
        return None
    try:
        _NET = cv2.dnn.readNetFromCaffe(str(paths[0]), str(paths[1]))
        logger.info("HED derin kenar modeli yüklendi: %s", paths[1].name)
    except Exception as e:  # noqa: BLE001
        logger.warning("HED modeli yüklenemedi (%s); klasik yola düşülüyor.", e)
        _NET = None
    return _NET


def is_available() -> bool:
    return _load_net() is not None


def _content_key(rgb: np.ndarray) -> str:
    thumb = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_AREA)
    return f"{rgb.shape}:{hashlib.sha1(thumb.tobytes()).hexdigest()}"


def compute_edge_map(rgb: np.ndarray, max_side: int = 512) -> np.ndarray | None:
    """RGB görüntü için 0..1 anlamsal kenar haritası (girdiyle aynı boyutta).

    Model yoksa/başarısızsa ``None`` döner. Hız için ağ en fazla ``max_side``
    çözünürlükte koşar; sonuç girdi boyutuna bilinear ölçeklenir. Refinement
    varyantları için içerik anahtarlı küçük bir LRU önbellek kullanılır.
    """
    net = _load_net()
    if net is None:
        return None
    try:
        key = _content_key(rgb)
        if key in _CACHE:
            _CACHE.move_to_end(key)
            return _CACHE[key]

        h, w = rgb.shape[:2]
        scale = min(1.0, max_side / float(max(h, w)))
        sw, sh = max(16, round(w * scale)), max(16, round(h * scale))
        small = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(
            cv2.cvtColor(small, cv2.COLOR_RGB2BGR), scalefactor=1.0, size=(sw, sh),
            mean=(104.00699, 116.66877, 122.67891), swapRB=False, crop=False,
        )
        net.setInput(blob)
        out = net.forward()[0, 0]
        edge = np.clip(out, 0.0, 1.0).astype(np.float32)
        if edge.shape != (h, w):
            edge = cv2.resize(edge, (w, h), interpolation=cv2.INTER_LINEAR)

        _CACHE[key] = edge
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
        return edge
    except Exception as e:  # noqa: BLE001
        logger.warning("HED kenar haritası üretilemedi (%s); klasik yola düşülüyor.", e)
        return None
