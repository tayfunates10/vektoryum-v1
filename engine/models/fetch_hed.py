"""HED (Holistically-Nested Edge Detection) model dosyalarını indirir.

Kullanım::

    python models/fetch_hed.py

Dosyalar bu klasöre iner; motor onları otomatik bulur. Model açık kaynaklıdır
(Xie & Tu, ICCV 2015; BSD kaynaklı yayın ağırlıkları). İndirilmezse motor
derin kenar haritası olmadan, klasik yolla aynen çalışmaya devam eder.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/ashukid/hed-edge-detector/master"
FILES = {
    "deploy.prototxt": f"{BASE}/deploy.prototxt",
    "hed_pretrained_bsds.caffemodel": f"{BASE}/hed_pretrained_bsds.caffemodel",
}
HERE = Path(__file__).resolve().parent


def main() -> None:
    for name, url in FILES.items():
        dst = HERE / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[atla] {name} zaten var ({dst.stat().st_size} bayt)")
            continue
        print(f"[indir] {name} <- {url}")
        urllib.request.urlretrieve(url, dst)  # noqa: S310
        print(f"[tamam] {name} ({dst.stat().st_size} bayt)")


if __name__ == "__main__":
    main()
