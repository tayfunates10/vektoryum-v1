"""FAZ 3A — provenance-aware artwork kimliği (kör path/node sayımının yerine).

Painter, kaynak alfayı yeniden üretmek için transform-OWNED geometri ekler
(luminance maskesi, destek çizgisi, karşılaştırma-tuvali arşivi). Eski doğrulama
final SVG'nin TOPLAM path/node sayısını parent ile birebir karşılaştırıyordu; bu,
`<path>` tabanlı kompakt maske kodlamalarını (FAZ 3B) imkânsız kılıyordu.

Bu modül kimliği iki ayrı sözleşmeye böler:

1. **Artwork identity** — sanat eserinin geometri + renk kimliği. Transformun
   ASLA değiştirmediği nitelikler üzerinden kanonik parmak izi: local-name,
   render/DOM sırası, geometri (`d`/`points`/koordinatlar), `fill`, `fill-rule`,
   `transform`, `clip-path`/`mask`/`filter` referansları, `visibility`/`display`
   ve referans edilen gradient/pattern/use tanımları (stop offset/renk/opaklık,
   gradientUnits/Transform, spreadMethod, viewBox/preserveAspectRatio). Bu yüzey
   transform tarafından birebir korunur → tam string eşitliği güvenlidir
   (koordinat yuvarlaması YOK, farklı geometri asla aynı sayılmaz).

2. **Transform-owned complexity** — maske geometrisi ve destek yüzeyi
   (`stroke*`, `paint-order`, `opacity`, `fill-opacity`, `stroke-opacity`).
   Bunlar `_expand_candidate_paint`/`_strip_content_alpha` tarafından ölçülü
   biçimde yeniden yazılır; GÖRSEL etkileri değişmemiş SSIM/edge/seam/topology
   journal kapılarına aittir. Parmak izinden hariçtir; ama serbest DEĞİLdir:
   final SVG'nin toplam byte/path/node karmaşıklığı mevcut journal sınırlarınca
   bağlanmaya devam eder.

Kabul = artwork parmak izi AYNI **ve** toplam karmaşıklık mevcut sınırlar içinde.
"""
from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from typing import Any

_SVG_NS = "http://www.w3.org/2000/svg"
_XLINK_NS = "http://www.w3.org/1999/xlink"

# Provenance nitelikleri — YALNIZ transformun kendi oluşturduğu node'lara yazılır.
PROVENANCE_OWNER_ATTR = "data-vektoryum-owner"
PROVENANCE_ROLE_ATTR = "data-vektoryum-role"
PROVENANCE_TXN_ATTR = "data-vektoryum-alpha-transaction"
PROVENANCE_OWNER = "source-alpha-transform"

# Roller. "artwork-container" transformun oluşturduğu ama İÇİNDE sanat eserini
# taşıyan saydam kaptır: kabın kendisi parmak izine girmez, çocukları (sanat)
# girer. Diğer roller saf transform geometrisidir → alt ağaç tümüyle hariç.
ROLE_ARTWORK_CONTAINER = "artwork-preserved-container"
ROLE_MASK_DEFINITION = "alpha-mask-definition"
ROLE_MASK_GEOMETRY = "alpha-mask-geometry"
ROLE_MASK_APPLICATION = "alpha-mask-application"
ROLE_CANVAS_KNOCKOUT = "comparison-canvas-knockout"
ROLE_CANVAS_UNDERPAINT = "comparison-canvas-underpaint"

_SKIP_SUBTREE_ROLES = frozenset(
    {
        ROLE_MASK_DEFINITION,
        ROLE_MASK_GEOMETRY,
        ROLE_MASK_APPLICATION,
        ROLE_CANVAS_KNOCKOUT,
        ROLE_CANVAS_UNDERPAINT,
    }
)

# Alfa yüzeyi: `_strip_content_alpha` bunları TÜM sanat düğümlerinden soyar →
# her zaman kimlik parmak izinden hariç (görsel etkisi journal alfa kapısına ait).
_ALWAYS_OWNED_ATTRS = frozenset(
    {"opacity", "fill-opacity", "stroke-opacity", "paint-order"}
)

# Stroke destek yüzeyi: `_expand_candidate_paint` bunları YALNIZ dolgulu (fill!=none)
# geometride destek çizgisine yeniden yazar. Bu yüzden stroke-only sanat eserinde
# (fill=none) KORUNUR ve kimliğe dahildir; dolgulu geometride transform-owned'dır.
_STROKE_SURFACE_ATTRS = frozenset(
    {
        "stroke",
        "stroke-width",
        "stroke-linejoin",
        "stroke-linecap",
        "stroke-dasharray",
        "stroke-dashoffset",
    }
)

_GEOMETRY_TAGS = frozenset(
    {"path", "rect", "circle", "ellipse", "polygon", "polyline"}
)
_FILL_NONE = frozenset({"none", "transparent", "rgba(0,0,0,0)"})

# Provenance nitelikleri parmak izine asla girmez (transformun meta işareti).
_PROVENANCE_ATTRS = frozenset(
    {PROVENANCE_OWNER_ATTR, PROVENANCE_ROLE_ATTR, PROVENANCE_TXN_ATTR}
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def alpha_transaction_id(
    parent_sha256: str,
    source_alpha_sha256: str,
    mode: str,
    encoding: str,
) -> str:
    """Deterministik işlem kimliği — sabit girdilerden türetilir (rastgele UUID YOK).

    Aynı parent + kaynak alfa + mod + kodlama her zaman aynı kimliği verir; bu
    sayede artifact SHA'sı deterministik kalır ve iki çalıştırma birebir eşleşir.
    """
    digest = hashlib.sha256(
        "".join(
            [
                str(parent_sha256),
                str(source_alpha_sha256),
                str(mode),
                str(encoding),
            ]
        ).encode("utf-8")
    ).hexdigest()
    return digest[:24]


def tag_transform_node(
    element: ET.Element, role: str, transaction_id: str
) -> ET.Element:
    """Bir node'u transform-owned olarak işaretle (owner + role + transaction)."""
    element.set(PROVENANCE_OWNER_ATTR, PROVENANCE_OWNER)
    element.set(PROVENANCE_ROLE_ATTR, role)
    element.set(PROVENANCE_TXN_ATTR, transaction_id)
    return element


def _owned_by_transaction(element: ET.Element, transaction_id: str) -> bool:
    return (
        element.get(PROVENANCE_OWNER_ATTR) == PROVENANCE_OWNER
        and element.get(PROVENANCE_TXN_ATTR) == transaction_id
    )


def _element_fill(element: ET.Element) -> str | None:
    fill = element.get("fill")
    if fill is None:
        for declaration in (element.get("style") or "").split(";"):
            name, _sep, value = declaration.partition(":")
            if name.strip() == "fill":
                fill = value.strip()
                break
    return fill


def _stroke_is_transform_owned(element: ET.Element) -> bool:
    """Bu elemanın stroke yüzeyini painter destek çizgisi olarak yeniden yazar mı?

    Yalnız dolgulu (fill != none) geometri elemanlarında; stroke-only sanat
    eserinde (fill=none) stroke korunur → kimliğe dahildir.
    """
    if _local_name(str(element.tag)).lower() not in _GEOMETRY_TAGS:
        return False
    fill = _element_fill(element)
    if fill is not None and fill.strip().lower() in _FILL_NONE:
        return False
    return True


def _owned_attr_names(element: ET.Element) -> frozenset:
    owned = _ALWAYS_OWNED_ATTRS
    if _stroke_is_transform_owned(element):
        owned = owned | _STROKE_SURFACE_ATTRS
    return owned


def _style_pairs(style: str, owned: frozenset) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for declaration in style.split(";"):
        if ":" not in declaration:
            continue
        name, _sep, value = declaration.partition(":")
        name = name.strip()
        value = value.strip()
        if not name or name in owned:
            continue
        pairs.append((name, " ".join(value.split())))
    pairs.sort()
    return pairs


def _canonical_attrs(element: ET.Element) -> list[tuple[str, str]]:
    """Kimlik yüzeyindeki nitelikler — sıralı, transform-owned + provenance hariç.

    Değerler transform tarafından birebir korunur; yalnız boşluk normalize edilir,
    sayısal yuvarlama YAPILMAZ (farklı geometrinin çakışmasını önlemek için). Stroke
    yüzeyi yalnız dolgulu geometride hariç tutulur (painter destek çizgisi); alfa
    yüzeyi (`opacity`/`fill-opacity`/`stroke-opacity`/`paint-order`) her zaman hariç.
    """
    owned = _owned_attr_names(element)
    result: list[tuple[str, str]] = []
    for raw_name, value in element.attrib.items():
        name = _local_name(str(raw_name))
        if name in _PROVENANCE_ATTRS or raw_name in _PROVENANCE_ATTRS:
            continue
        if name in owned:
            continue
        if name == "style":
            pairs = _style_pairs(str(value), owned)
            if pairs:
                result.append(("style", ";".join(f"{k}:{v}" for k, v in pairs)))
            continue
        # xlink:href ve href aynı semantik anahtara indirgenir.
        key = "href" if name == "href" else name
        result.append((key, " ".join(str(value).split())))
    result.sort()
    return result


def _reference_ids(element: ET.Element) -> list[str]:
    """Bu elemanın işaret ettiği tanım id'leri (fill=url(#g), href=#p, clip-path…)."""
    ids: list[str] = []
    for raw_name, value in element.attrib.items():
        name = _local_name(str(raw_name))
        text = str(value)
        if name in ("href",):
            if text.startswith("#"):
                ids.append(text[1:])
        for match in re.findall(r"url\(#([^)\s]+)\)", text):
            ids.append(match)
    return ids


def _descriptor(element: ET.Element) -> str:
    name = _local_name(str(element.tag))
    attrs = _canonical_attrs(element)
    return name + "|" + "&".join(f"{k}={v}" for k, v in attrs)


def _collect_definitions(root: ET.Element) -> dict[str, ET.Element]:
    return {
        str(element.get("id")): element
        for element in root.iter()
        if element.get("id")
    }


def _definition_fingerprint(
    definition: ET.Element,
    definitions: dict[str, ET.Element],
    seen: set[str],
) -> str:
    """Referans edilen tanımın (gradient/pattern/use hedefi) kanonik parmak izi."""
    element_id = str(definition.get("id") or "")
    if element_id and element_id in seen:
        return f"@cycle:{element_id}"
    if element_id:
        seen = seen | {element_id}
    parts = [_descriptor(definition)]
    for child in list(definition):
        parts.append(
            _definition_fingerprint(child, definitions, seen)
        )
    for ref in _reference_ids(definition):
        target = definitions.get(ref)
        if target is not None:
            parts.append("->" + _definition_fingerprint(target, definitions, seen))
    return "(" + ";".join(parts) + ")"


_PROTECTED_ROOT_TAGS = frozenset(
    {"defs", "title", "desc", "metadata", "style"}
)


def _find_artwork_container(
    root: ET.Element, transaction_id: str
) -> ET.Element | None:
    for element in root.iter():
        if (
            _owned_by_transaction(element, transaction_id)
            and element.get(PROVENANCE_ROLE_ATTR) == ROLE_ARTWORK_CONTAINER
        ):
            return element
    return None


def _artwork_children(
    element: ET.Element,
    transaction_id: str,
    excluded: frozenset,
) -> list[ET.Element]:
    """Bir elemanın sanat eseri çocukları: transform-owned saf geometri atlanır,
    saydam artwork-container hoist edilir (relatif sıra korunur)."""
    children: list[ET.Element] = []
    for child in list(element):
        if id(child) in excluded:
            continue
        if _owned_by_transaction(child, transaction_id):
            role = child.get(PROVENANCE_ROLE_ATTR)
            if role == ROLE_ARTWORK_CONTAINER:
                children.extend(
                    _artwork_children(child, transaction_id, excluded)
                )
            # mask/geometry/application/knockout/bilinmeyen → alt ağaç atlanır
            continue
        children.append(child)
    return children


def _node_repr(
    element: ET.Element,
    definitions: dict[str, ET.Element],
    transaction_id: str,
    excluded: frozenset,
) -> str:
    parts = [_descriptor(element)]
    for ref in _reference_ids(element):
        target = definitions.get(ref)
        if target is not None:
            parts.append("->" + _definition_fingerprint(target, definitions, set()))
    child_reprs = [
        _node_repr(child, definitions, transaction_id, excluded)
        for child in _artwork_children(element, transaction_id, excluded)
    ]
    return "(" + "|".join(parts) + "[" + ",".join(child_reprs) + "])"


def artwork_fingerprint(
    root: ET.Element,
    transaction_id: str,
    excluded_elements: Any = (),
) -> str:
    """Sanat eserinin konum-bağımsız kanonik geometri+renk parmak izi.

    Sanat eseri ORMANI içerikten bulunur (konumdan değil): aday tarafta bu işleme
    ait ``artwork-preserved-container`` çocukları; parent tarafta kök seviyesindeki
    movable çocuklar (defs/protected hariç). ``excluded_elements`` parent tarafında
    knockout edilecek kanıtlı karşılaştırma-tuvalini dışlar (aday tarafta o zaten
    arşive taşınıp maske olarak işaretlidir), böylece meşru arka-plan knockout'u
    kimlik ihlali sayılmaz.

    Transform-owned saf geometri alt ağaçları atlanır; artwork-container saydamdır.
    Descriptor, transformun ASLA değiştirmediği kimlik yüzeyini birebir (yuvarlama
    yok) yakalar; stroke/opacity destek yüzeyi hariçtir. Referans edilen
    gradient/pattern/use tanımları çözülerek dahil edilir. BAŞKA/sahte transaction
    etiketi taşıyan node dışlanmaz — yalnız bu id transform-owned sayılır.
    """
    excluded = frozenset(id(element) for element in excluded_elements)
    definitions = _collect_definitions(root)
    container = _find_artwork_container(root, transaction_id)
    if container is not None:
        forest = _artwork_children(container, transaction_id, excluded)
    else:
        forest = [
            child
            for child in list(root)
            if id(child) not in excluded
            and _local_name(str(child.tag)).lower() not in _PROTECTED_ROOT_TAGS
            and not _owned_by_transaction(child, transaction_id)
        ]
    payload = "\n".join(
        _node_repr(element, definitions, transaction_id, excluded)
        for element in forest
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def artwork_identity_report(
    parent_fingerprint: str,
    candidate_root: ET.Element,
    transaction_id: str,
) -> dict[str, Any]:
    """Aday sanat kimliğini parent parmak izi ile karşılaştır (fail-closed)."""
    candidate_fingerprint = artwork_fingerprint(candidate_root, transaction_id)
    return {
        "artwork_identity_preserved": candidate_fingerprint == parent_fingerprint,
        "parent_artwork_fingerprint": parent_fingerprint,
        "candidate_artwork_fingerprint": candidate_fingerprint,
        "alpha_transaction_id": transaction_id,
    }
