from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except Exception:
    TfidfVectorizer = None
    cosine_similarity = None

from backend.config import DATA_FILE


# =========================================================
# 1) NORMALIZATION
# =========================================================

PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    LATIN_DIGITS + LATIN_DIGITS,
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""

    text = str(value).translate(DIGIT_TRANSLATION)

    text = (
        text
        .replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
        .replace("ـ", "")
    )

    text = text.lower()

    text = re.sub(r"[–—−]", "-", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def flatten_value(value: Any) -> list[str]:
    result: list[str] = []

    if value is None:
        return result

    if isinstance(value, dict):
        for key, item in value.items():
            result.append(str(key))
            result.extend(flatten_value(item))
        return result

    if isinstance(value, (list, tuple, set)):
        for item in value:
            result.extend(flatten_value(item))
        return result

    result.append(str(value))
    return result


# =========================================================
# 2) MATERIALS
# =========================================================

MATERIAL_ALIASES = {
    "فولاد": (
        "فولاد",
        "فولادی",
        "steel",
    ),

    "استیل": (
        "استیل",
        "stainless steel",
        "stainless",
        "فولاد ضد زنگ",
        "فولاد زنگ نزن",
        "ضد زنگ",
    ),

    "فولاد ضد زنگ": (
        "فولاد ضد زنگ",
        "فولاد زنگ نزن",
        "استیل",
        "stainless steel",
        "stainless",
        "ضد زنگ",
    ),

    "چدن": (
        "چدن",
        "cast iron",
        "cast-iron",
    ),

    "آلومینیوم": (
        "آلومینیوم",
        "aluminium",
        "aluminum",
    ),

    "مس": (
        "مس",
        "copper",
    ),

    "برنج": (
        "برنج",
        "brass",
    ),

    "تیتانیوم": (
        "تیتانیوم",
        "titanium",
    ),
}


# وقتی کاربر جنس قطعه را مشخص کرده، وجود این مواد به‌عنوان
# ناسازگار/متفاوت می‌تواند محصول را رد کند.
MATERIAL_FAMILIES = {
    "steel": (
        "فولاد",
        "فولادی",
        "steel",
        "stainless",
        "stainless steel",
        "استیل",
        "فولاد ضد زنگ",
        "فولاد زنگ نزن",
        "ضد زنگ",
    ),

    "cast_iron": (
        "چدن",
        "cast iron",
        "cast-iron",
    ),

    "aluminum": (
        "آلومینیوم",
        "aluminium",
        "aluminum",
    ),

    "copper": (
        "مس",
        "copper",
    ),

    "brass": (
        "برنج",
        "brass",
    ),

    "titanium": (
        "تیتانیوم",
        "titanium",
    ),

    "wood": (
        "چوب",
        "wood",
    ),

    "plastic": (
        "پلاستیک",
        "plastic",
    ),

    "concrete": (
        "بتن",
        "concrete",
    ),

    "masonry": (
        "مصالح",
        "masonry",
        "brick",
        "آجر",
    ),
}


def detect_material(
    query: str,
) -> tuple[str | None, tuple[str, ...]]:
    text = normalize_text(query)

    for canonical in sorted(
        MATERIAL_ALIASES,
        key=len,
        reverse=True,
    ):
        aliases = MATERIAL_ALIASES[canonical]

        if any(
            normalize_text(alias) in text
            for alias in aliases
        ):
            return canonical, aliases

    return None, ()


# =========================================================
# 3) TOOL MATERIAL
# =========================================================

TOOL_MATERIAL_ALIASES = {
    "hss": (
        "hss",
        "hss-r",
        "hssr",
        "hss-g",
        "hssg",
        "hssco",
        "hssco5",
        "hss-e",
        "تندبر",
        "فولاد تندبر",
    ),

    "cobalt": (
        "کبالت",
        "کبالت دار",
        "کبالت‌دار",
        "cobalt",
        "hssco",
        "hssco5",
    ),

    "carbide": (
        "کارباید",
        "کاربید",
        "carbide",
        "solid carbide",
    ),

    "carbide_tipped": (
        "نوک کارباید",
        "کارباید دار",
        "کاربید دار",
        "tipped carbide",
        "carbide tipped",
    ),
}


def detect_tool_material(
    query: str,
) -> tuple[str | None, tuple[str, ...]]:
    text = normalize_text(query)

    # ابتدا دقیق‌ترین گزینه‌ها
    for canonical in (
        "carbide_tipped",
        "carbide",
        "cobalt",
        "hss",
    ):
        aliases = TOOL_MATERIAL_ALIASES[canonical]

        if any(
            normalize_text(alias) in text
            for alias in aliases
        ):
            return canonical, aliases

    return None, ()


def product_has_tool_material(
    product: dict,
    aliases: tuple[str, ...],
) -> bool:
    evidence = _product_evidence_text(product)

    return any(
        normalize_text(alias) in evidence
        for alias in aliases
    )


# =========================================================
# 4) INTENT
# =========================================================

INTENT_KEYWORDS = {

    "drilling": (
        "مته",
        "سوراخ",
        "سوراخکاری",
        "سوراخ کاری",
        "دریل",
        "drill",
        "drilling",
    ),

    "tapping": (
        "قلاویز",
        "رزوه",
        "رزوه زنی",
        "رزوه‌زنی",
        "دنده زنی",
        "دنده‌زنی",
        "tap",
        "tapping",
        "thread",
        "threading",
    ),

    "milling": (
        "فرز",
        "فرزکاری",
        "فرز کاری",
        "فرز انگشتی",
        "milling",
        "end mill",
    ),

    "turning": (
        "تراش",
        "تراشکاری",
        "تراش کاری",
        "تراش کار",
        "turning",
        "lathe",
    ),

    "measurement": (
        "اندازه گیری",
        "اندازه‌گیری",
        "میکرومتر",
        "کولیس",
        "اندیکاتور",
        "measurement",
        "caliper",
        "indicator",
    ),

    "sharpening": (
        "تیز کن",
        "تیزکنی",
        "تیز کردن",
        "تیزکن",
        "sharpen",
        "sharpening",
    ),

    "holding": (
        "هولدر",
        "نگهدارنده",
        "کولت",
        "فشنگی",
        "سه نظام",
        "سه‌نظام",
        "چاک",
        "holder",
        "collet",
        "chuck",
    ),
}


PRODUCT_INTENT_ALIASES = {

    "drilling": (
        "مته",
        "drill",
        "drilling",
        "دریل",
    ),

    "tapping": (
        "قلاویز",
        "tap",
        "tapping",
    ),

    "milling": (
        "فرز",
        "فرزکاری",
        "فرز انگشتی",
        "milling",
        "end mill",
    ),

    "turning": (
        "تراش",
        "تراشکاری",
        "turning",
        "lathe",
    ),

    "measurement": (
        "میکرومتر",
        "کولیس",
        "اندیکاتور",
        "measurement",
        "caliper",
        "indicator",
    ),

    "sharpening": (
        "تیز کن",
        "تیزکنی",
        "sharpen",
        "sharpening",
    ),

    "holding": (
        "هولدر",
        "holder",
        "کولت",
        "collet",
        "چاک",
        "chuck",
        "فشنگی",
        "سه نظام",
        "سه‌نظام",
    ),
}


FORBIDDEN_BY_INTENT = {

    "drilling": (
        "قلاویز",
        "رزوه",
        "فرز انگشتی",
        "هولدر",
        "holder",
    ),

    "tapping": (
        "مته",
        "drill",
        "فرز انگشتی",
    ),

    "milling": (
        "قلاویز",
        "مته مرغک",
        "مته مرکز",
    ),

    "turning": (
        "قلاویز",
    ),

    "measurement": (
        "قلاویز",
        "مته",
    ),

    "sharpening": (),
    "holding": (),
}


def detect_intent(
    query: str,
) -> str | None:
    text = normalize_text(query)

    scores: list[tuple[int, str]] = []

    for intent, keywords in INTENT_KEYWORDS.items():

        score = sum(
            1
            for word in keywords
            if normalize_text(word) in text
        )

        if score:
            scores.append(
                (
                    score,
                    intent,
                )
            )

    if not scores:
        return None

    scores.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )

    return scores[0][1]


# =========================================================
# 5) DIAMETER
# =========================================================

def extract_diameter(
    query: str,
) -> float | None:

    text = normalize_text(query)

    patterns = [

        # قطر 30
        r"(?:قطر|سوراخ|سایز)\s*[:=]?\s*"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:میلی\s*متر|mm)?",

        # 30 mm
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:میلی\s*متر|mm)\b",

        # مته 30
        r"(?:مته|drill)\s*"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:میلی\s*متر|mm)?",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if not match:
            continue

        try:
            return float(
                match.group(1).replace(",", ".")
            )
        except ValueError:
            continue

    return None


# =========================================================
# 6) THREAD
# =========================================================

def extract_thread(
    query: str,
) -> dict[str, float | None] | None:

    text = normalize_text(query)

    patterns = (
        r"\bm\s*(\d{1,3})"
        r"(?:\s*[x×*]\s*"
        r"(\d+(?:[.,]\d+)?))?"
        r"\b",

        r"رزوه\s*m\s*(\d{1,3})"
        r"(?:\s*[x×*]\s*"
        r"(\d+(?:[.,]\d+)?))?",
    )

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if not match:
            continue

        nominal = float(
            match.group(1)
        )

        pitch = None

        if match.group(2):
            try:
                pitch = float(
                    match.group(2).replace(",", ".")
                )
            except ValueError:
                pitch = None

        return {
            "nominal": nominal,
            "pitch": pitch,
        }

    return None


# =========================================================
# 7) THREAD HAND
# =========================================================

def detect_thread_hand(
    query: str,
) -> str | None:

    text = normalize_text(query)

    left_terms = (
        "چپ گرد",
        "چپ‌گرد",
        "چپگرد",
        "رزوه چپ",
        "قلاویز چپ",
        "left hand",
        "left-hand",
        "left hand tap",
        "lh",
    )

    right_terms = (
        "راست گرد",
        "راست‌گرد",
        "راستگرد",
        "رزوه راست",
        "قلاویز راست",
        "right hand",
        "right-hand",
        "right hand tap",
    )

    if any(
        term in text
        for term in left_terms
    ):
        return "left"

    if any(
        term in text
        for term in right_terms
    ):
        return "right"

    return None


def product_thread_hand(
    product: dict,
) -> str | None:

    evidence = _thread_evidence_text(product)

    left_terms = (
        "چپ گرد",
        "چپ‌گرد",
        "چپگرد",
        "left hand",
        "left-hand",
        "left hand tap",
        "lh",
    )

    right_terms = (
        "راست گرد",
        "راست‌گرد",
        "راستگرد",
        "right hand",
        "right-hand",
        "right hand tap",
    )

    if any(
        term in evidence
        for term in left_terms
    ):
        return "left"

    if any(
        term in evidence
        for term in right_terms
    ):
        return "right"

    return None


# =========================================================
# 8) RANGES
# =========================================================

def _extract_ranges(
    text: str,
) -> list[tuple[float, float]]:

    result: list[
        tuple[float, float]
    ] = []

    normalized = normalize_text(text)

    pattern = re.compile(
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:تا|الی|-|~|–|—)\s*"
        r"(\d+(?:[.,]\d+)?)"
    )

    for match in pattern.finditer(
        normalized
    ):

        try:

            low = float(
                match.group(1).replace(",", ".")
            )

            high = float(
                match.group(2).replace(",", ".")
            )

        except ValueError:
            continue

        if low > high:
            low, high = high, low

        result.append(
            (
                low,
                high,
            )
        )

    return result


def _numeric_keys(
    value: Any,
) -> list[float]:

    result: list[float] = []

    if not isinstance(value, dict):
        return result

    for key in value.keys():

        key_text = normalize_text(key).strip()

        if re.fullmatch(
            r"\d+(?:[.,]\d+)?",
            key_text,
        ):

            try:
                result.append(
                    float(
                        key_text.replace(",", ".")
                    )
                )
            except ValueError:
                pass

    return result


# =========================================================
# 9) PRODUCT DIAMETER EXTRACTION
# =========================================================

def extract_product_diameters(
    product: dict,
) -> list[
    tuple[float, float]
]:

    ranges: list[
        tuple[float, float]
    ] = []

    name = normalize_text(
        product.get(
            "name",
            "",
        )
    )

    ranges.extend(
        _extract_ranges(name)
    )

    # فقط فیلدهایی که معمولاً ابعاد فنی دارند
    for field in (
        "attributes",
        "technical_specs",
    ):

        value = product.get(
            field,
            {},
        )

        if isinstance(value, dict):

            # فقط بازه‌های صریح مثل 2-10، 1 تا 13،
            # 10-20 و ... را بررسی می‌کنیم.
            #
            # کلیدهای عددی مثل "20" به‌تنهایی قطر محسوب
            # نمی‌شوند، چون ممکن است کد، طول، مدل یا
            # اندازه دیگری از محصول باشند.

            for item in value.values():

                ranges.extend(
                    _extract_ranges(
                        str(item)
                    )
                )
            # مقادیر مثل 2-10 / 10 تا 25
            for item in value.values():
                ranges.extend(
                    _extract_ranges(
                        str(item)
                    )
                )

    evidence_fields: list[str] = []

    for field in (
        "short_description",
        "description",
        "features",
        "applications",
        "search_text",
    ):

        evidence_fields.extend(
            flatten_value(
                product.get(field)
            )
        )

    for text in evidence_fields:

        normalized = normalize_text(text)

        for match in re.finditer(
            r"(?:سایز|قطر|diameter)\s*"
            r"(\d+(?:[.,]\d+)?)",
            normalized,
        ):

            try:

                value = float(
                    match.group(1).replace(",", ".")
                )

                ranges.append(
                    (
                        value,
                        value,
                    )
                )

            except ValueError:
                pass

        ranges.extend(
            _extract_ranges(
                normalized
            )
        )

    deduped: list[
        tuple[float, float]
    ] = []

    seen = set()

    for item in ranges:

        key = (
            round(item[0], 6),
            round(item[1], 6),
        )

        if key not in seen:

            seen.add(key)
            deduped.append(item)

    return deduped


def product_matches_diameter(
    product: dict,
    requested_diameter: float,
) -> bool:

    ranges = extract_product_diameters(
        product
    )

    if not ranges:
        return False

    for low, high in ranges:

        if (
            low - 1e-9
            <= requested_diameter
            <= high + 1e-9
        ):
            return True

    return False


# =========================================================
# 10) PRODUCT TEXT
# =========================================================

def _product_evidence_text(
    product: dict,
) -> str:

    parts: list[str] = []

    for field in (
        "name",
        "short_description",
        "description",
        "features",
        "advantages",
        "applications",
        "attributes",
        "technical_specs",
        "search_text",
    ):

        parts.extend(
            flatten_value(
                product.get(field)
            )
        )

    return normalize_text(
        " ".join(parts)
    )


def _thread_evidence_text(
    product: dict,
) -> str:

    parts: list[str] = []

    for field in (
        "name",
        "features",
        "attributes",
        "technical_specs",
        "description",
        "applications",
    ):

        parts.extend(
            flatten_value(
                product.get(field)
            )
        )

    return normalize_text(
        " ".join(parts)
    )


# =========================================================
# 11) PRODUCT TYPE / INTENT FILTER
# =========================================================
def product_matches_intent(
    product: dict,
    intent: str,
) -> bool:

    name = normalize_text(
        product.get("name", "")
    )

    category = normalize_text(
        product.get("category", "")
    )

    text = f"{name} {category}".strip()

    forbidden = FORBIDDEN_BY_INTENT.get(
        intent,
        (),
    )

    if any(
        normalize_text(word) in text
        for word in forbidden
    ):
        return False

    # -------------------------
    # DRILLING
    # -------------------------

    if intent == "drilling":

        excluded_terms = (
            "مته مرغک",
            "مرغک",
            "مته مرکز",
            "center drill",
            "center-drill",
            "spot drill",
            "spot-drill",

            "مته تیز کن",
            "مته تیزکن",
            "مته‌تیزکن",
            "تیز کن مته",
            "تیزکن مته",
            "دستگاه مته تیز کن",
            "دستگاه مته تیزکن",
            "drill sharpener",
            "drill sharpening",
            "sharpening machine",
            "sharpener",
        )

        if any(
            term in text
            for term in excluded_terms
        ):
            return False

        drilling_terms = (
            "مته",
            "drill",
        )

        return any(
            term in name
            for term in drilling_terms
        )

    # -------------------------
    # TAPPING
    # -------------------------

    if intent == "tapping":

        non_tap_thread_tools = (
            "دنده تراش",
            "thread mill",
            "thread milling",
            "فرز رزوه",
            "threading insert",
            "اینسرت رزوه",
        )

        if any(
            term in text
            for term in non_tap_thread_tools
        ):
            return False

        tapping_terms = (
            "قلاویز",
            "tap",
        )

        return any(
            term in name
            for term in tapping_terms
        )

    # -------------------------
    # MILLING
    # -------------------------

    if intent == "milling":

        non_milling_tools = (
            "کولت",
            "collet",
            "هولدر",
            "holder",
            "نگهدارنده",
            "فشنگی",
            "سه نظام",
            "سه‌نظام",
            "چاک",
            "chuck",
            "ابزارگیر",
            "tool holder",
            "toolholder",
            "دنباله",
            "adapter",
            "آداپتور",
            "رابط",

            # دستگاه‌های تیزکننده نباید ابزار فرز محسوب شوند
            "تیز کن",
            "تیزکن",
            "تیزکنی",
            "تیز کردن",
            "دستگاه فرز تیزکن",
            "دستگاه فرز انگشتی تیزکن",
            "دستگاه تیز کن فرز",
            "دستگاه تیزکن فرز",
            "end mill sharpener",
            "endmill sharpener",
            "sharpening machine",
            "sharpener",
        )

        if any(
            term in name
            for term in non_milling_tools
        ):
            return False



        return any(
            term in name
            for term in milling_terms
        )

    # -------------------------
    # TURNING
    # -------------------------

    if intent == "turning":

        turning_terms = (
            "تراش",
            "تراشکاری",
            "turning",
            "lathe",
        )

        return any(
            term in name
            for term in turning_terms
        )

    # -------------------------
    # MEASUREMENT
    # -------------------------

    if intent == "measurement":

        measurement_terms = (
            "میکرومتر",
            "کولیس",
            "اندیکاتور",
            "caliper",
            "indicator",
            "micrometer",
        )

        return any(
            term in name
            for term in measurement_terms
        )

    # -------------------------
    # SHARPENING
    # -------------------------

    if intent == "sharpening":

        sharpening_terms = (
            "تیز کن",
            "تیزکن",
            "تیزکنی",
            "sharpener",
            "sharpening",
        )

        return any(
            term in name
            for term in sharpening_terms
        )

    # -------------------------
    # HOLDING
    # -------------------------

    if intent == "holding":

        holding_terms = (
            "هولدر",
            "holder",
            "کولت",
            "collet",
            "فشنگی",
            "سه نظام",
            "سه‌نظام",
            "چاک",
            "chuck",
            "ابزارگیر",
            "tool holder",
            "toolholder",
            "دنباله",
        )

        return any(
            term in name
            for term in holding_terms
        )

    return False
# =========================================================
# 12) MATERIAL MATCHING
# =========================================================

def _detect_product_material_families(
    product: dict,
) -> set[str]:

    evidence = _product_evidence_text(product)

    result: set[str] = set()

    for family, aliases in MATERIAL_FAMILIES.items():

        if any(
            normalize_text(alias) in evidence
            for alias in aliases
        ):
            result.add(family)

    return result


def product_matches_material(
    product: dict,
    canonical_material: str | None,
    aliases: tuple[str, ...],
) -> bool:

    if not canonical_material:
        return True

    evidence = _product_evidence_text(product)

    # حالت ۱: جنس دقیقاً در اطلاعات محصول آمده
    if any(
        normalize_text(alias) in evidence
        for alias in aliases
    ):
        return True

    families = _detect_product_material_families(
        product
    )

    # نگاشت جنس درخواستی
    requested_family = {
        "فولاد": "steel",
        "استیل": "steel",
        "فولاد ضد زنگ": "steel",
        "چدن": "cast_iron",
        "آلومینیوم": "aluminum",
        "مس": "copper",
        "برنج": "brass",
        "تیتانیوم": "titanium",
    }.get(canonical_material)

    if requested_family is None:
        return True

    # اگر محصول جنس دیگری را به‌صراحت معرفی کرده باشد،
    # محصول مشکوک/نامرتبط است.
    incompatible_families = {
        "wood",
        "plastic",
        "concrete",
        "masonry",
    }

    if families.intersection(incompatible_families):
        return False

    # اگر برای فلز دیگری به‌صورت انحصاری معرفی شده
    other_metal_families = {
        "cast_iron",
        "aluminum",
        "copper",
        "brass",
        "titanium",
    }

    if requested_family == "steel":

        explicit_other = (
            families
            .intersection(other_metal_families)
        )

        # اگر هم فولاد ندارد و هم صریحاً برای فلز دیگری است
        if explicit_other:
            return False

    elif requested_family != "steel":

        explicit_other = (
            families
            .intersection(
                other_metal_families
                - {requested_family}
            )
        )

        if explicit_other:
            return False

    # جنس در دیتای محصول ذکر نشده،
    # ولی هیچ ناسازگاری صریح هم نداریم.
    # برای جلوگیری از False Negative، اینجا قبول می‌کنیم
    # و در امتیازدهی وزن کمتری می‌دهیم.
    return True


# =========================================================
# 13) THREAD MATCHING
# =========================================================

def product_matches_thread(
    product: dict,
    requested_thread: dict[
        str,
        float | None,
    ],
) -> bool:

    nominal = requested_thread["nominal"]
    pitch = requested_thread["pitch"]

    evidence = _thread_evidence_text(
        product
    )

    nominal_text = str(int(nominal))

    matches = list(
        re.finditer(
            rf"\bm\s*"
            rf"{re.escape(nominal_text)}"
            rf"(?:\s*[x×*]\s*"
            rf"(\d+(?:[.,]\d+)?))?"
            rf"\b",
            evidence,
        )
    )

    if not matches:
        return False

    # اگر pitch کاربر مشخص نکرده،
    # وجود Mxx کافی است.
    if pitch is None:
        return True

    for match in matches:

        product_pitch = match.group(1)

        if product_pitch is None:
            continue

        try:

            value = float(
                product_pitch.replace(",", ".")
            )

        except ValueError:

            continue

        if abs(
            value - pitch
        ) < 1e-9:

            return True

    return False


# =========================================================
# 14) QUERY TOKENS
# =========================================================

def _query_tokens(
    text: str,
) -> list[str]:

    tokens = re.findall(
        r"[a-zA-Z0-9\u0600-\u06FF]+",
        normalize_text(text),
    )

    return [
        token
        for token in tokens
        if len(token) >= 2
    ]


# =========================================================
# 15) RAG CLASS
# =========================================================

class ProductRAG:

    def __init__(
        self,
        data_file: Path = DATA_FILE,
    ):

        self.data_file = Path(
            data_file
        )

        self.products: list[
            dict
        ] = []

        self.matrix = None
        self.vectorizer = None
        self.ready = False

        self._load()

    # -----------------------------------------------------
    # LOAD
    # -----------------------------------------------------

    def _load(self) -> None:

        if not self.data_file.exists():

            raise FileNotFoundError(
                f"Product data file not found: "
                f"{self.data_file}"
            )

        with self.data_file.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(
            data,
            list,
        ):

            raise ValueError(
                "Product data file must "
                "contain a JSON list."
            )

        self.products = [
            product
            for product in data
            if isinstance(
                product,
                dict,
            )
        ]

        texts = []

        for product in self.products:

            search_text = product.get(
                "search_text"
            )

            if not search_text:

                search_text = " ".join(
                    flatten_value(
                        {
                            "name": product.get(
                                "name",
                                "",
                            ),
                            "brand": product.get(
                                "brand",
                                "",
                            ),
                            "category": product.get(
                                "category",
                                "",
                            ),
                            "description": product.get(
                                "description",
                                "",
                            ),
                            "features": product.get(
                                "features",
                                "",
                            ),
                            "applications": product.get(
                                "applications",
                                "",
                            ),
                        }
                    )
                )

            texts.append(
                normalize_text(
                    search_text
                )
            )

        if (
            TfidfVectorizer is not None
            and texts
        ):

            self.vectorizer = (
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                )
            )

            self.matrix = (
                self.vectorizer.fit_transform(
                    texts
                )
            )

        self.ready = bool(
            self.products
        )

    # -----------------------------------------------------
    # PARSE REQUEST
    # -----------------------------------------------------

    def parse_request(
        self,
        query: str,
    ) -> dict:

        material, material_aliases = (
            detect_material(query)
        )

        tool_material, tool_material_aliases = (
            detect_tool_material(query)
        )

        thread = extract_thread(query)

        return {
            "intent": detect_intent(query),
            "diameter": extract_diameter(query),
            "thread": thread,
            "material": material,
            "material_aliases": material_aliases,
            "tool_material": tool_material,
            "tool_material_aliases": tool_material_aliases,
            "thread_hand": detect_thread_hand(query),
        }

    # -----------------------------------------------------
    # TF-IDF
    # -----------------------------------------------------

    def _tfidf_scores(
        self,
        query: str,
    ) -> np.ndarray:

        if (
            self.vectorizer is None
            or self.matrix is None
        ):

            return np.zeros(
                len(self.products),
                dtype=float,
            )

        vector = (
            self.vectorizer.transform(
                [normalize_text(query)]
            )
        )

        return cosine_similarity(
            vector,
            self.matrix,
        ).ravel()

    # -----------------------------------------------------
    # KEYWORD SCORE
    # -----------------------------------------------------

    def _keyword_score(
        self,
        query: str,
        product: dict,
    ) -> float:

        query_tokens = set(
            _query_tokens(query)
        )

        if not query_tokens:
            return 0.0

        name = normalize_text(
            product.get(
                "name",
                "",
            )
        )

        category = normalize_text(
            product.get(
                "category",
                "",
            )
        )

        search_text = normalize_text(
            product.get(
                "search_text",
                "",
            )
        )

        score = 0.0

        for token in query_tokens:

            if token in name:
                score += 2.0

            elif token in category:
                score += 1.0

            elif token in search_text:
                score += 0.5

        return score

    # -----------------------------------------------------
    # SEMANTIC SCORE
    # -----------------------------------------------------

    def _semantic_score(
        self,
        query: str,
        product: dict,
        tfidf_score: float,
    ) -> float:

        parsed = self.parse_request(query)

        score = float(
            tfidf_score
        )

        intent = parsed["intent"]

        name = normalize_text(
            product.get(
                "name",
                "",
            )
        )

        category = normalize_text(
            product.get(
                "category",
                "",
            )
        )

        text = f"{name} {category}"

        # -----------------------------------------------
        # Intent
        # -----------------------------------------------

        if intent:

            keywords = (
                INTENT_KEYWORDS.get(
                    intent,
                    (),
                )
            )

            if any(
                normalize_text(keyword) in text
                for keyword in keywords
            ):
                score += 0.25

        # -----------------------------------------------
        # Diameter
        # -----------------------------------------------

        diameter = parsed["diameter"]

        if (
            intent == "drilling"
            and diameter is not None
            and product_matches_diameter(
                product,
                diameter,
            )
        ):

            score += 0.60

        # -----------------------------------------------
        # Thread
        # -----------------------------------------------

        thread = parsed["thread"]

        if (
            intent == "tapping"
            and thread is not None
            and product_matches_thread(
                product,
                thread,
            )
        ):

            score += 0.60

        # -----------------------------------------------
        # Material
        # -----------------------------------------------

        material = parsed["material"]
        material_aliases = parsed[
            "material_aliases"
        ]

        if material:

            evidence = _product_evidence_text(
                product
            )

            if any(
                normalize_text(alias) in evidence
                for alias in material_aliases
            ):
                score += 0.30

        # -----------------------------------------------
        # Tool material
        # -----------------------------------------------

        tool_material_aliases = parsed[
            "tool_material_aliases"
        ]

        if tool_material_aliases:

            if product_has_tool_material(
                product,
                tool_material_aliases,
            ):
                score += 0.45

        # -----------------------------------------------
        # HSS/HSSCO bonus for steel drilling
        # -----------------------------------------------

        if (
            intent == "drilling"
            and material
            in {
                "فولاد",
                "استیل",
                "فولاد ضد زنگ",
            }
        ):

            evidence = _product_evidence_text(
                product
            )

            if any(
                term in evidence
                for term in (
                    "hss",
                    "hssco",
                    "hssco5",
                    "کبالت",
                    "کبالت دار",
                )
            ):
                score += 0.15

        return score

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:

        query = query.strip()

        if not query:
            return []

        top_k = max(
            1,
            min(top_k, 10),
        )

        parsed = self.parse_request(query)

        intent = parsed["intent"]
        diameter = parsed["diameter"]
        thread = parsed["thread"]
        material = parsed["material"]
        material_aliases = parsed[
            "material_aliases"
        ]
        tool_material_aliases = parsed[
            "tool_material_aliases"
        ]
        thread_hand = parsed[
            "thread_hand"
        ]

        tfidf_scores = (
            self._tfidf_scores(query)
        )

        candidates: list[
            tuple[float, int, dict]
        ] = []

        for index, product in enumerate(
            self.products
        ):

            # =========================================
            # HARD FILTER 1: INTENT
            # =========================================

            if (
                intent
                and not product_matches_intent(
                    product,
                    intent,
                )
            ):
                continue

            # =========================================
            # HARD FILTER 2: DIAMETER
            # =========================================

            if (
                intent == "drilling"
                and diameter is not None
                and not product_matches_diameter(
                    product,
                    diameter,
                )
            ):
                continue

            # =========================================
            # HARD FILTER 3: THREAD
            # =========================================

            if intent == "tapping":

                product_hand = product_thread_hand(product)

                # اگر کاربر نوع رزوه را مشخص کرده،
                # فقط همان نوع را قبول کن.
                if thread_hand is not None:

                    if (
                            product_hand is not None
                            and product_hand != thread_hand
                    ):
                        continue

                # اگر کاربر نوع رزوه را مشخص نکرده،
                # قلاویز چپ‌گرد را در جستجوی اصلی کنار بگذار.
                # چون LH یک ویژگی تخصصی است و نباید به‌صورت پیش‌فرض
                # به کاربر پیشنهاد شود.
                else:

                    if product_hand == "left":
                        continue
            # =========================================
            # HARD FILTER 4: THREAD HAND
            # =========================================
            # =========================================
            # HARD FILTER 5: WORKPIECE MATERIAL
            # =========================================

            if material:

                if not product_matches_material(
                    product,
                    material,
                    material_aliases,
                ):
                    continue

            # =========================================
            # HARD FILTER 6: TOOL MATERIAL
            # =========================================

            if tool_material_aliases:

                if not product_has_tool_material(
                    product,
                    tool_material_aliases,
                ):
                    continue

            # =========================================
            # SCORE
            # =========================================

            semantic_score = (
                self._semantic_score(
                    query,
                    product,
                    float(
                        tfidf_scores[index]
                    )
                    if index < len(tfidf_scores)
                    else 0.0,
                )
            )

            keyword_score = (
                self._keyword_score(
                    query,
                    product,
                )
            )

            final_score = (
                semantic_score
                + min(
                    keyword_score * 0.04,
                    0.24,
                )
            )

            candidates.append(
                (
                    final_score,
                    index,
                    product,
                )
            )

        # =============================================
        # SORT
        # =============================================

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[2].get(
                    "name",
                    "",
                ),
            )
        )

        # =============================================
        # MINIMUM SCORE
        # =============================================

        if intent:
            minimum_score = 0.08
        else:
            minimum_score = 0.12

        filtered_candidates = [
            item
            for item in candidates
            if item[0] >= minimum_score
        ]

        # اگر هیچ‌کدام امتیاز کافی نگرفتند
        if not filtered_candidates:
            return []

        # =============================================
        # FINAL RESULTS
        # =============================================

        results: list[dict] = []

        for (
            score,
            _,
            product,
        ) in filtered_candidates[:top_k]:

            item = dict(product)

            item["_score"] = round(
                float(score),
                4,
            )

            results.append(
                item
            )

        return results

    # -----------------------------------------------------
    # BUILD CONTEXT
    # -----------------------------------------------------

    def build_context(
        self,
        products: list[dict],
        max_chars: int = 5000,
    ) -> str:

        chunks: list[str] = []

        for index, product in enumerate(
            products,
            start=1,
        ):

            fields = {

                "نام محصول":
                    product.get(
                        "name",
                        "",
                    ),

                "برند":
                    product.get(
                        "brand",
                        "",
                    ),

                "دسته بندی":
                    product.get(
                        "category",
                        "",
                    ),

                "توضیحات":
                    product.get(
                        "description",
                        "",
                    ),

                "ویژگی ها":
                    product.get(
                        "features",
                        "",
                    ),

                "مزایا":
                    product.get(
                        "advantages",
                        "",
                    ),

                "کاربردها":
                    product.get(
                        "applications",
                        "",
                    ),

                "مشخصات فنی":
                    product.get(
                        "technical_specs",
                        {},
                    ),

                "ویژگی های ساختاری":
                    product.get(
                        "attributes",
                        {},
                    ),
            }

            parts = [
                f"محصول {index}:"
            ]

            for key, value in fields.items():

                if value in (
                    None,
                    "",
                    [],
                    {},
                ):
                    continue

                rendered = " ".join(
                    flatten_value(value)
                ).strip()

                if rendered:

                    parts.append(
                        f"{key}: {rendered}"
                    )

            chunks.append(
                "\n".join(parts)
            )

        context = "\n\n".join(
            chunks
        )

        if len(context) <= max_chars:
            return context

        return context[:max_chars]

    # -----------------------------------------------------
    # NO MATCH MESSAGE
    # -----------------------------------------------------

    def no_match_message(
        self,
        query: str,
    ) -> str:

        parsed = self.parse_request(
            query
        )

        # ---------------------------------------------
        # DIAMETER
        # ---------------------------------------------

        if (
            parsed["diameter"]
            is not None
            and parsed["intent"]
            == "drilling"
        ):

            value = parsed["diameter"]

            return (
                "در محصولات ثبت‌شده، "
                "مته‌ای که قطر "
                f"{value:g} میلی‌متر "
                "و مشخصات درخواستی شما را "
                "پوشش دهد پیدا نکردم."
            )

        # ---------------------------------------------
        # THREAD
        # ---------------------------------------------

        if (
            parsed["thread"]
            is not None
            and parsed["intent"]
            == "tapping"
        ):

            nominal = parsed[
                "thread"
            ]["nominal"]

            pitch = parsed[
                "thread"
            ]["pitch"]

            if pitch is not None:

                return (
                    "در اطلاعات محصولات "
                    "ثبت‌شده، قلاویزی با "
                    f"رزوه M{nominal:g}"
                    f"×{pitch:g} "
                    "و مشخصات درخواستی پیدا نکردم."
                )

            return (
                "در اطلاعات محصولات ثبت‌شده، "
                f"قلاویزی با رزوه M{nominal:g} "
                "و مشخصات موردنظر به‌صورت "
                "صریح تأیید نشده است."
            )

        # ---------------------------------------------
        # TOOL MATERIAL
        # ---------------------------------------------

        if parsed["tool_material"]:

            tool_material = parsed[
                "tool_material"
            ]

            label = {
                "hss": "HSS",
                "cobalt": "کبالت",
                "carbide": "کارباید",
                "carbide_tipped": "نوک کارباید",
            }.get(
                tool_material,
                tool_material,
            )

            return (
                "در اطلاعات محصولات ثبت‌شده، "
                f"محصولی با جنس ابزار {label} "
                "و مشخصات موردنظر پیدا نکردم."
            )

        # ---------------------------------------------
        # MATERIAL
        # ---------------------------------------------

        if parsed["material"]:

            material = parsed["material"]

            return (
                "در اطلاعات محصولات ثبت‌شده، "
                "محصولی که برای "
                f"{material} "
                "و نوع ابزار درخواستی شما "
                "به‌صورت قابل تأیید مطابقت داشته "
                "باشد پیدا نکردم."
            )

        # ---------------------------------------------
        # DEFAULT
        # ---------------------------------------------

        return (
            "در محصولات ثبت‌شده، "
            "گزینه‌ای که مشخصات درخواست شما "
            "را به‌صورت قابل تأیید داشته باشد "
            "پیدا نکردم."
        )