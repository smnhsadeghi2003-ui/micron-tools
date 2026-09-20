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


PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
LATIN_DIGITS = "0123456789"

DIGIT_TRANSLATION = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    LATIN_DIGITS + LATIN_DIGITS,
)


MATERIAL_ALIASES = {
    "فولاد": (
        "فولاد",
        "فولادی",
        "steel",
    ),

    "استیل": (
        "استیل",
        "stainless steel",
        "فولاد ضد زنگ",
        "ضد زنگ",
    ),

    "فولاد ضد زنگ": (
        "فولاد ضد زنگ",
        "استیل",
        "stainless steel",
        "ضد زنگ",
    ),

    "چدن": (
        "چدن",
        "cast iron",
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


INTENT_KEYWORDS = {

    "drilling": (
        "مته",
        "سوراخ",
        "سوراخکاری",
        "سوراخ کاری",
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
        "milling",
        "end mill",
    ),

    "turning": (
        "تراش",
        "تراشکاری",
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


FORBIDDEN_BY_INTENT = {

    "drilling": (
        "قلاویز",
        "رزوه",
    ),

    "tapping": (
        "مته",
        "سوراخ",
        "فرز",
    ),

    "milling": (
        "قلاویز",
        "مته مرغک",
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


def normalize_text(value: Any) -> str:

    if value is None:
        return ""

    text = str(value).translate(
        DIGIT_TRANSLATION
    )

    text = (
        text
        .replace("ي", "ی")
        .replace("ى", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
    )

    text = text.lower()

    text = re.sub(
        r"[–—−]",
        "-",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def flatten_value(value: Any) -> list[str]:

    result: list[str] = []

    if value is None:
        return result

    if isinstance(value, dict):

        for key, item in value.items():

            result.append(
                str(key)
            )

            result.extend(
                flatten_value(item)
            )

        return result

    if isinstance(
        value,
        (list, tuple, set),
    ):

        for item in value:

            result.extend(
                flatten_value(item)
            )

        return result

    result.append(
        str(value)
    )

    return result


def extract_diameter(
    query: str,
) -> float | None:

    text = normalize_text(query)

    patterns = [

        r"(?:قطر|سوراخ|سایز)\s*[:=]?\s*"
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:میلی\s*متر|mm)?",

        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:میلی\s*متر|mm)\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:

            try:

                return float(
                    match.group(1)
                    .replace(",", ".")
                )

            except ValueError:
                pass

    return None


def extract_thread(
    query: str,
) -> dict[str, float | None] | None:

    text = normalize_text(query)

    match = re.search(
        r"\bm\s*(\d{1,3})"
        r"(?:\s*[x×*]\s*"
        r"(\d+(?:[.,]\d+)?))?"
        r"\b",
        text,
    )

    if not match:
        return None

    nominal = float(
        match.group(1)
    )

    pitch = None

    if match.group(2):

        pitch = float(
            match.group(2)
            .replace(",", ".")
        )

    return {
        "nominal": nominal,
        "pitch": pitch,
    }


def detect_intent(
    query: str,
) -> str | None:

    text = normalize_text(query)

    scores: list[
        tuple[int, str]
    ] = []

    for intent, keywords in INTENT_KEYWORDS.items():

        score = sum(
            1
            for word in keywords
            if word in text
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


def detect_material(
    query: str,
) -> tuple[
    str | None,
    tuple[str, ...],
]:

    text = normalize_text(query)

    for canonical in sorted(
        MATERIAL_ALIASES,
        key=len,
        reverse=True,
    ):

        aliases = MATERIAL_ALIASES[
            canonical
        ]

        if any(
            alias in text
            for alias in aliases
        ):

            return (
                canonical,
                aliases,
            )

    return None, ()


def _extract_ranges(
    text: str,
) -> list[tuple[float, float]]:

    result: list[
        tuple[float, float]
    ] = []

    normalized = normalize_text(
        text
    )

    pattern = re.compile(
        r"(\d+(?:[.,]\d+)?)"
        r"\s*(?:تا|الی|-|~)\s*"
        r"(\d+(?:[.,]\d+)?)"
    )

    for match in pattern.finditer(
        normalized
    ):

        try:

            low = float(
                match.group(1)
                .replace(",", ".")
            )

            high = float(
                match.group(2)
                .replace(",", ".")
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

    if not isinstance(
        value,
        dict,
    ):
        return result

    for key in value.keys():

        key_text = normalize_text(
            key
        ).strip()

        if re.fullmatch(
            r"\d+(?:[.,]\d+)?",
            key_text,
        ):

            try:

                result.append(
                    float(
                        key_text
                        .replace(",", ".")
                    )
                )

            except ValueError:
                pass

    return result


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

    for field in (
        "attributes",
        "technical_specs",
    ):

        value = product.get(
            field,
            {},
        )

        if isinstance(
            value,
            dict,
        ):

            for numeric in _numeric_keys(
                value
            ):

                ranges.append(
                    (
                        numeric,
                        numeric,
                    )
                )

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

        normalized = normalize_text(
            text
        )

        for match in re.finditer(
            r"(?:سایز|قطر)\s*"
            r"(\d+(?:[.,]\d+)?)",
            normalized,
        ):

            try:

                value = float(
                    match.group(1)
                    .replace(",", ".")
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


def product_matches_thread(
    product: dict,
    requested_thread: dict[
        str,
        float | None,
    ],
) -> bool:

    nominal = requested_thread[
        "nominal"
    ]

    pitch = requested_thread[
        "pitch"
    ]

    evidence = _thread_evidence_text(
        product
    )

    nominal_text = str(
        int(nominal)
    )

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

    if pitch is None:
        return True

    for match in matches:

        product_pitch = match.group(1)

        if product_pitch is None:
            continue

        try:

            value = float(
                product_pitch
                .replace(",", ".")
            )

        except ValueError:
            continue

        if abs(
            value - pitch
        ) < 1e-9:

            return True

    return False


def product_matches_material(
    product: dict,
    aliases: tuple[str, ...],
) -> bool:

    evidence = _product_evidence_text(
        product
    )

    return any(
        normalize_text(alias)
        in evidence
        for alias in aliases
    )


def product_matches_intent(
    product: dict,
    intent: str,
) -> bool:

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

    forbidden = FORBIDDEN_BY_INTENT.get(
        intent,
        (),
    )

    if any(
        word in name
        for word in forbidden
    ):

        return False

    # Center/spot drills are not treated as
    # general final-hole drills.
    if (
        intent == "drilling"
        and any(
            term in name
            for term in (
                "مته مرغک",
                "مرغک",
                "مته مرکز",
                "center drill",
                "center-drill",
            )
        )
    ):

        return False

    keywords = INTENT_KEYWORDS.get(
        intent,
        (),
    )

    return any(
        word in name
        or word in category
        for word in keywords
    )


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
                "products_full.json must "
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

        texts = [
            normalize_text(
                product.get(
                    "search_text",
                    product.get(
                        "name",
                        "",
                    ),
                )
            )
            for product in self.products
        ]

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

    def parse_request(
        self,
        query: str,
    ) -> dict:

        material, material_aliases = (
            detect_material(query)
        )

        return {
            "intent": detect_intent(query),
            "diameter": extract_diameter(query),
            "thread": extract_thread(query),
            "material": material,
            "material_aliases": material_aliases,
        }

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

            elif token in search_text:
                score += 0.5

        return score

    def _semantic_score(
        self,
        query: str,
        product: dict,
        tfidf_score: float,
    ) -> float:

        parsed = self.parse_request(
            query
        )

        score = float(
            tfidf_score
        )

        intent = parsed[
            "intent"
        ]

        if intent:

            keywords = (
                INTENT_KEYWORDS.get(
                    intent,
                    (),
                )
            )

            name = normalize_text(
                product.get(
                    "name",
                    "",
                )
            )

            if any(
                keyword in name
                for keyword in keywords
            ):

                score += 0.18

        diameter = parsed[
            "diameter"
        ]

        if (
            intent == "drilling"
            and diameter is not None
            and product_matches_diameter(
                product,
                diameter,
            )
        ):

            score += 0.35

        thread = parsed[
            "thread"
        ]

        if (
            intent == "tapping"
            and thread is not None
            and product_matches_thread(
                product,
                thread,
            )
        ):

            score += 0.35

        aliases = parsed[
            "material_aliases"
        ]

        if (
            aliases
            and product_matches_material(
                product,
                aliases,
            )
        ):

            score += 0.18

        if (
            intent == "drilling"
            and parsed["material"]
            in {
                "فولاد",
                "استیل",
                "فولاد ضد زنگ",
            }
        ):

            evidence = (
                _product_evidence_text(
                    product
                )
            )

            if (
                "hssco" in evidence
                or "hssco5" in evidence
                or "کبالت" in evidence
            ):

                score += 0.08

        return score

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

        parsed = self.parse_request(
            query
        )

        intent = parsed[
            "intent"
        ]

        diameter = parsed[
            "diameter"
        ]

        thread = parsed[
            "thread"
        ]

        material_aliases = parsed[
            "material_aliases"
        ]

        tfidf_scores = (
            self._tfidf_scores(
                query
            )
        )

        candidates: list[
            tuple[float, int, dict]
        ] = []

        for index, product in enumerate(
            self.products
        ):

            if (
                intent
                and not product_matches_intent(
                    product,
                    intent,
                )
            ):

                continue

            if (
                intent == "drilling"
                and diameter is not None
                and not product_matches_diameter(
                    product,
                    diameter,
                )
            ):

                continue

            if (
                intent == "tapping"
                and thread is not None
                and not product_matches_thread(
                    product,
                    thread,
                )
            ):

                continue

            if (
                material_aliases
                and not product_matches_material(
                    product,
                    material_aliases,
                )
            ):

                continue

            semantic_score = (
                self._semantic_score(
                    query,
                    product,
                    float(
                        tfidf_scores[index]
                    )
                    if index
                    < len(tfidf_scores)
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
                    keyword_score * 0.03,
                    0.18,
                )
            )

            candidates.append(
                (
                    final_score,
                    index,
                    product,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[2].get(
                    "name",
                    "",
                ),
            )
        )

        results: list[
            dict
        ] = []

        for (
            score,
            _,
            product,
        ) in candidates[
            :top_k
        ]:

            item = dict(
                product
            )

            item["_score"] = round(
                float(score),
                4,
            )

            results.append(
                item
            )

        return results

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

        return context[
            :max_chars
        ]

    def no_match_message(
        self,
        query: str,
    ) -> str:

        parsed = self.parse_request(
            query
        )

        if (
            parsed["diameter"]
            is not None
            and parsed["intent"]
            == "drilling"
        ):

            value = parsed[
                "diameter"
            ]

            return (
                "در محصولات ثبت‌شده، "
                "مته‌ای که قطر "
                f"{value:g} میلی‌متر "
                "را پوشش دهد پیدا نکردم."
            )

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
                    f"×{pitch:g} پیدا نکردم."
                )

            return (
                "در اطلاعات محصولات ثبت‌شده، "
                f"قلاویزی با رزوه M{nominal:g} "
                "به‌صورت صریح تأیید نشده است."
            )

        if parsed["material"]:

            return (
                "در اطلاعات محصولات ثبت‌شده، "
                "محصولی که به‌صورت صریح با "
                "جنس موردنظر مطابقت داشته "
                "باشد پیدا نکردم."
            )

        return (
            "در محصولات ثبت ‌شده، گزینه مناسبی "
            "برای این درخواست پیدا نکردم."
        )
