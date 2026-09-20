from __future__ import annotations

import re

from backend.config import (
    CONTACT_ADDRESS,
    CONTACT_HOURS,
    CONTACT_PHONE,
    CONTACT_WHATSAPP,
    WEBSITE,
)



def _normalize(text: str) -> str:
    t = (
        str(text or "")
        .strip()
        .lower()
        .replace("ي", "ی")
        .replace("ك", "ک")
        .replace("\u200c", " ")
    )
    # Persian digits → Latin
    for i, d in enumerate("۰۱۲۳۴۵۶۷۸۹"):
        t = t.replace(d, str(i))
    return re.sub(r"\s+", " ", t)


def is_english(message: str) -> bool:
    text = message or ""
    latin = len(re.findall(r"[a-zA-Z]", text))
    persian = len(re.findall(r"[\u0600-\u06FF]", text))
    return latin > persian


def classify_non_product_request(message: str) -> str | None:
    """
    Fast deterministic replies for non-product intents.
    Returns None if the message should go to RAG.
    """
    text = _normalize(message)
    words = re.sub(r"[^\w\u0600-\u06FF]+", " ", text).strip().split()
    english = is_english(message)

    # ── Contact ────────────────────────────────────────────────
    contact_kw = (
        "شماره", "تلفن", "واتساپ", "whatsapp", "آدرس", "address",
        "تماس", "contact", "phone", "call", "location", "where are you",
        "ساعت کاری", "ساعات کار", "ایمیل", "پشتیبانی",
    )
    if any(k in text for k in contact_kw):
        if english:
            return (
                f"📞 Contact Micron Tools\n\n"
                f"Phone: {CONTACT_PHONE}\n"
                f"WhatsApp: {CONTACT_WHATSAPP}\n"
                f"Address: {CONTACT_ADDRESS}\n"
                f"Hours: {CONTACT_HOURS}\n"
                f"Website: {WEBSITE}"
            )
        return (
            f"📞 راه‌های تماس میکرون ابزار\n\n"
            f"تلفن: {CONTACT_PHONE}\n"
            f"واتساپ: {CONTACT_WHATSAPP}\n"
            f"آدرس: {CONTACT_ADDRESS}\n"
            f"ساعات کاری: {CONTACT_HOURS}\n"
            f"سایت: {WEBSITE}"
        )

    # ── About ──────────────────────────────────────────────────
    about_kw = (
        "درباره", "شرکت", "میکرون", "شما کیستید", "چیکار میکنید",
        "about", "who are you", "company", "what do you do",
    )
    product_signal = any(
        x in text
        for x in ("مته", "قلاویز", "کولت", "فرز", "drill", "tap", "سوراخ", "رزوه")
    )
    if any(k in text for k in about_kw) and not product_signal:
        if english:
            return (
                "Micron Tools is a specialized supplier of industrial cutting tools, "
                "measuring instruments and holding systems in Tehran.\n\n"
                "Ask me about drills, taps, collets, end mills or measuring tools."
            )
        return (
            "میکرون ابزار تأمین‌کننده تخصصی ابزارهای برشی، اندازه‌گیری و سیستم‌های نگه‌دارنده صنعتی در تهران است.\n\n"
            "می‌توانید درباره مته، قلاویز، کولت، فرز یا ابزار اندازه‌گیری بپرسید."
        )

    # ── Greeting ───────────────────────────────────────────────
    greet_kw = (
        "سلام", "درود", "وقت بخیر", "صبح بخیر", "عصر بخیر", "شب بخیر",
        "hello", "hi", "hey", "good morning", "good afternoon", "good evening","salam","slm",
    )
    if words and len(words) <= 6 and any(k in text for k in greet_kw):
        if english:
            return (
                "Hello! 👋\n"
                "I'm the AI consultant of Micron Tools.\n\n"
                "Tell me the operation, diameter, material or tool type you need\n"
                "(e.g. «drill 8 mm on steel» or «M10 tap»)."
            )
        return (
            "سلام! 👋\n"
            "من مشاور هوشمند میکرون ابزار هستم.\n\n"
            "نوع عملیات، قطر، جنس قطعه یا نام ابزار را بفرمایید\n"
            "(مثلاً: «مته ۸ میلی‌متر برای فولاد» یا «قلاویز M10»)."
        )

    # ── Goodbye ────────────────────────────────────────────────
    bye_kw = (
        "خداحافظ", "خدانگهدار", "بای", "فعلا", "شب خوش",
        "bye", "goodbye", "see you", "take care"
    )
    if words and len(words) <= 5 and any(k in text for k in bye_kw):
        if english:
            return "Goodbye! 👋 Feel free to come back anytime. Have a great day!"
        return "خداحافظ! 👋 هر وقت سوالی داشتید در خدمتم. موفق باشید."

    # ── Thanks ─────────────────────────────────────────────────
    thanks_kw = (
        "ممنون", "متشکرم", "مرسی", "تشکر", "خیلی ممنون",
        "thanks", "thank you", "tnx"
    )
    if words and len(words) <= 5 and any(k in text for k in thanks_kw):
        if english:
            return "You're welcome! 🙏 Ask if you need anything else."
        return "خواهش می‌کنم 🙏 اگر کمک دیگری لازم داشتید بفرمایید."

    return None


def build_advice(user_message: str, products: list[dict]) -> str:
    """Safe deterministic fallback when LLM is unavailable."""
    if not products:
        return "بر اساس اطلاعات موجود، محصول منطبق با درخواست شما پیدا نشد."

    product = products[0]
    evidence = ""

    for field in ("applications", "features", "advantages"):
        values = product.get(field) or []
        if values:
            evidence = str(values[0]).strip()
            break

    if not evidence:
        desc = (product.get("short_description") or product.get("description") or "").strip()
        if desc:
            evidence = desc

    if evidence:
        evidence = re.sub(r"\s+", " ", evidence)
        if len(evidence) > 240:
            evidence = evidence[:237].rsplit(" ", 1)[0] + "..."
        return f"بر اساس اطلاعات ثبت‌شده، این محصول {evidence}"

    return "بر اساس اطلاعات موجود، این محصول با درخواست شما مطابقت دارد."
