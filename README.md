
مشاور هوشمند حرفه‌ای ابزارآلات صنعتی + ویجت وردپرس



- RAG deterministic قوی (قطر، رزوه M، جنس، Intent)
- فقط محصولات واقعاً مرتبط – بدون پیشنهاد الکی
- درخواست خاص → حداکثر ۱ محصول | عمومی → حداکثر ۳
- Intentهای فوری: سلام، خداحافظی، تشکر، تماس، درباره ما
- پشتیبانی کامل فارسی + انگلیسی
- LLM فقط توضیح می‌دهد، invent نمی‌کند
- پشتیبانی Ollama + OpenAI / Groq / xAI
- **ویجت شناور چت برای وردپرس و هر وب‌سایتی**
- رابط کاربری تمیز و حرفه‌ای

## نصب سریع سرور

```bash
# Windows: دوبار کلیک روی run_windows.bat

# یا دستی:
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

باز کنید: http://127.0.0.1:8000

## نصب ویجت روی وردپرس

جزئیات کامل در پوشه `wordpress/INSTALL.md`

خلاصه – این کد را در فوتر سایت بگذارید:

```html
<script>
  window.MICRON_CHAT_API = "https://YOUR-API-DOMAIN";
</script>
<script src="https://YOUR-API-DOMAIN/static/micron-chat-widget.js" defer></script>
```

## ساختار پروژه

```
micron_tools_premium/
├── main.py
├── backend/
│   ├── config.py
│   ├── consultant.py
│   ├── llm.py
│   └── rag.py
├── data/products_full.json
├── static/
│   ├── index.html
│   └── micron-chat-widget.js
├── wordpress/
│   ├── micron-chat-widget.js
│   └── INSTALL.md
├── requirements.txt
├── .env.example
└── run_windows.bat
```

## تست‌های کلیدی

| سؤال | نتیجه مورد انتظار |
|------|-------------------|
| hello / سلام | خوش‌آمدگویی فوری |
| bye / خداحافظ | خداحافظی |
| thanks / ممنون | تشکر |
| تماس | اطلاعات تماس کامل |
| سوراخ ۳۰ میلی‌متر روی فولاد | صادقانه: محصول مناسب نیست |
| مته کبالت | محصولات مرتبط واقعی |
| قلاویز M10 | فقط قلاویزهای منطبق |

## نکات تولید

- برای بهترین سرعت از Ollama محلی استفاده کنید
- کلید API را هرگز در گیت‌هاب نگذارید
- روی سرور حتماً HTTPS فعال باشد

