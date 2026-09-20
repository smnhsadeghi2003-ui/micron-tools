# نصب ویجت چت روی وردپرس / هر سایتی

## روش ۱ – ساده (پیشنهادی)

1. API را روی یک دامنه یا ساب‌دامین آنلاین کنید (مثلاً `https://ai.microntoolss.ir`).
2. فایل `micron-chat-widget.js` را در پوشه `static` سرور قرار دهید (یا از همان مسیر پروژه کپی کنید).
3. در **فوتر** قالب وردپرس (یا Appearance → Theme File Editor → footer.php) این کد را اضافه کنید:

```html
<script>
  window.MICRON_CHAT_API = "https://ai.microntoolss.ir";  /* آدرس API خودتان */
</script>
<script src="https://ai.microntoolss.ir/static/micron-chat-widget.js" defer></script>
```

4. ذخیره کنید و صفحه را رفرش کنید. دکمه چت آبی‌رنگ در گوشه پایین سمت چپ ظاهر می‌شود.

---

## روش ۲ – با افزونه Insert Headers and Footers

1. افزونه **Insert Headers and Footers** (یا مشابه) را نصب کنید.
2. به بخش Scripts in Footer بروید و همان کد بالا را قرار دهید.
3. ذخیره کنید.

---

## روش ۳ – کد کوتاه PHP در functions.php

```php
add_action('wp_footer', function () {
  ?>
  <script>
    window.MICRON_CHAT_API = "https://ai.microntoolss.ir";
  </script>
  <script src="https://ai.microntoolss.ir/static/micron-chat-widget.js" defer></script>
  <?php
});
```

---

## نکات مهم

- حتماً CORS روی API باز باشد (`ALLOWED_ORIGINS=*` یا دامنه سایت).
- اگر سایت HTTPS است، API هم باید HTTPS باشد (محتوای mixed content بلاک می‌شود).
- برای تست محلی می‌توانید `MICRON_CHAT_API` را روی `http://127.0.0.1:8000` بگذارید.

---

## سفارشی‌سازی سریع

| متغیر | توضیح |
|-------|--------|
| `window.MICRON_CHAT_API` | آدرس پایه API (اجباری) |

اگر بخواهید رنگ یا موقعیت دکمه را عوض کنید، داخل فایل JS بخش `STYLE` را ویرایش کنید.

