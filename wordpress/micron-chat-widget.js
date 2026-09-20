/**
 * Micron Tools AI Chat Widget – Premium
 * Floating chat button for WordPress / any website.
 *
 * Usage:
 *   <script>
 *     window.MICRON_CHAT_API = "https://YOUR-API-DOMAIN";
 *   </script>
 *   <script src="https://YOUR-API-DOMAIN/static/micron-chat-widget.js"></script>
 */
(function () {
  "use strict";

  if (window.__MICRON_CHAT_LOADED__) return;
  window.__MICRON_CHAT_LOADED__ = true;

  const API = (window.MICRON_CHAT_API || "").replace(/\/$/, "");
  if (!API) {
    console.warn("[Micron Chat] MICRON_CHAT_API is not set.");
    return;
  }

  const STYLE = `
    #mt-chat-root * { box-sizing: border-box; font-family: Vazirmatn, Tahoma, sans-serif; }
    #mt-chat-btn {
      position: fixed; bottom: 24px; left: 24px; z-index: 99999;
      width: 60px; height: 60px; border-radius: 50%;
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      color: #fff; border: none; cursor: pointer;
      box-shadow: 0 6px 24px rgba(59,130,246,.45);
      display: flex; align-items: center; justify-content: center;
      font-size: 26px; transition: transform .2s, box-shadow .2s;
    }
    #mt-chat-btn:hover { transform: scale(1.08); box-shadow: 0 8px 28px rgba(59,130,246,.55); }
    #mt-chat-btn.open { background: #1e293b; }
    #mt-chat-panel {
      position: fixed; bottom: 96px; left: 24px; z-index: 99998;
      width: 380px; max-width: calc(100vw - 32px);
      height: 520px; max-height: calc(100vh - 120px);
      background: #0f172a; border: 1px solid #1e293b;
      border-radius: 18px; overflow: hidden;
      display: none; flex-direction: column;
      box-shadow: 0 12px 40px rgba(0,0,0,.45);
      direction: rtl;
    }
    #mt-chat-panel.open { display: flex; animation: mtSlideUp .25s ease; }
    @keyframes mtSlideUp {
      from { opacity: 0; transform: translateY(16px); }
      to { opacity: 1; transform: none; }
    }
    #mt-chat-header {
      background: linear-gradient(135deg, #1e3a5f, #1e293b);
      padding: 14px 16px; display: flex; align-items: center; gap: 12px;
      border-bottom: 1px solid #334155;
    }
    #mt-chat-header .logo {
      width: 38px; height: 38px; border-radius: 10px;
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      display: grid; place-items: center; color: #fff; font-weight: 700; font-size: 14px;
    }
    #mt-chat-header h3 { margin: 0; font-size: 15px; color: #f1f5f9; font-weight: 600; }
    #mt-chat-header p { margin: 2px 0 0; font-size: 11px; color: #94a3b8; }
    #mt-chat-close {
      margin-right: auto; background: transparent; border: none;
      color: #94a3b8; font-size: 20px; cursor: pointer; padding: 4px 8px;
    }
    #mt-chat-close:hover { color: #fff; }
    #mt-chat-body {
      flex: 1; overflow-y: auto; padding: 14px; display: flex;
      flex-direction: column; gap: 10px; background: #0c1220;
    }
    .mt-msg {
      max-width: 88%; padding: 10px 13px; border-radius: 14px;
      font-size: 13.5px; line-height: 1.7; white-space: pre-wrap; word-break: break-word;
    }
    .mt-msg.bot {
      align-self: flex-start; background: #1e293b; color: #e2e8f0;
      border: 1px solid #334155; border-bottom-right-radius: 4px;
    }
    .mt-msg.user {
      align-self: flex-end; background: #1a3a2e; color: #e2e8f0;
      border: 1px solid #2a4a3a; border-bottom-left-radius: 4px;
    }
    .mt-product {
      background: #1e293b; border: 1px solid #334155; border-radius: 10px;
      padding: 10px 12px; font-size: 12.5px; color: #cbd5e1;
    }
    .mt-product a { color: #60a5fa; text-decoration: none; }
    .mt-product a:hover { text-decoration: underline; }
    .mt-product .meta { color: #94a3b8; font-size: 11.5px; margin-top: 3px; }
    #mt-chat-footer {
      padding: 10px 12px; border-top: 1px solid #1e293b;
      display: flex; gap: 8px; background: #0f172a;
    }
    #mt-chat-input {
      flex: 1; background: #1e293b; border: 1px solid #334155;
      color: #f1f5f9; padding: 11px 14px; border-radius: 12px;
      font-size: 13.5px; outline: none; font-family: inherit;
    }
    #mt-chat-input:focus { border-color: #3b82f6; }
    #mt-chat-send {
      background: linear-gradient(135deg, #3b82f6, #1d4ed8);
      border: none; color: #fff; padding: 0 16px; border-radius: 12px;
      font-weight: 600; cursor: pointer; font-size: 13.5px; font-family: inherit;
    }
    #mt-chat-send:disabled { opacity: .5; cursor: not-allowed; }
    .mt-typing { color: #64748b; font-size: 12px; padding: 2px 6px; }
    .mt-suggestions { display: flex; flex-wrap: wrap; gap: 6px; padding: 0 0 4px; }
    .mt-suggestions button {
      background: #1e293b; border: 1px solid #334155; color: #cbd5e1;
      padding: 5px 11px; border-radius: 999px; font-size: 11.5px;
      cursor: pointer; font-family: inherit;
    }
    .mt-suggestions button:hover { border-color: #3b82f6; }
  `;

  // Inject font + styles
  const font = document.createElement("link");
  font.rel = "stylesheet";
  font.href = "https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700&display=swap";
  document.head.appendChild(font);

  const style = document.createElement("style");
  style.textContent = STYLE;
  document.head.appendChild(style);

  // DOM
  const root = document.createElement("div");
  root.id = "mt-chat-root";
  root.innerHTML = `
    <button id="mt-chat-btn" title="مشاور هوشمند میکرون ابزار" aria-label="باز کردن چت">💬</button>
    <div id="mt-chat-panel">
      <div id="mt-chat-header">
        <div class="logo">MT</div>
        <div>
          <h3>میکرون ابزار</h3>
          <p>مشاور هوشمند ابزارآلات صنعتی</p>
        </div>
        <button id="mt-chat-close" aria-label="بستن">×</button>
      </div>
      <div id="mt-chat-body"></div>
      <div id="mt-chat-footer">
        <input id="mt-chat-input" type="text" placeholder="سؤال خود را بنویسید..." autocomplete="off" />
        <button id="mt-chat-send">ارسال</button>
      </div>
    </div>
  `;
  document.body.appendChild(root);

  const btn = document.getElementById("mt-chat-btn");
  const panel = document.getElementById("mt-chat-panel");
  const body = document.getElementById("mt-chat-body");
  const input = document.getElementById("mt-chat-input");
  const sendBtn = document.getElementById("mt-chat-send");
  const closeBtn = document.getElementById("mt-chat-close");

  let opened = false;

  function toggle() {
    opened = !opened;
    panel.classList.toggle("open", opened);
    btn.classList.toggle("open", opened);
    btn.textContent = opened ? "×" : "💬";
    if (opened) input.focus();
  }

  btn.addEventListener("click", toggle);
  closeBtn.addEventListener("click", toggle);

  function addMsg(text, role) {
    const div = document.createElement("div");
    div.className = "mt-msg " + role;
    div.textContent = text;
    body.appendChild(div);
    body.scrollTop = body.scrollHeight;
  }

  function addProducts(products) {
    if (!products || !products.length) return;
    products.forEach((p, i) => {
      const card = document.createElement("div");
      card.className = "mt-product";
      let html = `<strong>${i + 1}) ${p.name || "محصول"}</strong>`;
      if (p.brand) html += `<div class="meta">برند: ${p.brand}</div>`;
      if (p.price) html += `<div class="meta">قیمت: ${p.price}</div>`;
      if (p.url) html += `<div class="meta"><a href="${p.url}" target="_blank" rel="noopener">مشاهده محصول ←</a></div>`;
      card.innerHTML = html;
      body.appendChild(card);
    });
    body.scrollTop = body.scrollHeight;
  }

  async function send(text) {
    text = (text || input.value || "").trim();
    if (!text) return;
    input.value = "";
    sendBtn.disabled = true;
    addMsg(text, "user");

    const typing = document.createElement("div");
    typing.className = "mt-typing";
    typing.textContent = "در حال بررسی...";
    body.appendChild(typing);
    body.scrollTop = body.scrollHeight;

    try {
      const res = await fetch(API + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, top_k: 5 }),
      });
      const data = await res.json();
      typing.remove();
      addMsg(data.answer || "پاسخی دریافت نشد.", "bot");
      if (data.products && data.products.length) addProducts(data.products);
    } catch (err) {
      typing.remove();
      addMsg("خطا در ارتباط. لطفاً دوباره تلاش کنید.", "bot");
    }
    sendBtn.disabled = false;
    input.focus();
  }

  sendBtn.addEventListener("click", () => send());
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  // Welcome + suggestions
  addMsg(
    "سلام! 👋 من مشاور هوشمند میکرون ابزار هستم.\nنوع عملیات، قطر، جنس قطعه یا نام ابزار را بپرسید.",
    "bot"
  );

  const sug = document.createElement("div");
  sug.className = "mt-suggestions";
  [
    ["مته کبالت", "مته کبالت برای فولاد"],
    ["قلاویز M10", "قلاویز M10"],
    ["سوراخ ۳۰mm", "برای سوراخ ۳۰ میلی‌متر روی فولاد چه ابزاری؟"],
    ["تماس", "تماس"],
  ].forEach(([label, q]) => {
    const b = document.createElement("button");
    b.textContent = label;
    b.addEventListener("click", () => send(q));
    sug.appendChild(b);
  });
  body.appendChild(sug);
})();
