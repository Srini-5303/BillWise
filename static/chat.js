// chat.js — BillWise Assistant

// ── Session ID ────────────────────────────────────────────────────────────────
function getSessionId() {
  let id = localStorage.getItem("bill_chat_session");
  if (!id) {
    id = "web-" + crypto.randomUUID();
    localStorage.setItem("bill_chat_session", id);
  }
  return id;
}
const SESSION_ID = getSessionId();

// ── DOM refs ──────────────────────────────────────────────────────────────────
const messagesEl = document.getElementById("messages");
const inputEl    = document.getElementById("input");
const btnSend    = document.getElementById("btn-send");
const btnReset   = document.getElementById("btn-reset");
const btnReload  = document.getElementById("btn-reload");

// ── Auto-resize textarea ──────────────────────────────────────────────────────
function autoResize() {
  inputEl.style.height = "auto";
  inputEl.style.height = inputEl.scrollHeight + "px";
}
inputEl.addEventListener("input", autoResize);

// ── Markdown-lite renderer ────────────────────────────────────────────────────
function renderMarkdown(text) {
  if (!text) return "";

  // Markdown table → HTML table
  text = text.replace(/(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)+)/g, (table) => {
    const rows = table.trim().split("\n");
    const headers = rows[0].split("|").filter(c => c.trim()).map(c =>
      `<th>${c.trim()}</th>`).join("");
    const body = rows.slice(2).map(row => {
      const cells = row.split("|").filter(c => c.trim() !== undefined && c !== "");
      return "<tr>" + cells.map(c => `<td>${c.trim()}</td>`).join("") + "</tr>";
    }).join("");
    return `<table><thead><tr>${headers}</tr></thead><tbody>${body}</tbody></table>`;
  });

  // **bold**
  text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // `inline code`
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  // *italic*
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  // Line breaks
  text = text.replace(/\n/g, "<br>");

  return text;
}

// ── Append message ────────────────────────────────────────────────────────────
function appendMessage(role, html, sqlText) {
  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  // Avatar
  const avatar = document.createElement("div");
  avatar.className = "avatar";

  if (role === "assistant" || role === "assistant error") {
    avatar.className += " bw-avatar";
    avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/>
      <rect x="9" y="3" width="6" height="4" rx="1"/>
      <line x1="9" y1="12" x2="15" y2="12"/>
      <line x1="9" y1="16" x2="13" y2="16"/>
    </svg>`;
  } else if (role === "user") {
    avatar.className += " user-avatar";
    avatar.textContent = "You";
  }

  wrap.appendChild(avatar);

  // Body
  const body = document.createElement("div");
  body.className = "message-body";

  const content = document.createElement("div");
  content.className = "message-content";
  content.innerHTML = html;
  body.appendChild(content);

  // SQL toggle
  if ((role === "assistant" || role === "assistant error") && sqlText) {
    const toggle = document.createElement("button");
    toggle.className = "sql-toggle";
    toggle.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>
    </svg> Show SQL`;

    const pre = document.createElement("pre");
    pre.className = "sql-block hidden";
    pre.textContent = sqlText;

    toggle.addEventListener("click", () => {
      const isHidden = pre.classList.toggle("hidden");
      toggle.innerHTML = isHidden
        ? `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg> Show SQL`
        : `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg> Hide SQL`;
    });

    body.appendChild(toggle);
    body.appendChild(pre);
  }

  wrap.appendChild(body);
  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function showTyping() {
  const wrap = document.createElement("div");
  wrap.className = "message assistant";

  const avatar = document.createElement("div");
  avatar.className = "avatar bw-avatar";
  avatar.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
    <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2"/>
    <rect x="9" y="3" width="6" height="4" rx="1"/>
    <line x1="9" y1="12" x2="15" y2="12"/>
    <line x1="9" y1="16" x2="13" y2="16"/>
  </svg>`;
  wrap.appendChild(avatar);

  const body = document.createElement("div");
  body.className = "message-body";
  body.innerHTML = `<div class="typing-dots"><span></span><span></span><span></span></div>`;
  wrap.appendChild(body);

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  inputEl.style.height = "auto";
  inputEl.disabled = true;
  btnSend.disabled = true;

  appendMessage("user", escapeHtml(text), null);
  const typingEl = showTyping();

  try {
    const res = await fetch("/api/query", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: SESSION_ID, question: text }),
    });

    const data = await res.json();
    typingEl.remove();

    if (data.success) {
      appendMessage("assistant", renderMarkdown(data.answer), data.sql || null);
    } else {
      appendMessage("assistant error", `⚠️ ${escapeHtml(data.error || "Something went wrong.")}`, null);
    }
  } catch (err) {
    typingEl.remove();
    appendMessage("assistant error", "⚠️ Network error — please try again.", null);
  } finally {
    inputEl.disabled = false;
    btnSend.disabled = false;
    inputEl.focus();
  }
}

// ── Reset / Reload ────────────────────────────────────────────────────────────
async function resetChat() {
  await fetch("/api/reset", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ session_id: SESSION_ID }),
  });
  messagesEl.innerHTML = "";
  appendMessage("assistant", "Chat reset. What would you like to know?", null);
}

async function reloadData() {
  btnReload.disabled = true;
  const res  = await fetch("/api/reload", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ session_id: SESSION_ID }),
  });
  const data = await res.json();
  btnReload.disabled = false;
  appendMessage("assistant", data.message || "Data reloaded.", null);
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Events ────────────────────────────────────────────────────────────────────
btnSend.addEventListener("click", sendMessage);
btnReset.addEventListener("click", resetChat);
btnReload.addEventListener("click", reloadData);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
