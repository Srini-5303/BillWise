// chat.js — Bill Assistant web UI logic

// ── Session ID (persisted in localStorage) ────────────────────────────────────
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
const messagesEl  = document.getElementById("messages");
const inputEl     = document.getElementById("input");
const btnSend     = document.getElementById("btn-send");
const btnReset    = document.getElementById("btn-reset");
const btnReload   = document.getElementById("btn-reload");

// ── Markdown-lite renderer ────────────────────────────────────────────────────
// Handles **bold**, `code`, markdown tables, and line breaks. No external lib.
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

// ── Bubble helpers ────────────────────────────────────────────────────────────
function appendBubble(role, html, sqlText) {
  const wrap = document.createElement("div");
  wrap.className = `bubble ${role}`;

  const content = document.createElement("div");
  content.className = "bubble-content";
  content.innerHTML = html;
  wrap.appendChild(content);

  // "Show SQL" toggle for assistant messages that have SQL
  if (role === "assistant" && sqlText) {
    const toggle = document.createElement("button");
    toggle.className = "sql-toggle";
    toggle.textContent = "Show SQL";

    const pre = document.createElement("pre");
    pre.className = "sql-block hidden";
    pre.textContent = sqlText;

    toggle.addEventListener("click", () => {
      const isHidden = pre.classList.toggle("hidden");
      toggle.textContent = isHidden ? "Show SQL" : "Hide SQL";
    });

    wrap.appendChild(toggle);
    wrap.appendChild(pre);
  }

  messagesEl.appendChild(wrap);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return wrap;
}

function showTyping() {
  return appendBubble("assistant typing-wrap", "<span class='typing'><span></span><span></span><span></span></span>", null);
}

// ── Send message ──────────────────────────────────────────────────────────────
async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text) return;

  inputEl.value = "";
  inputEl.disabled = true;
  btnSend.disabled = true;

  appendBubble("user", escapeHtml(text), null);
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
      appendBubble("assistant", renderMarkdown(data.answer), data.sql || null);
    } else {
      appendBubble("assistant error", `⚠️ ${escapeHtml(data.error || "Something went wrong.")}`, null);
    }
  } catch (err) {
    typingEl.remove();
    appendBubble("assistant error", "⚠️ Network error — please try again.", null);
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
  appendBubble("assistant", "Chat reset. What would you like to know?", null);
}

async function reloadData() {
  btnReload.disabled = true;
  btnReload.textContent = "Reloading…";
  const res  = await fetch("/api/reload", {
    method:  "POST",
    headers: { "Content-Type": "application/json" },
    body:    JSON.stringify({ session_id: SESSION_ID }),
  });
  const data = await res.json();
  btnReload.disabled = false;
  btnReload.textContent = "↺ Reload data";
  appendBubble("assistant", data.message || "Data reloaded.", null);
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

// ── Event listeners ───────────────────────────────────────────────────────────
btnSend.addEventListener("click", sendMessage);
btnReset.addEventListener("click", resetChat);
btnReload.addEventListener("click", reloadData);

inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});
