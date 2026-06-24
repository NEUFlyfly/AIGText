/**
 * AIGText — Chat Frontend Logic
 *
 * Pure vanilla JS — SSE streaming, RAG toggle, localStorage persistence,
 * markdown formatting, AbortController for stop generation.
 */

/* ==================================================================
   GLOBAL STATE
   ================================================================== */
const state = {
  messages: [
    {
      role: "system",
      content:
        "You are a helpful assistant. Answer concisely in the same language as the user.",
    },
  ],
  streaming: false,
  ragEnabled: true,
  activeStreamBubble: null,
  healthTimer: null,
  abortController: null,
};

/* ==================================================================
   DOM REFERENCES
   ================================================================== */
const $ = (s) => document.querySelector(s);

const dom = {
  messages: $("#messages"),
  emptyState: $("#empty-state"),
  input: $("#message-input"),
  sendBtn: $("#send-button"),
  stopBtn: $("#stop-button"),
  newChatBtn: $("#new-chat-btn"),
  healthDot: $("#health-dot"),
  ragSwitch: $("#rag-switch"),
};

/* ==================================================================
   UTILITIES
   ================================================================== */
function formatTime() {
  const d = new Date();
  return (
    String(d.getHours()).padStart(2, "0") +
    ":" +
    String(d.getMinutes()).padStart(2, "0")
  );
}

/* ==================================================================
   LOCAL STORAGE PERSISTENCE
   ================================================================== */
const STORAGE_KEY = "aigtext_messages";
const RAG_STORAGE_KEY = "aigtext_rag";

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state.messages));
    localStorage.setItem(RAG_STORAGE_KEY, JSON.stringify(state.ragEnabled));
  } catch (_e) {}
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) state.messages = JSON.parse(raw);
    const r = localStorage.getItem(RAG_STORAGE_KEY);
    if (r !== null) {
      state.ragEnabled = JSON.parse(r);
      if (dom.ragSwitch) dom.ragSwitch.checked = state.ragEnabled;
    }
  } catch (_e) {
    state.messages = [state.messages[0]];
  }
}

function clearStoredMessages() {
  try { localStorage.removeItem(STORAGE_KEY); } catch (_e) {}
}

/* ==================================================================
   MARKDOWN FORMATTING
   ================================================================== */
function formatMarkdown(text) {
  if (!text) return "";

  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Code blocks ``` ... ```
  html = html.replace(
    /```(\w*)\n?([\s\S]*?)```/g,
    (_, lang, code) =>
      `<pre><code>${code.replace(/\n$/, "")}</code></pre>`
  );

  // Inline code `...`
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold **...**
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");

  // Italic *...*
  html = html.replace(/\*([^*]+)\*/g, "<em>$1</em>");

  // Line breaks
  html = html.replace(/\n\n/g, "</p><p>");
  html = html.replace(/\n/g, "<br>");

  return "<p>" + html + "</p>";
}

/* ==================================================================
   MESSAGE RENDERING
   ================================================================== */
function addMessage(role, content, isStreaming) {
  hideEmptyState();

  const el = document.createElement("div");
  el.className = "message message--" + role;

  const bubble = document.createElement("div");
  bubble.className = "message__bubble";

  const contentEl = document.createElement("div");
  contentEl.className = "message__content";
  contentEl.innerHTML = isStreaming ? content : formatMarkdown(content);

  const timeEl = document.createElement("div");
  timeEl.className = "message__time";
  timeEl.textContent = formatTime();

  bubble.appendChild(contentEl);
  bubble.appendChild(timeEl);
  el.appendChild(bubble);
  dom.messages.appendChild(el);

  scrollToBottom();
  return contentEl;
}

function hideEmptyState() {
  if (dom.emptyState && !dom.emptyState.classList.contains("hidden")) {
    dom.emptyState.classList.add("hidden");
  }
}

function renderStoredMessages() {
  dom.messages.innerHTML = "";

  const emptyDiv = document.createElement("div");
  emptyDiv.id = "empty-state";
  emptyDiv.className = "empty-state";
  emptyDiv.innerHTML =
    '<div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"></path></svg></div>' +
    '<h2 class="empty-title">AIGText</h2>' +
    '<p class="empty-subtitle">本地 LLM 聊天助手</p>';
  dom.messages.appendChild(emptyDiv);
  dom.emptyState = emptyDiv;

  for (const msg of state.messages) {
    if (msg.role === "system") continue;
    addMessage(msg.role, msg.content, false);
  }
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    dom.messages.scrollTop = dom.messages.scrollHeight;
  });
}

/* ==================================================================
   STREAMING CONTROLS
   ================================================================== */
function setStreaming(active) {
  state.streaming = active;
  dom.input.disabled = active;

  if (active) {
    dom.sendBtn.classList.add("hidden");
    dom.stopBtn.classList.remove("hidden");
  } else {
    dom.sendBtn.classList.remove("hidden");
    dom.stopBtn.classList.add("hidden");
  }
}

function stopGeneration() {
  if (state.abortController) {
    state.abortController.abort();
    state.abortController = null;
  }
  if (state.activeStreamBubble) {
    const bubble = state.activeStreamBubble;
    const text = bubble.textContent || "";
    if (text) {
      state.messages.push({ role: "assistant", content: text });
      saveState();
      bubble.innerHTML = formatMarkdown(text);
    }
    state.activeStreamBubble = null;
  }
  setStreaming(false);
}

function newChat() {
  stopGeneration();
  state.messages = [state.messages[0]]; // keep system message
  clearStoredMessages();
  saveState();

  dom.messages.innerHTML = "";
  const emptyDiv = document.createElement("div");
  emptyDiv.id = "empty-state";
  emptyDiv.className = "empty-state";
  emptyDiv.innerHTML =
    '<div class="empty-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"></path></svg></div>' +
    '<h2 class="empty-title">AIGText</h2>' +
    '<p class="empty-subtitle">本地 LLM 聊天助手</p>';
  dom.messages.appendChild(emptyDiv);
  dom.emptyState = emptyDiv;
}

/* ==================================================================
   SSE STREAMING
   ================================================================== */
async function streamChat(messages) {
  const controller = new AbortController();
  state.abortController = controller;
  let fullContent = "";
  let contentEl = null;

  // Safety timeout: 5 minutes
  const safetyTimer = setTimeout(() => {
    if (state.streaming) controller.abort();
  }, 300000);

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-RAG-Enabled": state.ragEnabled ? "true" : "false",
      },
      body: JSON.stringify({
        messages,
        temperature: 0.7,
        max_tokens: 1024,
        stream: true,
      }),
      signal: controller.signal,
    });

    if (!resp.ok) {
      const errText = await resp.text().catch(() => "Unknown error");
      throw new Error(`HTTP ${resp.status}: ${errText}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || !trimmed.startsWith("data:")) continue;

        const data = trimmed.slice(5).trim();
        if (data === "[DONE]") continue;

        try {
          const parsed = JSON.parse(data);
          const delta = parsed.choices?.[0]?.delta?.content;
          if (!delta) continue;

          if (!contentEl) {
            contentEl = addMessage("assistant", "", true);
            state.activeStreamBubble = contentEl;
          }

          fullContent += delta;
          contentEl.innerHTML = formatMarkdown(fullContent);
          scrollToBottom();
        } catch (_e) {
          // Skip malformed SSE chunks
        }
      }
    }
  } catch (err) {
    if (err.name === "AbortError") {
      // User stopped generation — partial content already handled by stopGeneration
      return;
    }
    // Network / server error
    addMessage("assistant", err.message || "请求失败，请检查后端服务", false);
  } finally {
    clearTimeout(safetyTimer);
    state.abortController = null;

    if (fullContent && contentEl) {
      state.messages.push({ role: "assistant", content: fullContent });
      saveState();
      contentEl.innerHTML = formatMarkdown(fullContent);
    }
    state.activeStreamBubble = null;
    setStreaming(false);
  }
}

/* ==================================================================
   SEND HANDLER
   ================================================================== */
async function handleSend() {
  const text = dom.input.value.trim();
  if (!text || state.streaming) return;

  dom.input.value = "";
  dom.input.style.height = "auto";

  // RAG mode: show status
  showRagStatus(text);

  state.messages.push({ role: "user", content: text });
  saveState();
  addMessage("user", text, false);

  setStreaming(true);

  // Build API messages (send ALL history — RAG is injected server-side)
  await streamChat(state.messages);
}

async function showRagStatus(query) {
  // RAG status is server-side now, just a visual hint
}

/* ==================================================================
   HEALTH CHECK
   ================================================================== */
async function checkHealth() {
  const dot = dom.healthDot;
  if (!dot) return;

  dot.className = "health-dot checking";
  dot.title = "检查中...";

  try {
    const resp = await fetch("/api/health", {
      method: "GET",
      signal: AbortSignal.timeout ? AbortSignal.timeout(5000) : undefined,
    });
    if (resp.ok) {
      dot.className = "health-dot connected";
      dot.title = "已连接";
    } else {
      dot.className = "health-dot error";
      dot.title = "服务异常";
    }
  } catch (_e) {
    dot.className = "health-dot error";
    dot.title = "无法连接";
  }
}

/* ==================================================================
   INITIALIZATION
   ================================================================== */
function init() {
  loadState();
  if (dom.ragSwitch) dom.ragSwitch.checked = state.ragEnabled;
  renderStoredMessages();

  checkHealth();
  state.healthTimer = setInterval(checkHealth, 30000);

  // Send button
  dom.sendBtn.addEventListener("click", handleSend);

  // Stop button
  dom.stopBtn.addEventListener("click", stopGeneration);

  // New chat button
  dom.newChatBtn.addEventListener("click", newChat);

  // RAG toggle
  dom.ragSwitch.addEventListener("change", () => {
    state.ragEnabled = dom.ragSwitch.checked;
    saveState();
  });

  // Auto-resize textarea
  dom.input.addEventListener("input", () => {
    dom.input.style.height = "auto";
    dom.input.style.height = Math.min(dom.input.scrollHeight, 120) + "px";
  });

  // Keyboard: Enter to send, Shift+Enter for newline
  dom.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
}

document.addEventListener("DOMContentLoaded", init);
