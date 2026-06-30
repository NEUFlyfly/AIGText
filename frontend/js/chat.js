/**
 * AIGText Chat — 简化可靠版本
 * 功能：文本聊天（SSE 流式）、图片上传、对话历史
 */
(function () {
  "use strict";

  const state = {
    messages: [],
    streaming: false,
    ragEnabled: true,
    conversationId: null,
    pendingPhoto: null,
    abortController: null,
  };

  const dom = {
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("empty-state"),
    input: document.getElementById("message-input"),
    sendBtn: document.getElementById("send-button"),
    stopBtn: document.getElementById("stop-button"),
    newChatBtn: document.getElementById("new-chat-btn"),
    ragSwitch: document.getElementById("rag-switch"),
    sidebar: document.getElementById("sidebar"),
    sidebarOverlay: document.getElementById("sidebar-overlay"),
    sidebarToggle: document.getElementById("sidebar-toggle"),
    sidebarClose: document.getElementById("sidebar-close"),
    sidebarNewBtn: document.getElementById("sidebar-new-btn"),
    sidebarList: document.getElementById("sidebar-list"),
    sidebarEmpty: document.getElementById("sidebar-empty"),
    galleryInput: document.getElementById("gallery-input"),
    cameraCaptureInput: document.getElementById("camera-capture-input"),
    galleryButton: document.getElementById("gallery-button"),
    cameraButton: document.getElementById("camera-button"),
    cameraThumb: document.getElementById("camera-thumb"),
    cameraThumbImg: document.getElementById("camera-thumb-img"),
    cameraThumbRemove: document.getElementById("camera-thumb-remove"),
  };

  function formatTime() {
    const d = new Date();
    return String(d.getHours()).padStart(2, "0") + ":" +
           String(d.getMinutes()).padStart(2, "0");
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function formatMd(text) {
    if (!text) return "";
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/```[\s\S]*?```/g, match => `<pre><code>${match.slice(3, -3)}</code></pre>`)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]+)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br>");
    return "<p>" + html + "</p>";
  }

  function hideEmpty() {
    if (dom.emptyState) dom.emptyState.style.display = "none";
  }

  function scrollToBottom() {
    dom.messages.scrollTop = dom.messages.scrollHeight;
  }

  function setStreaming(v) {
    state.streaming = v;
    dom.sendBtn.classList.toggle("hidden", v);
    dom.stopBtn.classList.toggle("hidden", !v);
  }

  function renderMessage(role, content) {
    hideEmpty();
    const div = document.createElement("div");
    div.className = "message message--" + role;
    div.innerHTML =
      "<div class='message__bubble'>" +
      "<div class='message__content'>" + formatMd(content) + "</div>" +
      "<div class='message__time'>" + formatTime() + "</div>" +
      "</div>";
    dom.messages.appendChild(div);
    scrollToBottom();
    return div.querySelector(".message__content");
  }

  function handleSend() {
    if (state.streaming) return;
    const text = dom.input.value.trim();
    if (!text && !state.pendingPhoto) return;

    // 用户消息气泡
    renderMessage("user", text || "[图片]");
    state.messages.push({ role: "user", content: text, timestamp: Date.now() });
    dom.input.value = "";

    if (state.pendingPhoto) {
      // 图片 + 文字 → vision API
      sendVision(state.pendingPhoto.blob, text);
      state.pendingPhoto = null;
      dom.cameraThumb.classList.add("hidden");
    } else {
      // 纯文本聊天
      streamChat(text);
    }
    saveConversation();
  }

  async function streamChat(userText) {
    setStreaming(true);
    state.abortController = new AbortController();
    const aiEl = renderMessage("assistant", "");
    let fullReply = "";

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-RAG-Enabled": String(state.ragEnabled),
        },
        body: JSON.stringify({
          stream: true,
          messages: [
            { role: "system", content: "You are a helpful assistant." },
            { role: "user", content: userText },
          ],
        }),
        signal: state.abortController.signal,
      });

      if (!resp.ok) throw new Error("HTTP " + resp.status);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() || "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") continue;
          try {
            const obj = JSON.parse(data);
            const delta = obj.choices?.[0]?.delta?.content || "";
            if (delta) {
              fullReply += delta;
              aiEl.innerHTML = formatMd(fullReply);
              scrollToBottom();
            }
          } catch (e) {}
        }
      }

      if (!fullReply) aiEl.innerHTML = formatMd("(无回复)");
    } catch (err) {
      if (err.name !== "AbortError") {
        aiEl.innerHTML = formatMd("发送失败: " + err.message);
      }
    } finally {
      state.messages.push({ role: "assistant", content: fullReply, timestamp: Date.now() });
      setStreaming(false);
      state.abortController = null;
      saveConversation();
    }
  }

  async function sendVision(blob, question) {
    setStreaming(true);
    const aiEl = renderMessage("assistant", "📷 识别中...");

    const form = new FormData();
    form.append("image", blob, "photo.jpg");
    form.append("question", question || "");

    try {
      const resp = await fetch("/api/vision/query", {
        method: "POST",
        body: form,
      });
      const result = await resp.json();

      let answer = "";
      if (result.status === "ok") {
        const cand = result.visual_candidates?.[0];
        if (cand) answer += "**设备**: " + (cand.sub_category || cand.doc_id || "未知") + "\n\n";
        if (result.answer) answer += result.answer;
        else if (result.message) answer += result.message;
      } else {
        answer = result.message || result.error || "识别失败";
      }
      aiEl.innerHTML = formatMd(answer);
      state.messages.push({ role: "assistant", content: answer, timestamp: Date.now(), metadata: { type: "vision_result" } });
    } catch (err) {
      aiEl.innerHTML = formatMd("识别失败: " + err.message);
    } finally {
      setStreaming(false);
      saveConversation();
    }
  }

  function stopGeneration() {
    if (state.abortController) state.abortController.abort();
    setStreaming(false);
  }

  // 图片选择
  dom.galleryButton.addEventListener("click", () => dom.galleryInput.click());
  dom.cameraButton.addEventListener("click", () => dom.cameraCaptureInput.click());

  dom.galleryInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) showThumb(file);
    dom.galleryInput.value = "";
  });

  dom.cameraCaptureInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) showThumb(file);
    dom.cameraCaptureInput.value = "";
  });

  function showThumb(file) {
    const url = URL.createObjectURL(file);
    dom.cameraThumbImg.src = url;
    dom.cameraThumb.classList.remove("hidden");
    state.pendingPhoto = { blob: file, url: url };
  }

  dom.cameraThumbRemove.addEventListener("click", () => {
    dom.cameraThumb.classList.add("hidden");
    state.pendingPhoto = null;
  });

  // RAG 开关
  dom.ragSwitch.checked = state.ragEnabled;
  dom.ragSwitch.addEventListener("change", () => {
    state.ragEnabled = dom.ragSwitch.checked;
  });

  // 事件绑定
  dom.sendBtn.addEventListener("click", handleSend);
  dom.stopBtn.addEventListener("click", stopGeneration);
  dom.newChatBtn.addEventListener("click", () => {
    state.messages = [];
    dom.messages.innerHTML = "";
    dom.emptyState.style.display = "flex";
    state.conversationId = null;
  });
  dom.input.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Sidebar
  dom.sidebarToggle.addEventListener("click", () => {
    dom.sidebar.classList.remove("hidden");
    dom.sidebarOverlay.classList.remove("hidden");
    loadConversations();
  });
  dom.sidebarClose.addEventListener("click", closeSidebar);
  dom.sidebarOverlay.addEventListener("click", closeSidebar);
  dom.sidebarNewBtn.addEventListener("click", newChat);

  function closeSidebar() {
    dom.sidebar.classList.add("hidden");
    dom.sidebarOverlay.classList.add("hidden");
  }

  async function newChat() {
    state.conversationId = null;
    state.messages = [];
    dom.messages.innerHTML = "";
    dom.emptyState.style.display = "flex";
    closeSidebar();
    const resp = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "新对话" }),
    });
    const data = await resp.json();
    state.conversationId = data.id;
  }

  async function loadConversations() {
    try {
      const resp = await fetch("/api/conversations");
      const list = await resp.json();
      dom.sidebarList.innerHTML = "";
      dom.sidebarEmpty.style.display = list.length ? "none" : "block";
      list.forEach(conv => {
        const item = document.createElement("div");
        item.className = "sidebar-item";
        item.innerHTML =
          "<div class='sidebar-item-content'>" +
          "<div class='sidebar-item-title'>" + escapeHtml(conv.title || "新对话") + "</div>" +
          "</div>";
        item.addEventListener("click", () => loadConversation(conv.id));
        dom.sidebarList.appendChild(item);
      });
    } catch (e) {}
  }

  async function loadConversation(id) {
    try {
      const resp = await fetch("/api/conversations/" + id);
      const conv = await resp.json();
      state.conversationId = id;
      state.messages = conv.messages || [];
      dom.messages.innerHTML = "";
      conv.messages.forEach(m => renderMessage(m.role, m.content));
      closeSidebar();
    } catch (e) {}
  }

  async function saveConversation() {
    if (!state.messages.length || !state.conversationId) return;
    try {
      await fetch("/api/conversations/" + state.conversationId, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: state.messages }),
      });
    } catch (e) {}
  }

  // 初始化
  async function init() {
    // 自动创建/加载对话
    const resp = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "新对话" }),
    });
    const data = await resp.json();
    state.conversationId = data.id;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
