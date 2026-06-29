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
// $ and $$ are provided by common.js (loaded first)

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

  // Camera initialization
  initCamera();
}

document.addEventListener("DOMContentLoaded", init);

/* ==================================================================
   CAMERA + VISION API
   ================================================================== */
const cameraState = {
  stream: null,
  facingMode: "environment",
  capturedBlob: null,
};

const cameraDom = {};

function initCameraDOM() {
  cameraDom.button = document.getElementById("camera-button");
  cameraDom.modal = document.getElementById("camera-modal");
  cameraDom.video = document.getElementById("camera-video");
  cameraDom.capture = document.getElementById("camera-capture");
  cameraDom.close = document.getElementById("camera-close");
  cameraDom.flip = document.getElementById("camera-flip");
  cameraDom.preview = document.getElementById("camera-preview");
  cameraDom.previewImg = document.getElementById("camera-preview-img");
  cameraDom.retake = document.getElementById("camera-retake");
  cameraDom.confirm = document.getElementById("camera-confirm");
  cameraDom.thumb = document.getElementById("camera-thumb");
  cameraDom.thumbImg = document.getElementById("camera-thumb-img");
  cameraDom.thumbRemove = document.getElementById("camera-thumb-remove");
}

function initCamera() {
  initCameraDOM();

  if (!cameraDom.button || !cameraDom.modal) return;

  cameraDom.button.addEventListener("click", openCamera);
  cameraDom.close.addEventListener("click", closeCamera);
  cameraDom.capture.addEventListener("click", capturePhoto);
  cameraDom.flip.addEventListener("click", flipCamera);
  cameraDom.retake.addEventListener("click", retakePhoto);
  cameraDom.confirm.addEventListener("click", confirmPhoto);
  cameraDom.thumbRemove.addEventListener("click", removeThumbnail);
}

async function openCamera() {
  if (state.streaming) return;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    Toast.show("当前设备不支持相机");
    return;
  }

  cameraDom.modal.classList.remove("hidden");
  cameraDom.modal.setAttribute("aria-hidden", "false");
  cameraDom.preview.classList.add("hidden");
  cameraDom.video.classList.remove("hidden");
  cameraDom.capture.parentElement.classList.remove("hidden");

  try {
    cameraState.stream = await Camera.open(
      cameraDom.video,
      cameraState.facingMode
    );
  } catch (err) {
    closeCamera();
    if (err.name === "NotAllowedError") {
      Toast.show("请允许相机权限后重试");
    } else if (err.name === "NotFoundError") {
      Toast.show("未检测到摄像头");
    } else {
      Toast.show("相机启动失败: " + (err.message || "未知错误"));
    }
  }
}

function closeCamera() {
  if (cameraState.stream) {
    Camera.stop(cameraState.stream);
    cameraState.stream = null;
  }
  // Revoke preview blob URL to prevent memory leaks
  if (cameraDom.previewImg && cameraDom.previewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(cameraDom.previewImg.src);
    cameraDom.previewImg.src = "";
  }
  cameraDom.video.srcObject = null;
  cameraDom.modal.classList.add("hidden");
  cameraDom.modal.setAttribute("aria-hidden", "true");
}

async function flipCamera() {
  cameraState.facingMode =
    cameraState.facingMode === "environment" ? "user" : "environment";

  if (cameraState.stream) {
    Camera.stop(cameraState.stream);
    cameraState.stream = null;
  }

  try {
    cameraState.stream = await Camera.open(
      cameraDom.video,
      cameraState.facingMode
    );
  } catch (err) {
    Toast.show("切换摄像头失败");
    cameraState.facingMode =
      cameraState.facingMode === "environment" ? "user" : "environment";
  }
}

async function capturePhoto() {
  if (!cameraState.stream) return;

  try {
    cameraState.capturedBlob = await Camera.capture(cameraDom.video, 0.9);

    const url = URL.createObjectURL(cameraState.capturedBlob);
    cameraDom.previewImg.src = url;

    cameraDom.video.classList.add("hidden");
    cameraDom.capture.parentElement.classList.add("hidden");
    cameraDom.preview.classList.remove("hidden");
  } catch (err) {
    Toast.show("拍照失败，请重试");
  }
}

function retakePhoto() {
  if (cameraDom.previewImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(cameraDom.previewImg.src);
  }
  cameraState.capturedBlob = null;
  cameraDom.previewImg.src = "";
  cameraDom.preview.classList.add("hidden");
  cameraDom.video.classList.remove("hidden");
  cameraDom.capture.parentElement.classList.remove("hidden");
}

async function confirmPhoto() {
  if (!cameraState.capturedBlob) return;

  const blob = cameraState.capturedBlob;

  // Show thumbnail in input area
  const url = URL.createObjectURL(blob);
  cameraDom.thumbImg.src = url;
  cameraDom.thumb.classList.remove("hidden");
  cameraState.capturedBlob = blob;

  closeCamera();

  // Send for classification
  await sendPhotoForClassification(blob);
}

function removeThumbnail() {
  if (cameraDom.thumbImg.src.startsWith("blob:")) {
    URL.revokeObjectURL(cameraDom.thumbImg.src);
  }
  cameraDom.thumbImg.src = "";
  cameraDom.thumb.classList.add("hidden");
  cameraState.capturedBlob = null;
}

/**
 * Map Visual RAG error codes to user-facing Chinese display text.
 */
function visualErrorDisplayText(code) {
  var messages = {
    "INVALID_IMAGE": "图片无效，请重新拍摄清晰的设备照片",
    "MODEL_NOT_READY": "识别模型未就绪，请稍后重试",
    "INDEX_NOT_READY": "知识库索引未就绪，请稍后重试",
    "NO_VISUAL_MATCH": "未识别到匹配的设备，请尝试不同角度或光线",
  };
  return messages[code] || "识别服务异常，请重试";
}

/**
 * Display Visual RAG API error as a system message + toast.
 */
function displayVisualError(result, httpStatus) {
  var errors = result.errors || [];
  if (errors.length > 0) {
    var code = errors[0].code;
    var text = visualErrorDisplayText(code);
    addMessage("system", "📷 " + text, false);
    Toast.show(text, 4000);
  } else {
    var msg = "识别服务异常 (HTTP " + httpStatus + ")";
    Toast.show(msg, 3000);
  }
}

/**
 * Display full Visual RAG response: candidate summary, answer, status messages.
 * Does NOT trigger a duplicate /api/chat request when answer is already present.
 */
function displayVisualRagResponse(result) {
  var status = result.status || "";
  var answer = result.answer;
  var candidates = result.visual_candidates || [];
  var errors = result.errors || [];
  var coarseCategory = result.coarse_category;
  var coarseConfidence = result.coarse_confidence;

  hideEmptyState();

  // Non-blocking error display
  if (errors.length > 0) {
    var errorTexts = errors.map(function(e) {
      return visualErrorDisplayText(e.code) || e.message || "未知错误";
    }).join("; ");
    addMessage("system", "⚠ " + errorTexts, false);
  }

  // Candidate summary with labels and scores
  if (candidates.length > 0) {
    var candidateLines = candidates.map(function(c) {
      var label = c.sub_category || c.coarse_category || "未知设备";
      var score = c.score != null ? Math.round(c.score * 100) : 0;
      return "<strong>" + label + "</strong> (" + score + "%)";
    });
    var categoryPrefix = "";
    if (coarseCategory) {
      var confPercent = coarseConfidence != null ? Math.round(coarseConfidence * 100) : 0;
      categoryPrefix = "大类: <strong>" + coarseCategory + "</strong> (" + confPercent + "%) · ";
    }
    addVisionResultBubble(categoryPrefix, candidateLines);
  } else if (status === "NO_VISUAL_MATCH") {
    addMessage("system", "📷 " + visualErrorDisplayText("NO_VISUAL_MATCH"), false);
  }

  // Display answer directly as assistant message (NO duplicate /api/chat)
  if (answer) {
    addMessage("assistant", answer, false);
    state.messages.push({ role: "assistant", content: answer });
    saveState();
  } else if (status === "OK" && candidates.length > 0) {
    addMessage("system", "📷 已识别设备但未能生成回答，请尝试文本提问", false);
  }
}

/**
 * Send captured photo to /api/vision/query
 * On success: display Visual RAG response with answer directly (no second /api/chat).
 * On error: display error text via toast and system message.
 */
async function sendPhotoForClassification(blob) {
  var closeLoading = Toast.loading("正在识别设备...");

  try {
    var formData = new FormData();
    formData.append("image", blob, "photo.jpg");

    // Include optional user question from the input
    var userQuestion = dom.input.value.trim();
    if (userQuestion) {
      formData.append("question", userQuestion);
    }

    var resp = await fetch("/api/vision/query", {
      method: "POST",
      body: formData,
    });

    var result = await resp.json().catch(function() { return null; });
    closeLoading();

    if (!result) {
      Toast.show("识别服务返回异常", 3000);
      return;
    }

    if (!resp.ok) {
      displayVisualError(result, resp.status);
      return;
    }

    displayVisualRagResponse(result);

  } catch (err) {
    closeLoading();
    Toast.show(
      err.name === "TypeError"
        ? "无法连接识别服务，请检查网络"
        : "设备识别失败: " + (err.message || "未知错误"),
      3000
    );
  }
}

/**
 * Display a system-style bubble showing visual RAG candidate summary.
 * @param {string} categoryPrefix - coarse category info HTML fragment
 * @param {string[]} candidateLines - array of candidate HTML lines (label + score)
 */
function addVisionResultBubble(categoryPrefix, candidateLines) {
  hideEmptyState();

  var el = document.createElement("div");
  el.className = "message message--system";

  var bubble = document.createElement("div");
  bubble.className = "message__bubble";

  var contentEl = document.createElement("div");
  contentEl.className = "message__content";
  var summaryHtml = candidateLines.join(", ");
  contentEl.innerHTML = "<p>📷 " + (categoryPrefix || "") + "候选: " + summaryHtml + "</p>";

  var timeEl = document.createElement("div");
  timeEl.className = "message__time";
  timeEl.textContent = formatTime();

  bubble.appendChild(contentEl);
  bubble.appendChild(timeEl);
  el.appendChild(bubble);
  dom.messages.appendChild(el);

  scrollToBottom();
}
