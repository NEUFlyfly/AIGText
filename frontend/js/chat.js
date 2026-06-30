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
    voiceMode: false,
    mediaRecorder: null,
    speechRecognition: null,
    voiceTranscript: "",
    audioChunks: [],
    lastSpokenText: "",
  };

  const dom = {
    messages: document.getElementById("messages"),
    emptyState: document.getElementById("empty-state"),
    input: document.getElementById("message-input"),
    sendBtn: document.getElementById("send-button"),
    stopBtn: document.getElementById("stop-button"),
    newChatBtn: document.getElementById("new-chat-btn"),
    ragSwitch: document.getElementById("rag-switch"),
    headerBack: document.getElementById("header-back-btn"),
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
    voiceToggleBtn: document.getElementById("voice-toggle-btn"),
    voiceMicBtn: document.getElementById("voice-mic-btn"),
    // voiceFields set in toggleVoiceMode
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

  // 切分原始文本为 { think, answer }：
  // - 有完整 <think>...</think> 时，提取两段
  // - 流式场景只有 <think> 未闭合时，把已累积部分当 think、answer 为空
  // - 完全没有 <think> 标签时，整段当作 answer
  function splitThinkAnswer(raw) {
    const trimmed = (raw || "").trim();
    if (!trimmed) return { think: "", answer: "" };

    const startIdx = trimmed.indexOf("<think>");
    if (startIdx === -1) {
      // 没有 think 标签，整段作 answer
      return { think: "", answer: trimmed };
    }

    const afterStart = trimmed.indexOf(">", startIdx);
    if (afterStart === -1) {
      // "<think>" 还没闭合（流式中），整段视作 think
      return { think: trimmed.slice(startIdx + 6).trim(), answer: "" };
    }
    const bodyStart = afterStart + 1;

    const endIdx = trimmed.indexOf("</think>");
    if (endIdx === -1) {
      // 开始标签已闭合，但结束标签未到 —— 流式中
      return { think: trimmed.slice(bodyStart, endIdx).trim(), answer: "" };
    }

    const thinkBody = trimmed.slice(bodyStart, endIdx).trim();
    const answerBody = trimmed.slice(endIdx + 10).trim(); // "</think>".length = 10
    return { think: thinkBody, answer: answerBody };
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
      .replace(/\*([^*]*)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br>");
    return html;
  }

  function renderAnswerBubble(raw) {
    // 切分 think/answer；answer 为空时降级显示 think 本体
    const { think, answer } = splitThinkAnswer(raw);
    const answerFallback = answer ? answer : (think ? "(等待回答部分…)" : "");
    return "" +
      (think
        ? "<div class='think-block'>" +
            "<details open>" +
              "<summary class='think-summary'><span class='think-icon'>💭</span><span class='think-label'>思考过程</span></summary>" +
              "<div class='think-content'>" + formatMd(think) + "</div>" +
            "</details>" +
          "</div>"
        : "") +
      "<div class='answer-block'>" + formatMd(answerFallback) + "</div>";
  }

  // 头像 emoji
  function avatarEmoji(role) {
    return role === "user" ? "🧑" : role === "assistant" ? "🤖" : "";
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

  function renderMessage(role, content, imageUrl) {
    hideEmpty();
    const div = document.createElement("div");
    div.className = "message message--" + role;
    const imageHtml = imageUrl ? `<img class="message__user-image" src="${imageUrl}">` : "";

    // user/assistant 加头像；system/error 不加
    const avatarHtml = (role === "user" || role === "assistant")
      ? `<div class="message__avatar message__avatar--${role}">${avatarEmoji(role)}</div>`
      : "";

    div.innerHTML =
      avatarHtml +
      "<div class='message__bubble'>" +
      imageHtml +
      "<div class='message__content'>" + formatMd(content) + "</div>" +
      "<div class='message__time'>" + formatTime() + "</div>" +
      "</div>";
    dom.messages.appendChild(div);
    scrollToBottom();
    return div.querySelector(".message__content");
  }

  // 把 File/Blob 转成 data URL（base64）以便持久化到数据库
  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(reader.result);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function handleSend() {
    if (state.streaming) return;
    const text = dom.input.value.trim();
    if (!text && !state.pendingPhoto) return;

    await ensureConversation();

    // 用户消息气泡（如果有图片，传入图片URL）
    let image_data = null;
    if (state.pendingPhoto) {
      // 转成 data URL 以便持久化到数据库
      image_data = await fileToDataUrl(state.pendingPhoto.blob);
    }
    renderMessage("user", text || "", image_data);
    state.messages.push({ role: "user", content: text, timestamp: Date.now(), image_data: image_data });
    dom.input.value = "";

    if (state.pendingPhoto) {
      // 图片 + 文字 → vision API
      sendVision(state.pendingPhoto.blob, text);
      URL.revokeObjectURL(state.pendingPhoto.url); // 释放临时 blob URL
      state.pendingPhoto = null;
      dom.cameraThumb.classList.add("hidden");
    } else {
      // 纯文本聊天
      streamChat(text);
    }
  }

  // ── Voice mode ──

  function toggleVoiceMode() {
    state.voiceMode = !state.voiceMode;
    const app = document.getElementById("app");

    if (state.voiceMode) {
      app.classList.add("voice-mode-active");
      dom.voiceToggleBtn.classList.add("active");
      dom.input.style.display = "none";
      dom.sendBtn.style.display = "none";
      dom.voiceMicBtn.classList.remove("hidden");
      ensureVoiceStatus();
      updateVoiceStatus("", "🎤", "点击麦克风开始录音");
    } else {
      app.classList.remove("voice-mode-active");
      dom.voiceToggleBtn.classList.remove("active");
      dom.input.style.display = "";
      dom.sendBtn.style.display = "";
      dom.voiceMicBtn.classList.add("hidden");
      removeVoiceStatus();
      stopSpeaking();
      stopVoiceRecording();
    }
  }

  function ensureVoiceStatus() {
    let el = document.getElementById("voice-status");
    if (!el) {
      el = document.createElement("div");
      el.id = "voice-status";
      el.className = "voice-status";
      el.innerHTML = "<span class='voice-status__icon'>🎤</span><span class='voice-status__text'></span>";
      dom.messages.parentNode.insertBefore(el, dom.messages.nextSibling);
    }
    return el;
  }

  function removeVoiceStatus() {
    const el = document.getElementById("voice-status");
    if (el) el.remove();
  }

  function updateVoiceStatus(className, icon, text) {
    const el = ensureVoiceStatus();
    el.className = "voice-status " + className;
    el.querySelector(".voice-status__icon").textContent = icon;
    el.querySelector(".voice-status__text").textContent = text;
  }

  function addSpeakingBars(el) {
    if (!el.querySelector(".voice-status__bars")) {
      const bars = document.createElement("span");
      bars.className = "voice-status__bars";
      bars.innerHTML = "<span></span><span></span><span></span><span></span><span></span>";
      el.appendChild(bars);
    }
  }

  // ── Voice recording + STT (Web Speech API) ──

  function startVoiceRecording() {
    stopSpeaking();
    state.voiceTranscript = "";

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      recognition.lang = "zh-CN";
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      recognition.continuous = true;

      recognition.onresult = (event) => {
        let interim = "";
        let final = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const r = event.results[i];
          if (r.isFinal) final += r[0].transcript;
          else interim += r[0].transcript;
        }
        state.voiceTranscript = final + interim;
        updateVoiceStatus("voice-status--recording", "🔴", state.voiceTranscript || "正在聆听...");
      };

      recognition.onerror = (event) => {
        console.warn("Speech recognition error:", event.error);
        if (event.error === "no-speech") {
          updateVoiceStatus("voice-status--recording", "🔴", "没听到声音，请再说一遍");
        } else if (event.error === "aborted") {
          // 正常停止
        } else {
          updateVoiceStatus("", "⚠️", "语音识别不可用: " + event.error);
        }
      };

      recognition.onend = () => {
        if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
          try { recognition.start(); } catch (e) {}
        }
      };

      state.speechRecognition = recognition;
      recognition.start();
    }

    navigator.mediaDevices.getUserMedia({ audio: true }).then(stream => {
      state.mediaRecorder = new MediaRecorder(stream, {
        mimeType: MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "audio/mp4"
      });
      state.audioChunks = [];
      state.mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) state.audioChunks.push(e.data);
      };
      state.mediaRecorder.onstop = () => {
        stream.getTracks().forEach(t => t.stop());
        state.mediaRecorder = null;
      };
      state.mediaRecorder.start();
      dom.voiceMicBtn.classList.add("recording");
    }).catch(err => {
      updateVoiceStatus("", "⚠️", "麦克风权限被拒绝");
      console.error("Voice recording error:", err);
    });
  }

  function stopVoiceRecording() {
    if (state.speechRecognition) {
      state.speechRecognition.stop();
      state.speechRecognition = null;
    }
    if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
      state.mediaRecorder.stop();
    }
    dom.voiceMicBtn.classList.remove("recording");

    const text = state.voiceTranscript.trim();
    state.voiceTranscript = ""; // 立即清空，防止切换到文本模式时重复发送
    if (text) {
      updateVoiceStatus("", "📤", "发送: " + text);
      dom.input.value = text;
      handleSend();
    } else {
      updateVoiceStatus("", "⚠️", "未识别到语音，请重试");
      setTimeout(() => {
        if (state.voiceMode) updateVoiceStatus("", "🎤", "点击麦克风开始录音");
      }, 2000);
    }
  }

  // ── Text-to-Speech ──

  function speakNewText(fullText) {
    if (!state.voiceMode || !("speechSynthesis" in window)) return;

    // 只朗读 answer 部分，跳过 <think> 标签内容（TTS 不应朗读思考过程）
    const answerText = splitThinkAnswer(fullText).answer || fullText;

    // 已经入队到 TTS 的部分不再重复发送
    const newText = answerText.substring(state.lastSpokenText.length).trim();
    if (!newText) return;

    // 只在碰到句子结束标点时才入队朗读，避免半句话的间隙停顿
    const hasBreak = /[。！？\n]/.test(newText);
    if (!hasBreak) return;

    // 标标记这段已入队（基于 answer 文本位置）
    state.lastSpokenText = answerText;

    // 显示朗读状态
    const el = ensureVoiceStatus();
    el.className = "voice-status voice-status--speaking";
    el.querySelector(".voice-status__icon").textContent = "🔊";
    el.querySelector(".voice-status__text").textContent = "朗读中...";
    addSpeakingBars(el);

    const voices = window.speechSynthesis.getVoices();
    const zhVoice = voices.find(v => v.lang.startsWith("zh")) || voices.find(v => v.lang.startsWith("cmn"));

    // 按句子分段一次性入队，浏览器连续播放无间隙
    const sentences = newText.split(/(?<=[。！？\n])/);
    let spokeCount = 0;
    sentences.forEach((sentence) => {
      const s = sentence.trim();
      if (!s) return;
      const utterance = new SpeechSynthesisUtterance(s);
      utterance.lang = "zh-CN";
      utterance.rate = 2.6;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      if (zhVoice) utterance.voice = zhVoice;
      spokeCount++;
      // 最后一句播完恢复待机
      if (spokeCount === sentences.filter(x => x.trim()).length) {
        utterance.onend = () => {
          setTimeout(() => {
            if (!window.speechSynthesis.speaking && state.voiceMode) {
              updateVoiceStatus("", "🎤", "点击麦克风继续录音");
            }
          }, 200);
        };
      }
      window.speechSynthesis.speak(utterance);
    });
  }
        }, 200);
      }
    };

    // 不 cancel — 让浏览器自然排队播放
    window.speechSynthesis.speak(utterance);
  }

  function stopSpeaking() {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
    }
    state.lastSpokenText = "";
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
          voice_mode: state.voiceMode,
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
      aiEl.innerHTML = renderAnswerBubble(fullReply);
              scrollToBottom();
              speakNewText(fullReply);
            }
          } catch (e) {}
        }
      }

      if (!fullReply) aiEl.innerHTML = renderAnswerBubble("(无回复)");
    } catch (err) {
      if (err.name !== "AbortError") {
        aiEl.innerHTML = formatMd("发送失败: " + err.message);
      }
  } finally {
    state.messages.push({ role: "assistant", content: fullReply, timestamp: Date.now() });
    setStreaming(false);
    state.abortController = null;
    if (state.messages.length > 0) saveConversation();
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
      if (result.status === "ok" || result.status === "OK") {
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
      if (state.messages.length > 0) saveConversation();
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
  dom.voiceToggleBtn.addEventListener("click", toggleVoiceMode);
  dom.voiceMicBtn.addEventListener("click", () => {
    if (state.mediaRecorder && state.mediaRecorder.state === "recording") {
      stopVoiceRecording();
    } else {
      startVoiceRecording();
    }
  });
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

  // 返回按钮
  if (dom.headerBack) {
    dom.headerBack.addEventListener("click", () => {
      window.location.href = "index.html";
    });
  }

  function closeSidebar() {
    dom.sidebar.classList.add("hidden");
    dom.sidebarOverlay.classList.add("hidden");
  }

  async function newChat() {
    state.conversationId = null;
    state.messages = [];
    dom.messages.innerHTML = "";
    dom.emptyState.style.display = "flex";
    localStorage.removeItem("aigtext_last_conv");
    closeSidebar();
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
      localStorage.setItem("aigtext_last_conv", id);
      dom.messages.innerHTML = "";
      // 渲染消息：assistant 用 renderAnswerBubble 支持 <think> 拆分，其余用 renderMessage
      conv.messages.forEach(m => {
        const el = renderMessage(m.role, m.content, m.image_data);
        if (m.role === "assistant" && el) {
          el.innerHTML = renderAnswerBubble(m.content);
        }
      });
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

  // 确保有对话 ID（懒创建）
  async function ensureConversation() {
    if (state.conversationId) return;
    const resp = await fetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: "新对话" }),
    });
    const data = await resp.json();
    state.conversationId = data.id;
    localStorage.setItem("aigtext_last_conv", data.id);
  }

  async function init() {
    state.conversationId = null;
    state.messages = [];
    // 恢复上次停留的对话
    const lastId = localStorage.getItem("aigtext_last_conv");
    if (lastId) {
      try {
        const resp = await fetch("/api/conversations/" + lastId);
        if (resp.ok) {
          const conv = await resp.json();
          state.conversationId = lastId;
          state.messages = conv.messages || [];
          dom.messages.innerHTML = "";
          conv.messages.forEach(m => {
            const el = renderMessage(m.role, m.content, m.image_data);
            if (m.role === "assistant" && el) {
              el.innerHTML = renderAnswerBubble(m.content);
            }
          });
          return;
        }
      } catch (e) {
        // 加载失败则回退到新对话
        localStorage.removeItem("aigtext_last_conv");
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
