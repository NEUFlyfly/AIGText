/**
 * IoTBrain Chat — 简化可靠版本
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
    compareMode: false,
    comparePhotos: [],  // Array of { blob, url }
    cameraStream: null,     // MediaStream from getUserMedia
    cameraFacingMode: "environment",  // "environment" 或 "user"
    cameraCapturedBlob: null,  // 拍照后的 Blob
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
    compareToggleBtn: document.getElementById("compare-toggle-btn"),
    compareStrip: document.getElementById("compare-strip"),
    // camera modal
    cameraModal: document.getElementById("camera-modal"),
    cameraVideo: document.getElementById("camera-video"),
    cameraCaptureBtn: document.getElementById("camera-capture"),
    cameraFlipBtn: document.getElementById("camera-flip"),
    cameraCloseBtn: document.getElementById("camera-close"),
    cameraPreview: document.getElementById("camera-preview"),
    cameraPreviewImg: document.getElementById("camera-preview-img"),
    cameraRetakeBtn: document.getElementById("camera-retake"),
    cameraConfirmBtn: document.getElementById("camera-confirm"),
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
  // 标签匹配：大小写不敏感，允许 <think> 内有可选空格（如 <think >）
  function splitThinkAnswer(raw) {
    const text = raw || "";
    if (!text) return { think: "", answer: "" };

    // 匹配开标签：<think> 或 <Think> 或 <think >
    const openMatch = text.match(/<think\s*>/i);
    if (!openMatch || openMatch.index === undefined) {
      return { think: "", answer: text };
    }

    const bodyStart = openMatch.index + openMatch[0].length;

    // 匹配闭标签：</think> 或 </Think> 或 </think >
    const remainder = text.slice(bodyStart);
    const closeMatch = remainder.match(/<\/think\s*>/i);

    if (!closeMatch || closeMatch.index === undefined) {
      // 流式中：闭标签未到，剩余全部作 think（之前 slice(bodyStart, -1) 的截断 bug）
      return { think: text.slice(bodyStart).trim(), answer: "" };
    }

    const closeIdx = bodyStart + closeMatch.index;
    const thinkBody = text.slice(bodyStart, closeIdx).trim();
    const answerBody = text.slice(closeIdx + closeMatch[0].length).trim();

    // 安全网：如果 answer 中意外残留 think 标签，递归剥离
    if (/<think\s*>/i.test(answerBody)) {
      const nested = splitThinkAnswer(answerBody);
      return { think: thinkBody + (nested.think ? "\n" + nested.think : ""), answer: nested.answer };
    }

    return { think: thinkBody, answer: answerBody };
  }

   function formatMd(text) {
    if (!text) return "";

    // ── 提取 <comparison>...</comparison> 块，保护 HTML 表格不被转义 ──
    // 与 splitThinkAnswer 同理：用显式标签做可靠提取，避免正则脆弱性
    var cmpBlocks = [];
    var safe = text.replace(/<comparison>([\s\S]*?)<\/comparison>/gi, function (_, body) {
      cmpBlocks.push(body);
      return "\0CMP" + (cmpBlocks.length - 1) + "\0";
    });

    let html = safe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");

    // 表格：匹配连续以 | 开头（可带前导空白）的行（必须在 \n → <br> 之前）
    html = html.replace(/(?:^|\n)((?:\s*\|[^\n]+\n)+)/g, function(block, tableBlock) {
      const lines = tableBlock.split("\n").filter(function(l) {
        return /\|/.test(l);
      });
      if (lines.length < 2) return block;

      // 检测分隔行（行内主要由 | - : 空格组成，支持多列表格）
      let sepIdx = -1;
      for (let i = 1; i < lines.length && i < 3; i++) {
        if (/^\s*\|[\s\-:]+\|([\s\-:]+\|)+\s*$/.test(lines[i])) { sepIdx = i; break; }
      }
      if (sepIdx === -1) sepIdx = 1;

      const dataRows = [];
      for (let i = 0; i < lines.length; i++) {
        if (i === sepIdx) continue;
        dataRows.push(lines[i]);
      }
      if (dataRows.length < 2) return block;

      let table = "<table>";
      dataRows.forEach(function(row, i) {
        const cells = row.split("|").map(function(c) { return c.trim(); });
        // 去掉首尾空串（由首尾 | 产生）
        if (cells.length > 1 && cells[0] === "") cells.shift();
        if (cells.length > 1 && cells[cells.length - 1] === "") cells.pop();
        const tag = i === 0 ? "th" : "td";
        table += "<tr>" + cells.map(function(c) { return "<" + tag + ">" + c + "</" + tag + ">"; }).join("") + "</tr>";
      });
      table += "</table>";
      return table;
    });

    html = html
      .replace(/```[\s\S]*?```/g, match => `<pre><code>${match.slice(3, -3)}</code></pre>`)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/\*([^*]*)\*/g, "<em>$1</em>")
      .replace(/\n/g, "<br>");

    // ── 恢复 <comparison> 块（原样 HTML，不被转义） ──
    html = html.replace(/\0CMP(\d+)\0/g, function (_, i) {
      return cmpBlocks[parseInt(i, 10)] || "";
    });

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
    let imageHtml = "";
    if (Array.isArray(imageUrl) && imageUrl.length) {
      imageHtml = imageUrl.map(url => `<img class="message__user-image" src="${url}">`).join("");
    } else if (imageUrl) {
      imageHtml = `<img class="message__user-image" src="${imageUrl}">`;
    }

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

    if (state.compareMode && state.comparePhotos.length >= 2) {
      await ensureConversation();
      sendCompare();
      return;
    }

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

      // Turn off compare mode (mutually exclusive)
      if (state.compareMode) {
        state.compareMode = false;
        document.body.classList.remove("compare-mode");
        dom.compareToggleBtn.classList.remove("active");
        state.comparePhotos.forEach(p => URL.revokeObjectURL(p.url));
        state.comparePhotos = [];
        renderCompareStrip();
        dom.compareStrip.classList.add("hidden");
        dom.input.placeholder = "输入消息...";
      }
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

  // ── Compare mode ──

  function toggleCompareMode() {
    state.compareMode = !state.compareMode;

    if (state.compareMode) {
      // Turn off voice mode (mutually exclusive)
      if (state.voiceMode) {
        state.voiceMode = false;
        const app = document.getElementById("app");
        app.classList.remove("voice-mode-active");
        dom.voiceToggleBtn.classList.remove("active");
        dom.input.style.display = "";
        dom.sendBtn.style.display = "";
        dom.voiceMicBtn.classList.add("hidden");
        removeVoiceStatus();
        stopSpeaking();
        stopVoiceRecording();
      }
      document.body.classList.add("compare-mode");
      dom.compareToggleBtn.classList.add("active");
      dom.compareStrip.classList.remove("hidden");
      dom.input.placeholder = "输入对比需求...";
    } else {
      document.body.classList.remove("compare-mode");
      dom.compareToggleBtn.classList.remove("active");
      dom.compareStrip.classList.add("hidden");
      state.comparePhotos = [];
      renderCompareStrip();
      dom.input.placeholder = "输入消息...";
    }
  }

  function renderCompareStrip() {
    dom.compareStrip.innerHTML = "";

    state.comparePhotos.forEach((photo, index) => {
      const el = document.createElement("div");
      el.className = "compare-photo";
      el.innerHTML =
        `<img src="${photo.url}" alt="对比照片 ${index + 1}">` +
        `<button class="compare-photo-remove" data-index="${index}" aria-label="移除照片">&times;</button>`;
      dom.compareStrip.appendChild(el);
    });

    if (state.comparePhotos.length < 3) {
      const emptyEl = document.createElement("div");
      emptyEl.className = "compare-photo compare-photo--empty";
      emptyEl.innerHTML = `<span class="compare-photo-placeholder">+</span>`;
      dom.compareStrip.appendChild(emptyEl);
    }

    // Bind remove handlers
    dom.compareStrip.querySelectorAll(".compare-photo-remove").forEach(btn => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.dataset.index, 10);
        const removed = state.comparePhotos.splice(idx, 1)[0];
        if (removed) URL.revokeObjectURL(removed.url);
        renderCompareStrip();
      });
    });
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
      if (result.answer) {
        answer += result.answer;
      } else if (result.status === "ok" || result.status === "OK") {
        const cand = result.visual_candidates?.[0];
        if (cand) answer += "**设备**: " + (cand.sub_category || cand.doc_id || "未知") + "\n\n";
        answer += result.message || "分析完成";
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

  async function sendCompare() {
    setStreaming(true);
    state.abortController = new AbortController();
    const question = dom.input.value.trim();
    dom.input.value = "";

    // Render user message with embedded images
    const imageElementsHtml = state.comparePhotos
      .map(p => `<img class="message__user-image" src="${p.url}" alt="对比照片">`)
      .join("");

    hideEmpty();
    const userDiv = document.createElement("div");
    userDiv.className = "message message--user";
    userDiv.innerHTML =
      `<div class="message__avatar message__avatar--user">🧑</div>` +
      `<div class="message__bubble">` +
        imageElementsHtml +
        (question ? `<div class="message__content">${formatMd(question)}</div>` : "") +
        `<div class="message__time">${formatTime()}</div>` +
      `</div>`;
    dom.messages.appendChild(userDiv);
    scrollToBottom();

    // Save photos as data URLs for persistence
    const imageDataUrls = await Promise.all(
      state.comparePhotos.map(p => fileToDataUrl(p.blob))
    );
    state.messages.push({
      role: "user",
      content: question,
      timestamp: Date.now(),
      image_data: imageDataUrls,
    });

    // Show AI bubble
    const aiEl = renderMessage("assistant", "🔬 对比分析中...");

    const form = new FormData();
    state.comparePhotos.forEach((photo, i) => {
      form.append(`image_${i}`, photo.blob, `compare_${i}.jpg`);
    });
    form.append("question", question || "");
    form.append("stream", "true");

    let fullReply = "";

    try {
      const resp = await fetch("/api/vision/compare", {
        method: "POST",
        body: form,
        signal: state.abortController ? state.abortController.signal : undefined,
      });

      if (!resp.ok) throw new Error("HTTP " + resp.status);

      // 检查响应类型：流式 SSE vs JSON
      const ct = resp.headers.get("Content-Type") || "";
      if (ct.indexOf("text/event-stream") !== -1) {
        // 流式 SSE 响应
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
              }
            } catch (e) { /* 跳过非 JSON 行 */ }
          }
        }

        if (!fullReply) aiEl.innerHTML = renderAnswerBubble("(无回复)");
      } else {
        // 非流式（错误/降级）JSON 响应
        const result = await resp.json();
        const answer = result.answer || result.message || "对比分析完成";
        fullReply = answer;
        aiEl.innerHTML = renderAnswerBubble(answer);
      }

      state.messages.push({
        role: "assistant",
        content: fullReply,
        timestamp: Date.now(),
        metadata: { type: "compare_result" },
      });
    } catch (err) {
      if (err.name !== "AbortError") {
        aiEl.innerHTML = formatMd("对比分析失败: " + err.message);
      }
    } finally {
      // Clean up compare photos
      state.comparePhotos.forEach(p => URL.revokeObjectURL(p.url));
      state.comparePhotos = [];
      renderCompareStrip();
      setStreaming(false);
      state.abortController = null;
      if (state.messages.length > 0) saveConversation();
    }
  }

  function stopGeneration() {
    if (state.abortController) state.abortController.abort();
    setStreaming(false);
  }

  // 图片选择
  dom.galleryButton.addEventListener("click", () => dom.galleryInput.click());
  dom.galleryInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) showThumb(file);
    dom.galleryInput.value = "";
  });

  // 相机：优先用 getUserMedia 实时拍摄；不支持时回退到 capture="environment" 的隐藏 input
  dom.cameraButton.addEventListener("click", openCameraModal);
  dom.cameraCaptureInput.addEventListener("change", e => {
    const file = e.target.files[0];
    if (file) showThumb(file);
    dom.cameraCaptureInput.value = "";
  });

  // ── Camera Modal ──

  function openCameraModal() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      dom.cameraCaptureInput.click();
      return;
    }
    dom.cameraModal.classList.remove("hidden");
    dom.cameraModal.setAttribute("aria-hidden", "false");
    dom.cameraPreview.classList.add("hidden");
    startCamera();
  }

  function closeCameraModal() {
    Camera.stop(state.cameraStream);
    state.cameraStream = null;
    dom.cameraVideo.srcObject = null;
    dom.cameraModal.classList.add("hidden");
    dom.cameraModal.setAttribute("aria-hidden", "true");
  }

  async function startCamera() {
    try {
      const stream = await Camera.open(dom.cameraVideo, state.cameraFacingMode);
      state.cameraStream = stream;
    } catch (err) {
      console.error("Camera open failed:", err);
      closeCameraModal();
      dom.cameraCaptureInput.click();
    }
  }

  async function flipCamera() {
    if (!state.cameraStream) return;
    state.cameraFacingMode = state.cameraFacingMode === "environment" ? "user" : "environment";
    Camera.stop(state.cameraStream);
    state.cameraStream = null;
    await startCamera();
  }

  async function capturePhoto() {
    if (!state.cameraStream) return;
    try {
      const blob = await Camera.capture(dom.cameraVideo, 0.9);
      state.cameraCapturedBlob = blob;
      dom.cameraPreviewImg.src = URL.createObjectURL(blob);
      dom.cameraPreview.classList.remove("hidden");
    } catch (err) {
      console.error("Photo capture failed:", err);
    }
  }

  function retakePhoto() {
    dom.cameraPreview.classList.add("hidden");
    if (state.cameraCapturedBlob) {
      URL.revokeObjectURL(dom.cameraPreviewImg.src);
      state.cameraCapturedBlob = null;
    }
  }

  function confirmPhoto() {
    if (!state.cameraCapturedBlob) return;
    showThumb(state.cameraCapturedBlob);
    closeCameraModal();
    state.cameraCapturedBlob = null;
  }

  dom.cameraCloseBtn.addEventListener("click", closeCameraModal);
  dom.cameraFlipBtn.addEventListener("click", flipCamera);
  dom.cameraCaptureBtn.addEventListener("click", capturePhoto);
  dom.cameraRetakeBtn.addEventListener("click", retakePhoto);
  dom.cameraConfirmBtn.addEventListener("click", confirmPhoto);

  function showThumb(file) {
    if (state.compareMode) {
      if (state.comparePhotos.length >= 3) return;
      state.comparePhotos.push({ blob: file, url: URL.createObjectURL(file) });
      renderCompareStrip();
      dom.cameraThumb.classList.add("hidden");
      return;
    }
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
  dom.compareToggleBtn.addEventListener("click", toggleCompareMode);
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
          prefetchLearningData();
          return;
        }
      } catch (e) {
        // 加载失败则回退到新对话
        localStorage.removeItem("aigtext_last_conv");
      }
    }
    prefetchLearningData();
  }

  // 后台静默预加载学习界面数据，用户点开学习页面时缓存命中即秒开
  function prefetchLearningData() {
    if (sessionStorage.getItem("aigtext_learning_messages")) return;
    fetch("/api/learning/messages")
      .then(function (r) { return r.json(); })
      .then(function (data) {
        try { sessionStorage.setItem("aigtext_learning_messages", JSON.stringify(data)); } catch (_) {}
      })
      .catch(function () {}); // 静默失败，学习页自行请求
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
