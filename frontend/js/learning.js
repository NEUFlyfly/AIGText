/**
 * IoTBrain — Learning Report
 * Analytics dashboard: weekly stats, device distribution, daily trends, recent conversations.
 * Data source: Server API (SQLite database).
 */
(function () {
  "use strict";

  const dom = {
    backBtn:      document.getElementById("back-btn"),
    statCount:    document.getElementById("stat-count"),
    statTop:      document.getElementById("stat-top"),
    statDays:     document.getElementById("stat-days"),
    categoryCtx:  document.getElementById("category-chart"),
    dailyCtx:     document.getElementById("daily-chart"),
    recentList:   document.getElementById("recent-list"),
    reviewBtn:    document.getElementById("review-btn"),
    emptyState:   document.getElementById("empty-state"),
    statsGrid:    document.getElementById("stats-grid"),
  };

  // ── Init ──
  dom.backBtn.addEventListener("click", function () { Nav.go("index.html"); });
  dom.reviewBtn.addEventListener("click", function () { Nav.go("chat.html"); });
  loadData();

  async function loadData() {
    try {
      let allMessages = null;

      // 优先使用 chat.js 后台预加载的缓存
      try {
        const cached = sessionStorage.getItem("aigtext_learning_messages");
        if (cached) {
          allMessages = JSON.parse(cached);
        }
      } catch (_) { /* ignore cache parse errors */ }

      // 缓存未命中 → 单次请求获取所有对话的所有消息
      if (!allMessages) {
        const resp = await fetch("/api/learning/messages");
        if (!resp.ok) throw new Error("HTTP " + resp.status);
        allMessages = await resp.json();
        // 写入缓存供下次使用
        try {
          sessionStorage.setItem("aigtext_learning_messages", JSON.stringify(allMessages));
        } catch (_) { /* ignore quota */ }
      }

      if (!allMessages || allMessages.length === 0) {
        showEmptyState();
        return;
      }

      // 标准化字段名（兼容旧格式）
      allMessages.forEach(function (m) {
        if (m.conversation_id && !m.conversationId) {
          m.conversationId = m.conversation_id;
        }
        if (!m.timestamp && m.created_at) {
          m.timestamp = new Date(m.created_at).getTime();
        }
      });

      analyze(allMessages);
    } catch (e) {
      console.error("Failed to load data:", e);
      showEmptyState();
    }
  }

  function showEmptyState() {
    dom.statsGrid.style.display = "none";
    document.querySelector(".charts-section:nth-of-type(1)").style.display = "none";
    document.querySelector(".charts-section:nth-of-type(2)").style.display = "none";
    document.querySelector(".recent-section").style.display = "none";
    dom.emptyState.classList.remove("hidden");
  }

  // 判断是否为识别失败的回复
  function isVisionFailure(content) {
    if (!content) return true;
    const s = content.trim();
    return !s || s === "识别失败" || s.startsWith("识别失败") || s === "识别失败:";
  }

  // ── Analysis ──
  function analyze(messages) {
    const now = new Date();
    const weekStart = new Date(now);
    weekStart.setDate(now.getDate() - now.getDay());
    weekStart.setHours(0, 0, 0, 0);

    // 从数据库返回的消息中找出视觉识别对
    // 数据库消息格式: {id, role, content, image_data, created_at}
    const visionPairs = [];  // {userMsg, assistantMsg, timestamp, content, imageData}
    const categoryCount = {};
    const dailyCount = {};
    const studyDays = new Set();

    // 遍历消息，找到图片+回复的配对
    // 策略1: 用户消息有 image_data + 下一条是助手回复
    // 策略2: 助手回复包含 "**设备**" 格式（即使没有 image_data）
    for (let i = 0; i < messages.length - 1; i++) {
      const msg = messages[i];
      const nextMsg = messages[i + 1];
      
      // 策略1: 有 image_data（且非识别失败）
      if (msg.role === "user" && msg.image_data && nextMsg.role === "assistant") {
        if (isVisionFailure(nextMsg.content)) { i++; continue; }
        const pair = {
          userMsg: msg,
          assistantMsg: nextMsg,
          timestamp: msg.created_at ? new Date(msg.created_at).getTime() : now.getTime(),
          content: nextMsg.content,
          imageData: msg.image_data,
        };
        visionPairs.push(pair);

        const d = new Date(pair.timestamp);
        const dayKey = d.toISOString().slice(0, 10);
        
        if (d >= weekStart) {
          const catName = extractCategoryFromContent(nextMsg.content);
          if (catName) {
            categoryCount[catName] = (categoryCount[catName] || 0) + 1;
          }
          dailyCount[dayKey] = (dailyCount[dayKey] || 0) + 1;
          studyDays.add(dayKey);
        }
        continue; // 跳过策略2
      }
      
      // 策略2: 助手回复包含 **设备** 或视觉识别特征（且非识别失败）
      if (msg.role === "user" && nextMsg.role === "assistant") {
        if (isVisionFailure(nextMsg.content)) { i++; continue; }
        const content = nextMsg.content || "";
        const hasVisionMarker = content.indexOf("**设备**") !== -1 ||
                                content.indexOf("**Device**") !== -1 ||
                                content.indexOf("视觉分数") !== -1 ||
                                content.indexOf("visual score") !== -1;
        
        if (hasVisionMarker) {
          const pair = {
            userMsg: msg,
            assistantMsg: nextMsg,
            timestamp: msg.created_at ? new Date(msg.created_at).getTime() : now.getTime(),
            content: content,
            imageData: msg.image_data || msg.image_url || null,
          };
          visionPairs.push(pair);

          const d = new Date(pair.timestamp);
          const dayKey = d.toISOString().slice(0, 10);
          
          if (d >= weekStart) {
            const catName = extractCategoryFromContent(content);
            if (catName) {
              categoryCount[catName] = (categoryCount[catName] || 0) + 1;
            }
            dailyCount[dayKey] = (dailyCount[dayKey] || 0) + 1;
            studyDays.add(dayKey);
          }
        }
      }
    }

    if (visionPairs.length === 0) {
      showEmptyState();
      return;
    }

    // Stats
    const weekCount = visionPairs.filter(function (p) {
      return new Date(p.timestamp) >= weekStart;
    }).length;
    dom.statCount.textContent = weekCount;

    const topCategory = Object.keys(categoryCount).sort(function (a, b) {
      return (categoryCount[b] || 0) - (categoryCount[a] || 0);
    })[0];
    dom.statTop.textContent = topCategory ? truncate(topCategory, 8) : "—";
    dom.statTop.style.fontSize = topCategory && topCategory.length > 6 ? "18px" : "24px";

    const totalDays = studyDays.size;
    dom.statDays.textContent = totalDays + " 天";

    // Charts
    renderCategoryChart(categoryCount);
    renderDailyChart(dailyCount, now);

    // Recent vision conversations
    renderRecentList(visionPairs);
  }

  function extractCategoryFromContent(content) {
    if (!content) return null;
    
    // 先尝试从 "**设备**: xxx" 格式提取
    const deviceMatch = content.match(/\*\*设备\*\*[：:]\s*([^\n]+)/);
    if (deviceMatch) {
      return deviceMatch[1].trim();
    }
    
    // 再尝试从 "Device:" 格式提取
    const devMatch = content.match(/Device:\s*([^\n]+)/i);
    if (devMatch) {
      return devMatch[1].trim();
    }
    
    // 从第一行提取（如果是设备名）
    const lines = content.split('\n');
    if (lines.length > 0 && lines[0].trim()) {
      const firstLine = lines[0].trim();
      // 如果第一行不长且可能是设备名
      if (firstLine.length < 50 && !firstLine.includes('根据') && !firstLine.includes('图片')) {
        // 移除 markdown 格式
        return firstLine.replace(/\*\*/g, '').replace(/#/g, '').trim();
      }
    }
    
    // 关键词兜底
    const keywords = [
      "Arduino", "Raspberry", "ESP32", "ESP8266", "Jetson",
      "STM32", "FPGA", "超声波", "温湿度", "陀螺仪", "摄像头",
      "motor", "sensor", "camera", "LED", "relay"
    ];
    for (var i = 0; i < keywords.length; i++) {
      if (content.toLowerCase().indexOf(keywords[i].toLowerCase()) !== -1) {
        return keywords[i] + " 相关设备";
      }
    }
    
    return null;
  }

  function truncate(str, max) {
    return str.length > max ? str.slice(0, max) + "…" : str;
  }

  // ── Chart: Category Distribution ──
  function renderCategoryChart(data) {
    const keys = Object.keys(data);
    if (keys.length === 0) {
      dom.categoryCtx.parentElement.innerHTML = '<p style="color:#8e8ea0;text-align:center;padding:40px 0">本周暂无数据</p>';
      return;
    }

    const labels = keys.map(function (k) { return truncate(k, 10); });
    const values = keys.map(function (k) { return data[k]; });
    const colors = [
      "rgba(108,92,231,0.8)",
      "rgba(168,85,247,0.8)",
      "rgba(77,171,247,0.8)",
      "rgba(255,107,107,0.8)",
      "rgba(255,169,77,0.8)",
      "rgba(52,211,153,0.8)",
    ];

    new Chart(dom.categoryCtx, {
      type: "doughnut",
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: colors.slice(0, keys.length),
          borderWidth: 0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "60%",
        plugins: {
          legend: {
            position: "bottom",
            labels: {
              padding: 12,
              usePointStyle: true,
              font: { size: 11 },
            },
          },
        },
      },
    });
  }

  // ── Chart: Daily Trend ──
  function renderDailyChart(data, now) {
    const labels = [];
    const values = [];
    for (var i = 6; i >= 0; i--) {
      const d = new Date(now);
      d.setDate(now.getDate() - i);
      const key = d.toISOString().slice(0, 10);
      const dayLabel = (d.getMonth() + 1) + "/" + d.getDate();
      labels.push(dayLabel);
      values.push(data[key] || 0);
    }

    new Chart(dom.dailyCtx, {
      type: "bar",
      data: {
        labels: labels,
        datasets: [{
          data: values,
          backgroundColor: "rgba(108,92,231,0.6)",
          borderRadius: 6,
          borderSkipped: false,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
        },
        scales: {
          y: {
            beginAtZero: true,
            ticks: { stepSize: 1 },
            grid: { display: false },
          },
          x: {
            grid: { display: false },
          },
        },
      },
    });
  }

  // ── Recent Vision Conversations ──
  function renderRecentList(pairs) {
    // Show last 5 vision results, sorted by timestamp desc
    const sorted = pairs.slice().sort(function (a, b) {
      return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
    }).slice(0, 5);

    if (sorted.length === 0) {
      dom.recentList.innerHTML = '<p style="color:#8e8ea0;font-size:13px;text-align:center;padding:16px 0">暂无识别记录</p>';
      return;
    }

    dom.recentList.innerHTML = sorted.map(function (pair) {
      const device = extractCategoryFromContent(pair.content) || "未知设备";
      const ts = pair.timestamp ? formatTime(new Date(pair.timestamp)) : "未知时间";
      const preview = pair.content ? truncate(pair.content, 30) : "";
      const hasImage = !!pair.imageData;

      return '<div class="recent-item">'
        + '<div class="recent-item__icon">'
        + (hasImage
          ? '<img src="' + pair.imageData + '" style="width:100%;height:100%;object-fit:cover;border-radius:8px;">'
          : '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>')
        + '</div>'
        + '<div class="recent-item__body">'
        + '<div class="recent-item__title">' + escapeHtml(device) + '</div>'
        + '<div class="recent-item__time">' + ts + ' · ' + escapeHtml(preview) + '</div>'
        + '</div>'
        + '</div>';
    }).join("");
  }

  function formatTime(d) {
    var now = new Date();
    var diff = now - d;
    if (diff < 60 * 1000) return "刚刚";
    if (diff < 3600 * 1000) return Math.floor(diff / 60000) + " 分钟前";
    if (diff < 86400 * 1000) return Math.floor(diff / 3600000) + " 小时前";
    return (d.getMonth() + 1) + "月" + d.getDate() + "日";
  }

  function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

})();
