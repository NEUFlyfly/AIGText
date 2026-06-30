/**
 * AIGText — Learning Report
 * Analytics dashboard: weekly stats, device distribution, daily trends, recent conversations.
 * Data source: localStorage (client-side message history).
 */
(function () {
  "use strict";

  const STORAGE_KEY = "aigtext_messages";

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

  function loadData() {
    var messages = loadMessages();
    if (!messages || messages.length === 0) {
      showEmptyState();
      return;
    }
    analyze(messages);
  }

  function loadMessages() {
    try {
      // Try new format first
      var raw = localStorage.getItem("aigtext_chat_state_v2");
      if (raw) {
        var data = JSON.parse(raw);
        if (data.messages && data.messages.length) return data.messages;
      }
      // Fall back to old format
      raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return [];
      return JSON.parse(raw);
    } catch (e) {
      console.error("Failed to load messages:", e);
      return [];
    }
  }

  function showEmptyState() {
    dom.statsGrid.style.display = "none";
    dom.emptyState.classList.remove("hidden");
  }

  // ── Analysis ──
  function analyze(messages) {
    const now = new Date();
    const weekStart = new Date(now);
    weekStart.setDate(now.getDate() - now.getDay());
    weekStart.setHours(0, 0, 0, 0);

    const visionMessages = [];
    const categoryCount = {};
    const dailyCount = {};
    const studyDays = new Set();

    messages.forEach(function (msg) {
      // Match both new (metadata.type === "vision_result") and legacy (isVision === true)
      var isVisionMsg = (msg.metadata && (msg.metadata.type === "vision_result" ||
                          msg.metadata.type === "vision" ||
                          (msg.metadata.visual_candidates && msg.metadata.visual_candidates.length)));

      if (!isVisionMsg && !msg.isVision) return;
      if (msg.role !== "assistant") return;

      visionMessages.push(msg);

      const d = new Date(msg.timestamp || now);
      const dayKey = d.toISOString().slice(0, 10);
      
      // Within this week?
      if (d >= weekStart) {
        // Extract category from metadata (new) or legacy fields
        var catName = (msg.metadata && msg.metadata.visual_candidates &&
                       msg.metadata.visual_candidates[0] &&
                       msg.metadata.visual_candidates[0].sub_category) ||
                      (msg.metadata && msg.metadata.visual_candidates &&
                       msg.metadata.visual_candidates[0] &&
                       msg.metadata.visual_candidates[0].coarse_category) ||
                      msg.coarseCategory ||
                      msg.subCategory ||
                      extractCategoryFromContent(msg.content);
        if (catName) {
          categoryCount[catName] = (categoryCount[catName] || 0) + 1;
        }
        dailyCount[dayKey] = (dailyCount[dayKey] || 0) + 1;
        studyDays.add(dayKey);
      }
    });

    if (visionMessages.length === 0) {
      showEmptyState();
      return;
    }

    // Stats
    const weekCount = visionMessages.filter(function (m) {
      return new Date(m.timestamp || now) >= weekStart;
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
    renderRecentList(visionMessages);
  }

  function extractCategoryFromContent(content) {
    if (!content) return null;
    const keywords = [
      "Arduino", "Raspberry", "ESP32", "ESP8266", "Jetson",
      "STM32", "FPGA", "超声波", "温湿度", "陀螺仪", "摄像头"
    ];
    for (var i = 0; i < keywords.length; i++) {
      if (content.indexOf(keywords[i]) !== -1) {
        return keywords[i] + " 系列";
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
  function renderRecentList(messages) {
    // Show last 5 vision results, sorted by timestamp desc
    const sorted = messages.slice().sort(function (a, b) {
      return new Date(b.timestamp || 0) - new Date(a.timestamp || 0);
    }).slice(0, 5);

    if (sorted.length === 0) {
      dom.recentList.innerHTML = '<p style="color:#8e8ea0;font-size:13px;text-align:center;padding:16px 0">暂无识别记录</p>';
      return;
    }

    dom.recentList.innerHTML = sorted.map(function (msg) {
      const device = msg.subCategory || msg.coarseCategory || "未知设备";
      const ts = msg.timestamp ? formatTime(new Date(msg.timestamp)) : "未知时间";
      const preview = msg.content ? truncate(msg.content, 30) : "";

      return '<div class="recent-item">'
        + '<div class="recent-item__icon">'
        + '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 11.5a8.38 8.38 0 01-.9 3.8 8.5 8.5 0 01-7.6 4.7 8.38 8.38 0 01-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 01-.9-3.8 8.5 8.5 0 014.7-7.6 8.38 8.38 0 013.8-.9h.5a8.48 8.48 0 018 8v.5z"/></svg>'
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
