/**
 * AIGText — Shared Frontend Utilities
 * Navigation, toast, API helpers — used across all pages.
 */

/* ==================================================================
   DOM Helper
   ================================================================== */
const $ = (s, p) => (p || document).querySelector(s);
const $$ = (s, p) => Array.from((p || document).querySelectorAll(s));

/* ==================================================================
   Navigation
   ================================================================== */
const Nav = {
  /** Navigate to another page */
  go(url) {
    window.location.href = url;
  },

  /** Go back in history */
  back() {
    if (window.history.length > 1) {
      window.history.back();
    } else {
      this.go('index.html');
    }
  },

  /** Replace current page (no history entry) */
  replace(url) {
    window.location.replace(url);
  },
};

/* ==================================================================
   Toast Notification
   ================================================================== */
const Toast = {
  _timer: null,

  /** Show a brief toast message */
  show(msg, duration = 2000) {
    // Remove any existing toast
    const old = document.querySelector('.toast');
    if (old) old.remove();
    if (this._timer) clearTimeout(this._timer);

    const el = document.createElement('div');
    el.className = 'toast';
    el.textContent = msg;
    document.body.appendChild(el);

    this._timer = setTimeout(() => {
      if (el.parentNode) el.remove();
    }, duration + 300);
  },

  /** Show a persistent loading toast. Returns a close function. */
  loading(msg) {
    const el = document.createElement('div');
    el.className = 'toast';
    el.style.animation = 'toastIn 0.3s ease forwards';
    el.style.pointerEvents = 'auto';
    el.innerHTML = `<span style="display:inline-flex;align-items:center;gap:8px">
      <span class="spinner" style="width:14px;height:14px;border-width:2px"></span>${msg}
    </span>`;
    document.body.appendChild(el);
    return () => {
      el.style.animation = 'toastOut 0.3s ease forwards';
      setTimeout(() => { if (el.parentNode) el.remove(); }, 350);
    };
  },
};

/* ==================================================================
   API Helpers
   ================================================================== */
const API = {
  BASE: '',

  /** GET request */
  async get(path) {
    const resp = await fetch(this.BASE + path);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return resp.json();
  },

  /** POST request */
  async post(path, body, headers = {}) {
    const resp = await fetch(this.BASE + path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...headers },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const text = await resp.text().catch(() => 'Unknown');
      throw new Error(`HTTP ${resp.status}: ${text}`);
    }
    return resp;
  },

  /** POST multipart form data */
  async upload(path, formData, signal) {
    var opts = { method: 'POST', body: formData };
    if (signal) opts.signal = signal;
    const resp = await fetch(this.BASE + path, opts);
    if (!resp.ok) {
      var body = null;
      try { body = await resp.json(); } catch (_) { /* ignore parse errors */ }
      var msg = body && body.message ? body.message : ('HTTP ' + resp.status);
      throw new Error(msg);
    }
    return resp.json();
  },

  /** Health check */
  async health() {
    try {
      const resp = await fetch(this.BASE + '/api/health');
      return resp.ok;
    } catch (_e) {
      return false;
    }
  },
};

/* ==================================================================
   Camera Helper
   ================================================================== */
const Camera = {
  /** Request camera access and return video stream */
  async open(videoEl, facingMode = 'environment') {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode, width: { ideal: 1920 }, height: { ideal: 1080 } },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      return stream;
    } catch (err) {
      console.error('Camera error:', err);
      throw err;
    }
  },

  /** Capture a frame from video element as a Blob */
  capture(videoEl, quality = 0.85) {
    const canvas = document.createElement('canvas');
    canvas.width = videoEl.videoWidth;
    canvas.height = videoEl.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0);
    return new Promise((resolve) => {
      canvas.toBlob((blob) => resolve(blob), 'image/jpeg', quality);
    });
  },

  /** Stop all tracks in a stream */
  stop(stream) {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
    }
  },
};

/* ==================================================================
   Format Helpers
   ================================================================== */
function formatTime() {
  const d = new Date();
  return String(d.getHours()).padStart(2, '0') + ':' +
         String(d.getMinutes()).padStart(2, '0');
}

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / 1048576).toFixed(1) + ' MB';
}

/* ==================================================================
   Page Initialization Hook
   Each page calls Page.init() in its own script.
   ================================================================== */
const Page = {
  _backBtn: null,

  init(backUrl) {
    this._backBtn = $('.nav-bar__back');
    if (this._backBtn) {
      if (backUrl) {
        this._backBtn.addEventListener('click', () => Nav.go(backUrl));
      } else {
        this._backBtn.addEventListener('click', () => Nav.back());
      }
    }
  },
};
