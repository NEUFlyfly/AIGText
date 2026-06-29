/**
 * AIGText — 3D Modeling Frontend
 *
 * Photo upload UI, loading simulation, and Three.js 3D model viewer.
 * Uses shared Nav, Toast, API from common.js.
 * Three.js loaded dynamically via importmap + import().
 */

/* ==================================================================
   GLOBAL REFERENCES (from common.js — regular script, same scope)
   ================================================================== */
// $, $$, Nav, Toast, API, Camera, Page — available as global consts

/* ==================================================================
   STATE
   ================================================================== */
const state = {
  mode: 'upload',       // 'upload' | 'loading' | 'viewer'
  files: [],            // Array of File objects
  maxFiles: 6,
  deviceName: '我的模型',
  loadingTimer: null,
  progressInterval: null,
};

/* ==================================================================
   DOM REFERENCES
   ================================================================== */
const dom = {
  // Modes
  modeUpload:   $('#mode-upload'),
  modeLoading:  $('#mode-loading'),
  modeViewer:   $('#mode-viewer'),

  // Upload
  dropZone:     $('#drop-zone'),
  fileInput:    $('#file-input'),
  btnCamera:    $('#btn-camera'),
  btnGallery:   $('#btn-gallery'),
  thumbGrid:    $('#thumb-grid'),
  photoCount:   $('#photo-count'),
  btnStart:     $('#btn-start'),

  // Loading
  progressFill: $('#progress-fill'),
  btnCancel:    $('#btn-cancel'),

  // Viewer
  viewerCanvas: $('#viewer-canvas'),
  modelName:    $('#model-name'),
  btnRemodel:   $('#btn-remodel'),
};

/* ==================================================================
   THREE.JS GLOBALS (loaded dynamically)
   ================================================================== */
let THREE = null;
let OrbitControls = null;

/* ==================================================================
   MODE MANAGER
   ================================================================== */
function setMode(mode) {
  state.mode = mode;

  dom.modeUpload.classList.toggle('hidden', mode !== 'upload');
  dom.modeLoading.classList.toggle('hidden', mode !== 'loading');
  dom.modeViewer.classList.toggle('hidden', mode !== 'viewer');

  if (mode === 'upload') {
    // Reset animation by re-triggering
    dom.modeUpload.style.animation = 'none';
    dom.modeUpload.offsetHeight; // force reflow
    dom.modeUpload.style.animation = '';
  }
}

/* ==================================================================
   FILE HANDLING
   ================================================================== */
function addFiles(newFiles) {
  const remaining = state.maxFiles - state.files.length;
  if (remaining <= 0) {
    Toast.show('最多上传 ' + state.maxFiles + ' 张照片');
    return;
  }

  const toAdd = Array.from(newFiles).slice(0, remaining);
  state.files.push(...toAdd);
  renderThumbnails();
  updatePhotoCount();
  updateStartButton();
}

function removeFile(index) {
  state.files.splice(index, 1);
  renderThumbnails();
  updatePhotoCount();
  updateStartButton();
}

function renderThumbnails() {
  dom.thumbGrid.innerHTML = '';

  state.files.forEach((file, index) => {
    const reader = new FileReader();
    reader.onload = function (e) {
      const item = document.createElement('div');
      item.className = 'thumb-item';
      item.style.animationDelay = (index * 0.05) + 's';

      const img = document.createElement('img');
      img.src = e.target.result;
      img.alt = 'Photo ' + (index + 1);

      const removeBtn = document.createElement('button');
      removeBtn.className = 'thumb-item__remove';
      removeBtn.innerHTML = '&#10005;';
      removeBtn.setAttribute('aria-label', '删除照片');
      removeBtn.addEventListener('click', function (ev) {
        ev.stopPropagation();
        removeFile(index);
      });

      item.appendChild(img);
      item.appendChild(removeBtn);
      dom.thumbGrid.appendChild(item);
    };
    reader.readAsDataURL(file);
  });
}

function updatePhotoCount() {
  dom.photoCount.textContent =
    '已选择 ' + state.files.length + '/' + state.maxFiles + ' 张照片';
}

function updateStartButton() {
  dom.btnStart.disabled = state.files.length === 0;
}

function triggerFileInput(capture) {
  if (capture) {
    dom.fileInput.setAttribute('capture', 'environment');
  } else {
    dom.fileInput.removeAttribute('capture');
  }
  dom.fileInput.click();
}

/* ==================================================================
   DROP ZONE DRAG & DROP
   ================================================================== */
function setupDropZone() {
  const dz = dom.dropZone;

  dz.addEventListener('click', function () {
    dom.fileInput.removeAttribute('capture');
    dom.fileInput.click();
  });

  // Prevent defaults for drag events
  ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(function (eventName) {
    dz.addEventListener(eventName, function (e) {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  dz.addEventListener('dragenter', function () {
    dz.classList.add('drop-zone--active');
  });

  dz.addEventListener('dragover', function () {
    dz.classList.add('drop-zone--active');
  });

  dz.addEventListener('dragleave', function () {
    dz.classList.remove('drop-zone--active');
  });

  dz.addEventListener('drop', function (e) {
    dz.classList.remove('drop-zone--active');
    var files = e.dataTransfer.files;
    if (files && files.length > 0) {
      addFiles(files);
    }
  });

  // File input change
  dom.fileInput.addEventListener('change', function () {
    if (dom.fileInput.files && dom.fileInput.files.length > 0) {
      addFiles(dom.fileInput.files);
    }
    dom.fileInput.value = '';
  });
}

/* ==================================================================
   UPLOAD & LOADING SIMULATION
   ================================================================== */
function startModeling() {
  if (state.files.length === 0) {
    Toast.show('请先添加照片');
    return;
  }

  // Switch to loading mode
  setMode('loading');
  dom.progressFill.style.width = '0%';

  // Simulate progress over 3 seconds
  var progress = 0;
  state.progressInterval = setInterval(function () {
    progress += Math.random() * 12 + 4; // 4-16% per tick
    if (progress > 90) progress = 90; // cap at 90% until "done"
    dom.progressFill.style.width = progress + '%';
  }, 250);

  // After 3 seconds, switch to viewer
  state.loadingTimer = setTimeout(function () {
    clearInterval(state.progressInterval);
    dom.progressFill.style.width = '100%';
    setTimeout(function () {
      setMode('viewer');
      initViewer();
    }, 400);
  }, 3000);

  // Attempt backend upload (fire-and-forget for now)
  uploadToBackend();
}

function cancelModeling() {
  if (state.loadingTimer) {
    clearTimeout(state.loadingTimer);
    state.loadingTimer = null;
  }
  if (state.progressInterval) {
    clearInterval(state.progressInterval);
    state.progressInterval = null;
  }
  setMode('upload');
  Toast.show('已取消建模');
}

async function uploadToBackend() {
  var formData = new FormData();
  state.files.forEach(function (file, i) {
    formData.append('photos', file, 'photo_' + i + '.jpg');
  });

  try {
    await API.upload('/api/model/generate', formData);
    // Backend integration point: handle response with model URL
  } catch (_err) {
    // Backend not yet implemented — silently ignore
    console.log('Backend /api/model/generate not available (expected)');
  }
}

/* ==================================================================
   THREE.JS VIEWER
   ================================================================== */
const viewerState = {
  renderer: null,
  scene: null,
  camera: null,
  controls: null,
  animationId: null,
  modelGroup: null,
};

async function initViewer() {
  // Load Three.js dynamically
  if (!THREE) {
    try {
      var threeModule = await import('three');
      THREE = threeModule;
      var ocModule = await import('three/addons/controls/OrbitControls.js');
      OrbitControls = ocModule.OrbitControls;
    } catch (err) {
      console.error('Failed to load Three.js:', err);
      Toast.show('3D 引擎加载失败');
      return;
    }
  }

  setupScene();
  startAnimationLoop();
}

function setupScene() {
  var container = dom.viewerCanvas;
  var width = container.clientWidth;
  var height = container.clientHeight;

  // ---- Renderer ----
  viewerState.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  viewerState.renderer.setSize(width, height);
  viewerState.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  viewerState.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  viewerState.renderer.toneMappingExposure = 1.2;
  viewerState.renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(viewerState.renderer.domElement);

  // ---- Scene ----
  viewerState.scene = new THREE.Scene();

  // ---- Camera ----
  viewerState.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
  viewerState.camera.position.set(4, 2.5, 5);
  viewerState.camera.lookAt(0, 0, 0);

  // ---- Lighting ----
  var ambientLight = new THREE.AmbientLight(0x404060, 1.8);
  viewerState.scene.add(ambientLight);

  var keyLight = new THREE.DirectionalLight(0xffffff, 3.0);
  keyLight.position.set(5, 8, 5);
  viewerState.scene.add(keyLight);

  var fillLight = new THREE.DirectionalLight(0x8ea4f0, 1.5);
  fillLight.position.set(-3, 2, -2);
  viewerState.scene.add(fillLight);

  var rimLight = new THREE.DirectionalLight(0xa0c4ff, 2.0);
  rimLight.position.set(0, -1, 4);
  viewerState.scene.add(rimLight);

  // ---- Grid Helper ----
  var gridHelper = new THREE.GridHelper(8, 20, 0x333355, 0x1a1a30);
  viewerState.scene.add(gridHelper);

  // ---- Model Group (contains the placeholder + future model) ----
  viewerState.modelGroup = new THREE.Group();
  viewerState.scene.add(viewerState.modelGroup);

  // ---- Placeholder: Torus Knot ----
  var knotGeom = new THREE.TorusKnotGeometry(1.0, 0.32, 128, 24, 2, 3);
  var knotMat = new THREE.MeshStandardMaterial({
    color: 0x667eea,
    metalness: 0.3,
    roughness: 0.4,
  });
  var knot = new THREE.Mesh(knotGeom, knotMat);
  knot.position.y = 0.6;
  viewerState.modelGroup.add(knot);

  // ---- Inner ring (adds visual interest) ----
  var ringGeom = new THREE.TorusGeometry(0.75, 0.08, 32, 64);
  var ringMat = new THREE.MeshStandardMaterial({
    color: 0x4facfe,
    metalness: 0.6,
    roughness: 0.25,
  });
  var ring = new THREE.Mesh(ringGeom, ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.position.y = -0.65;
  viewerState.modelGroup.add(ring);

  // ---- Small orbiting spheres ----
  var sphereGeom = new THREE.SphereGeometry(0.1, 16, 16);
  var sphereMat = new THREE.MeshStandardMaterial({
    color: 0xf093fb,
    metalness: 0.1,
    roughness: 0.3,
    emissive: 0x330033,
    emissiveIntensity: 0.5,
  });

  for (var i = 0; i < 8; i++) {
    var angle = (i / 8) * Math.PI * 2;
    var radius = 1.5;
    var sphere = new THREE.Mesh(sphereGeom, sphereMat);
    sphere.position.set(Math.cos(angle) * radius, 0.7, Math.sin(angle) * radius);
    viewerState.modelGroup.add(sphere);
  }

  // ---- OrbitControls ----
  viewerState.controls = new OrbitControls(viewerState.camera, viewerState.renderer.domElement);
  viewerState.controls.target.set(0, 0.1, 0);
  viewerState.controls.enableDamping = true;
  viewerState.controls.dampingFactor = 0.08;
  viewerState.controls.minDistance = 2.5;
  viewerState.controls.maxDistance = 10;
  viewerState.controls.maxPolarAngle = Math.PI * 0.75;
  viewerState.controls.autoRotate = true;
  viewerState.controls.autoRotateSpeed = 0.6;
  viewerState.controls.touches = {
    ONE: 0,  // TOUCH.ROTATE
    TWO: 1,  // TOUCH.DOLLY_PAN
  };
  viewerState.controls.update();
}

function startAnimationLoop() {
  function animate() {
    viewerState.animationId = requestAnimationFrame(animate);

    if (viewerState.controls) {
      viewerState.controls.update();
    }

    if (viewerState.modelGroup) {
      // Subtle floating rotation on inner ring spheres
      var time = Date.now() * 0.001;
      var spheres = viewerState.modelGroup.children;
      // Spheres are the last 8 children (after knot and ring)
      for (var i = 2; i < spheres.length; i++) {
        var s = spheres[i];
        s.position.y = 0.7 + Math.sin(time * 2 + i) * 0.15;
      }
    }

    if (viewerState.renderer && viewerState.scene && viewerState.camera) {
      viewerState.renderer.render(viewerState.scene, viewerState.camera);
    }
  }

  animate();
}

function stopViewer() {
  if (viewerState.animationId) {
    cancelAnimationFrame(viewerState.animationId);
    viewerState.animationId = null;
  }
  if (viewerState.renderer) {
    viewerState.renderer.dispose();
    if (viewerState.renderer.domElement.parentNode) {
      viewerState.renderer.domElement.parentNode.removeChild(viewerState.renderer.domElement);
    }
    viewerState.renderer = null;
  }
  if (viewerState.controls) {
    viewerState.controls.dispose();
    viewerState.controls = null;
  }
  viewerState.scene = null;
  viewerState.camera = null;
  viewerState.modelGroup = null;
}

function onWindowResize() {
  if (!viewerState.renderer || !viewerState.camera) return;

  var container = dom.viewerCanvas;
  var width = container.clientWidth;
  var height = container.clientHeight;

  viewerState.camera.aspect = width / height;
  viewerState.camera.updateProjectionMatrix();
  viewerState.renderer.setSize(width, height);
}

function resetToUpload() {
  stopViewer();
  state.files = [];
  renderThumbnails();
  updatePhotoCount();
  updateStartButton();
  setMode('upload');
}

/* ==================================================================
   INITIALIZATION
   ================================================================== */
function init() {
  // Navigation back button
  Page.init();

  // Drop zone & file handling
  setupDropZone();

  // Camera button
  dom.btnCamera.addEventListener('click', function () {
    triggerFileInput(true);
  });

  // Gallery button
  dom.btnGallery.addEventListener('click', function () {
    triggerFileInput(false);
  });

  // Start modeling
  dom.btnStart.addEventListener('click', startModeling);

  // Cancel loading
  dom.btnCancel.addEventListener('click', cancelModeling);

  // Remodel button
  dom.btnRemodel.addEventListener('click', resetToUpload);

  // Window resize
  window.addEventListener('resize', function () {
    onWindowResize();
  });

  // Handle orientation change on mobile
  window.addEventListener('orientationchange', function () {
    setTimeout(onWindowResize, 200);
  });

  // Prevent accidental page navigation via back gesture during viewer rotation
  dom.viewerCanvas.addEventListener('touchstart', function (e) {
    if (e.touches.length === 1 && state.mode === 'viewer') {
      // Allow single touch for rotation
    }
  }, { passive: true });
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
