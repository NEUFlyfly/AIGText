/**
 * AIGText — AR Interaction Frontend
 *
 * Simulated AR: rear camera feed as background + Three.js transparent
 * renderer on top with a 3D model responding to device orientation.
 * Uses shared Camera, Nav, Toast from common.js.
 * Three.js loaded via CDN script tag (global THREE).
 */

/* ==================================================================
   DOM REFERENCES
   ================================================================== */
const dom = {
  video:      $('#ar-video'),
  canvas:     $('#ar-canvas'),
  backBtn:    $('#ar-back'),
  infoBtn:    $('#ar-info'),
  hud:        $('#ar-hud'),
  fpsEl:      $('#ar-fps'),
  gyroStatus: $('#ar-gyro-status'),
  scaleVal:   $('#ar-scale-val'),
  placeholder:$('#ar-placeholder'),
  pulse:      $('#ar-pulse'),
  btnPlace:   $('#btn-place'),
  scaleSlider:$('#scale-slider'),
  autoRotate: $('#auto-rotate'),
  btnScreenshot: $('#btn-screenshot'),
  permission: $('#ar-permission'),
  btnPermission: $('#btn-permission'),
};

/* ==================================================================
   STATE
   ================================================================== */
const state = {
  modelPlaced: false,
  stream: null,
  gyroAvailable: false,
  gyroGranted: false,
  gyroAlpha: 0,
  gyroBeta: 0,
  gyroGamma: 0,
  targetAlpha: 0,
  targetBeta: 0,
  targetGamma: 0,
  autoRotate: false,
  scale: 1.0,
  frameCount: 0,
  lastFpsTime: performance.now(),
  fps: 0,
};

/* ==================================================================
   THREE.JS GLOBALS (from CDN)
   ================================================================== */
let scene, camera, renderer, model, modelGroup;
let dirLight, ambientLight;

/* ==================================================================
   INITIALIZATION
   ================================================================== */
async function init() {
  // Back button
  dom.backBtn.addEventListener('click', () => Nav.back());

  // Info button toggles HUD visibility
  dom.infoBtn.addEventListener('click', () => {
    dom.hud.classList.toggle('hidden');
    dom.infoBtn.classList.toggle('ar-info--active');
  });

  // Open rear camera
  try {
    state.stream = await Camera.open(dom.video, 'environment');
  } catch (err) {
    Toast.show('无法访问摄像头，请检查权限');
    console.error('Camera open error:', err);
  }

  // Setup Three.js
  setupThree();

  // Setup gyroscope
  setupGyro();

  // Setup controls
  setupControls();

  // Start render loop
  requestAnimationFrame(animate);
}

/* ==================================================================
   THREE.JS SCENE SETUP
   ================================================================== */
function setupThree() {
  const w = window.innerWidth;
  const h = window.innerHeight;

  // Scene with transparent background
  scene = new THREE.Scene();

  // PerspectiveCamera matching phone FOV
  camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 100);
  camera.position.set(0, 0, 3);

  // Renderer with alpha channel
  renderer = new THREE.WebGLRenderer({ canvas: dom.canvas, alpha: true, antialias: true });
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);

  // Lighting
  dirLight = new THREE.DirectionalLight(0xffffff, 2.5);
  dirLight.position.set(3, 5, 4);
  scene.add(dirLight);

  ambientLight = new THREE.AmbientLight(0x4facfe, 1.2);
  scene.add(ambientLight);

  // Secondary fill light from below (reduces harsh shadows)
  const fillLight = new THREE.DirectionalLight(0x667eea, 1.0);
  fillLight.position.set(-2, -2, 1);
  scene.add(fillLight);

  // Group to hold model (so we can rotate group vs. model independently)
  modelGroup = new THREE.Group();
  modelGroup.visible = false;
  modelGroup.scale.set(0.01, 0.01, 0.01); // start tiny for entrance animation
  scene.add(modelGroup);

  // Build the placeholder 3D model
  buildModel();

  // Handle resize
  window.addEventListener('resize', onResize);
  window.addEventListener('orientationchange', () => {
    setTimeout(onResize, 300);
  });
}

/* ==================================================================
   BUILD PLACEHOLDER 3D MODEL
   ================================================================== */
function buildModel() {
  // Create a visually striking model: torus knot + surrounding elements
  model = new THREE.Group();

  // -- Core: metallic torus knot --
  const knotGeo = new THREE.TorusKnotGeometry(0.5, 0.15, 128, 32, 2, 3);
  const knotMat = new THREE.MeshStandardMaterial({
    color: 0x4facfe,
    metalness: 0.85,
    roughness: 0.25,
    envMapIntensity: 0.5,
  });
  const knot = new THREE.Mesh(knotGeo, knotMat);
  knot.name = 'knot';
  model.add(knot);

  // -- Inner ring: thin torus --
  const ringGeo = new THREE.TorusGeometry(0.52, 0.03, 32, 64);
  const ringMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 0.9,
    roughness: 0.15,
    emissive: 0x4facfe,
    emissiveIntensity: 0.3,
  });
  const ring = new THREE.Mesh(ringGeo, ringMat);
  ring.rotation.x = Math.PI / 2;
  ring.name = 'ringH';
  model.add(ring);

  // -- Vertical ring --
  const ringV = new THREE.Mesh(ringGeo.clone(), ringMat);
  ringV.rotation.y = Math.PI / 2;
  ringV.name = 'ringV';
  model.add(ringV);

  // -- Orbiting particles (small spheres around the knot) --
  const particleGeo = new THREE.SphereGeometry(0.04, 8, 8);
  const particleMat = new THREE.MeshStandardMaterial({
    color: 0xffffff,
    metalness: 0.3,
    roughness: 0.4,
    emissive: 0x4facfe,
    emissiveIntensity: 0.6,
  });
  const particleCount = 8;
  for (let i = 0; i < particleCount; i++) {
    const angle = (i / particleCount) * Math.PI * 2;
    const radius = 0.58;
    const particle = new THREE.Mesh(particleGeo, particleMat);
    particle.position.set(
      Math.cos(angle) * radius,
      Math.sin(angle) * radius,
      0
    );
    particle.name = 'particle' + i;
    model.add(particle);
  }

  // -- Wireframe outer shell --
  const wireGeo = new THREE.IcosahedronGeometry(0.68, 1);
  const wireMat = new THREE.MeshBasicMaterial({
    color: 0x4facfe,
    wireframe: true,
    transparent: true,
    opacity: 0.25,
  });
  const wireframe = new THREE.Mesh(wireGeo, wireMat);
  wireframe.name = 'wireframe';
  model.add(wireframe);

  modelGroup.add(model);
}

/* ==================================================================
   GYROSCOPE (DEVICE ORIENTATION)
   ================================================================== */
function setupGyro() {
  // Check if DeviceOrientationEvent exists
  if (!window.DeviceOrientationEvent) {
    state.gyroAvailable = false;
    dom.gyroStatus.textContent = 'OFF';
    dom.gyroStatus.style.color = 'var(--ar-danger)';
    return;
  }

  // iOS 13+ requires explicit permission
  if (typeof DeviceOrientationEvent.requestPermission === 'function') {
    // Show permission overlay
    dom.permission.classList.remove('hidden');

    dom.btnPermission.addEventListener('click', async () => {
      try {
        const permission = await DeviceOrientationEvent.requestPermission();
        if (permission === 'granted') {
          state.gyroGranted = true;
          state.gyroAvailable = true;
          dom.permission.classList.add('hidden');
          dom.gyroStatus.textContent = 'ON';
          dom.gyroStatus.style.color = 'var(--accent-green)';
          window.addEventListener('deviceorientation', onDeviceOrientation);
          Toast.show('陀螺仪已启用');
        } else {
          dom.permission.classList.add('hidden');
          dom.gyroStatus.textContent = 'DENY';
          dom.gyroStatus.style.color = 'var(--ar-danger)';
        }
      } catch (err) {
        dom.permission.classList.add('hidden');
        dom.gyroStatus.textContent = 'ERR';
        dom.gyroStatus.style.color = 'var(--ar-danger)';
        console.error('Gyro permission error:', err);
      }
    });
  } else {
    // Non-iOS: listen directly
    state.gyroAvailable = true;
    state.gyroGranted = true;
    dom.gyroStatus.textContent = 'ON';
    dom.gyroStatus.style.color = 'var(--accent-green)';
    window.addEventListener('deviceorientation', onDeviceOrientation);
  }
}

var DEG2RAD = Math.PI / 180;

function onDeviceOrientation(event) {
  if (event.alpha !== null) {
    state.targetAlpha = event.alpha * DEG2RAD;   // Y rotation (compass)
    state.targetBeta  = event.beta  * DEG2RAD;   // X rotation (tilt forward/back)
    state.targetGamma = event.gamma * DEG2RAD;   // Z rotation (tilt left/right)
  }
}

/* ==================================================================
   CONTROLS
   ================================================================== */
function setupControls() {
  // Place / Remove model
  dom.btnPlace.addEventListener('click', toggleModel);

  // Scale slider
  dom.scaleSlider.addEventListener('input', () => {
    state.scale = parseInt(dom.scaleSlider.value) / 100;
    if (state.modelPlaced) {
      modelGroup.scale.setScalar(state.scale);
    }
    dom.scaleVal.textContent = Math.round(state.scale * 100) + '%';
  });

  // Auto-rotate toggle
  dom.autoRotate.addEventListener('change', () => {
    state.autoRotate = dom.autoRotate.checked;
  });

  // Screenshot
  dom.btnScreenshot.addEventListener('click', takeScreenshot);
}

function toggleModel() {
  if (!state.modelPlaced) {
    placeModel();
  } else {
    removeModel();
  }
}

function placeModel() {
  state.modelPlaced = true;
  modelGroup.visible = true;

  // Animate in: scale from 0.01 to target scale
  const targetScale = state.scale;
  const startTime = performance.now();
  const duration = 600;

  function animateIn(now) {
    const elapsed = now - startTime;
    const t = Math.min(elapsed / duration, 1.0);
    // Ease-out cubic
    const eased = 1 - Math.pow(1 - t, 3);
    const s = 0.01 + (targetScale - 0.01) * eased;
    modelGroup.scale.setScalar(s);

    if (t < 1) {
      requestAnimationFrame(animateIn);
    } else {
      modelGroup.scale.setScalar(targetScale);
    }
  }

  requestAnimationFrame(animateIn);

  // Show pulse ring
  dom.pulse.classList.remove('hidden');
  dom.pulse.style.animation = 'none';
  dom.pulse.offsetHeight; // force reflow
  dom.pulse.style.animation = 'arPulseRing 1.2s var(--ease-out) forwards';
  setTimeout(() => dom.pulse.classList.add('hidden'), 1300);

  // Hide placeholder
  dom.placeholder.classList.add('ar-placeholder--hidden');

  // Update button
  dom.btnPlace.textContent = '移除模型';
  dom.btnPlace.classList.add('ar-controls__btn--placed');

  Toast.show('模型已放置');
}

function removeModel() {
  state.modelPlaced = false;
  modelGroup.visible = false;

  // Show placeholder
  dom.placeholder.classList.remove('ar-placeholder--hidden');
  dom.placeholder.style.animation = '';
  dom.placeholder.offsetHeight;
  dom.placeholder.style.animation = 'arPlaceholderPulse 3s var(--ease-in-out) infinite';

  // Update button
  dom.btnPlace.textContent = '放置模型';
  dom.btnPlace.classList.remove('ar-controls__btn--placed');

  Toast.show('模型已移除');
}

/* ==================================================================
   SCREENSHOT (composite video + canvas)
   ================================================================== */
function takeScreenshot() {
  if (!state.stream) {
    Toast.show('摄像头未就绪');
    return;
  }

  // Flash effect
  const flash = document.createElement('div');
  flash.className = 'ar-flash';
  document.body.appendChild(flash);
  setTimeout(() => flash.remove(), 450);

  const vw = dom.video.videoWidth || window.innerWidth;
  const vh = dom.video.videoHeight || window.innerHeight;

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = vw;
  tempCanvas.height = vh;
  const ctx = tempCanvas.getContext('2d');

  // Draw video frame (fills canvas)
  ctx.drawImage(dom.video, 0, 0, tempCanvas.width, tempCanvas.height);

  // Draw Three.js canvas on top (transparent background, so model overlays)
  ctx.drawImage(dom.canvas, 0, 0, tempCanvas.width, tempCanvas.height);

  // Trigger download
  tempCanvas.toBlob((blob) => {
    if (!blob) {
      Toast.show('截图失败');
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'AR_' + new Date().toISOString().replace(/[:.]/g, '-') + '.jpg';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    Toast.show('截图已保存');
  }, 'image/jpeg', 0.9);
}

/* ==================================================================
   RESIZE HANDLER
   ================================================================== */
function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;

  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
}

/* ==================================================================
   ANIMATION LOOP
   ================================================================== */
function animate(timestamp) {
  requestAnimationFrame(animate);

  // FPS counter
  state.frameCount++;
  if (timestamp - state.lastFpsTime >= 1000) {
    state.fps = Math.round(state.frameCount / ((timestamp - state.lastFpsTime) / 1000));
    state.frameCount = 0;
    state.lastFpsTime = timestamp;
    dom.fpsEl.textContent = state.fps;
  }

  // Smooth gyro → scene rotation (lerp)
  const lerpFactor = 0.08;
  if (state.gyroAvailable && state.gyroGranted) {
    state.gyroAlpha += (state.targetAlpha - state.gyroAlpha) * lerpFactor;
    state.gyroBeta  += (state.targetBeta  - state.gyroBeta)  * lerpFactor;
    state.gyroGamma += (state.targetGamma - state.gyroGamma) * lerpFactor;
  }

  // Apply gyro rotation to model group
  // Beta (forward/back tilt) → X rotation
  // Gamma (left/right tilt) → Z rotation
  // Alpha (compass) → Y rotation
  if (state.gyroAvailable && state.gyroGranted) {
    modelGroup.rotation.x = state.gyroBeta;
    modelGroup.rotation.z = state.gyroGamma;
    // Y axis: alpha rotated 90° to align with typical phone orientation
    modelGroup.rotation.y = state.gyroAlpha + Math.PI / 2;
  }

  // Auto-rotate: spin model on Y axis
  if (state.autoRotate && state.modelPlaced) {
    model.rotation.y += 0.01;
  }

  // Animate orbiting particles
  if (state.modelPlaced && model) {
    const t = performance.now() * 0.001;
    const particles = model.children.filter(c => c.name && c.name.startsWith('particle'));
    particles.forEach((p, i) => {
      const angle = t * 1.2 + (i / particles.length) * Math.PI * 2;
      const radius = 0.58;
      p.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, 0);
    });
  }

  // Render
  renderer.render(scene, camera);
}

/* ==================================================================
   BOOT
   ================================================================== */
document.addEventListener('DOMContentLoaded', init);
