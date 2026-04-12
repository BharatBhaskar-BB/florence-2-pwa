// ── BundleBox PWA — Live Detection + Gemini Inventory ──────────────────────

const BASE_PATH = window.location.pathname.replace(/\/+$/, '');
const API = window.location.origin + BASE_PATH;

// ── State ──────────────────────────────────────────────────────────────────
let scanning = false;
let paused = false;
let stream = null;
let frameCount = 0;
let startTime = 0;
let timerInterval = null;
let maxDuration = 60;
let detectedLabels = new Set();
let capturedFrames = []; // base64 frames for Gemini batch
const CAPTURE_INTERVAL = 2000; // capture a key frame every 2s for Gemini
let lastCaptureTime = 0;
let scanStats = { totalMs: 0, inferenceSum: 0, inferenceCount: 0 };
let mediaRecorder = null;
let recordedChunks = [];
let recordedBlob = null;
let lastInventoryResult = null;

const video = document.getElementById('s-video');
const canvas = document.getElementById('s-canvas');
const ctx = canvas.getContext('2d');
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');
const resizeCanvas = document.createElement('canvas');
const resizeCtx = resizeCanvas.getContext('2d');

// WebSocket state
let ws = null;
let wsSendInterval = null;
let wsFrameId = 0;
const frameStore = new Map(); // frame_id → ImageData (for synced mode)

// ── Colors ─────────────────────────────────────────────────────────────────
const COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FF8000', '#8000FF', '#00FF80', '#FF0080', '#0080FF', '#80FF00',
];

// ── Detection Filters ─────────────────────────────────────────────────────

// #2: Label normalization — merge over-granular labels into parent categories
const LABEL_MAP = {
    'door handle': 'door', 'hinge': null, 'door knob': 'door',
    'bottle cap': 'bottle', 'wine bottle': 'bottle',
    'computer keyboard': 'keyboard', 'computer mouse': 'mouse',
    'computer monitor': 'monitor', 'television': 'TV',
    'kitchen & dining room table': 'table', 'coffee table': 'table',
    'dining table': 'table', 'kitchen table': 'table',
    'power plugs and sockets': null, 'wall socket': null, 'light switch': null,
    'plastic bag': 'bag', 'handbag': 'bag',
    'flowerpot': 'plant pot', 'houseplant': 'plant',
    'couch': 'sofa', 'loveseat': 'sofa',
    'swivel chair': 'office chair', 'armchair': 'chair',
};

// #1: Temporal consistency — track label appearances over recent frames
const TEMPORAL_WINDOW = 3;
const TEMPORAL_MIN = 2;
let recentFrameLabels = []; // array of Sets, last N frames

function normalizeLabel(label) {
    const lower = label.toLowerCase();
    if (lower in LABEL_MAP) return LABEL_MAP[lower];
    return label;
}

function filterDetections(bboxes, labels, imgW, imgH) {
    const frameArea = imgW * imgH;
    const MIN_AREA_RATIO = 0.02; // #4: min 2% of frame

    const filtered = { bboxes: [], labels: [] };
    for (let i = 0; i < bboxes.length; i++) {
        const [x1, y1, x2, y2] = bboxes[i];

        // #4: Box size filter
        const boxArea = (x2 - x1) * (y2 - y1);
        if (boxArea / frameArea < MIN_AREA_RATIO) continue;

        // #2: Label normalization
        const normalized = normalizeLabel(labels[i]);
        if (normalized === null) continue; // explicitly removed

        filtered.bboxes.push(bboxes[i]);
        filtered.labels.push(normalized);
    }
    return filtered;
}

function updateTemporalFilter(frameLabels) {
    recentFrameLabels.push(new Set(frameLabels));
    if (recentFrameLabels.length > TEMPORAL_WINDOW) recentFrameLabels.shift();

    // Count appearances across recent frames
    const counts = new Map();
    for (const frameSet of recentFrameLabels) {
        for (const label of frameSet) {
            counts.set(label, (counts.get(label) || 0) + 1);
        }
    }

    // Only labels with >= TEMPORAL_MIN appearances go to ticker
    const stable = new Set();
    for (const [label, count] of counts) {
        if (count >= TEMPORAL_MIN) stable.add(label);
    }
    return stable;
}

// ── Init ───────────────────────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register(BASE_PATH + '/sw.js').catch(() => { });
}

// Check if onboarding was already completed
if (localStorage.getItem('bb_onboarded')) {
    document.getElementById('onboarding').classList.remove('active');
    document.getElementById('home').classList.add('active');
}
checkServer();
updateGreeting();

// ── Onboarding ─────────────────────────────────────────────────────────────
let _obPage = 1;
function nextOnboard() {
    if (_obPage >= 3) { finishOnboard(); return; }
    _obPage++;
    ['ob-page-1', 'ob-page-2', 'ob-page-3'].forEach((id, i) => {
        const p = i + 1;
        document.getElementById(id).className = 'ob-page ' + (p < _obPage ? 'left' : p === _obPage ? 'center' : 'right');
    });
    ['ob-dot-1', 'ob-dot-2', 'ob-dot-3'].forEach((id, i) => {
        document.getElementById(id).className = 'ob-dot' + (i + 1 === _obPage ? ' active' : '');
    });
    document.getElementById('ob-next-btn').textContent = _obPage === 3 ? 'GET STARTED' : 'NEXT';
}
function finishOnboard() {
    localStorage.setItem('bb_onboarded', '1');
    showScreen('home');
}

function updateGreeting() {
    const h = new Date().getHours();
    const g = h < 12 ? 'Good morning' : h < 17 ? 'Good afternoon' : 'Good evening';
    const el = document.getElementById('h-greeting');
    if (el) el.textContent = g + ' 👋';
}

async function checkServer() {
    const pill = document.getElementById('h-status-pill');
    const text = document.getElementById('h-status-text');
    const dot = pill ? pill.querySelector('.status-dot-sm') : null;
    try {
        const r = await fetch(API + '/api/health');
        const d = await r.json();
        if (pill) { pill.className = 'status-pill connected'; }
        if (dot) { dot.className = 'status-dot-sm on'; }
        if (text) { text.textContent = `Connected · ${d.device.toUpperCase()}`; }
        document.getElementById('h-start').disabled = false;

        // Populate available segmentors
        const segmentors = d.segmentors || ['none'];
        const segSelect = document.getElementById('h-segmentor');
        if (segSelect) {
            for (const opt of segSelect.options) {
                if (opt.value !== 'none') {
                    opt.disabled = !segmentors.includes(opt.value);
                    if (opt.disabled) opt.text += ' (not installed)';
                }
            }
        }
    } catch {
        if (pill) { pill.className = 'status-pill offline'; }
        if (dot) { dot.className = 'status-dot-sm off'; }
        if (text) { text.textContent = 'Server Offline'; }
        document.getElementById('h-start').disabled = true;
    }
}

// ── Screen navigation ──────────────────────────────────────────────────────
function showScreen(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
    hideSettings();
}

function goHome() {
    stopScan();
    showScreen('home');
    renderScanHistory();
}

// ── Settings Bottom Sheet ──────────────────────────────────────────────────
function showSettings() {
    document.getElementById('sheet-overlay').classList.add('active');
    document.getElementById('settings-sheet').classList.add('open');
}
function hideSettings() {
    const overlay = document.getElementById('sheet-overlay');
    const sheet = document.getElementById('settings-sheet');
    if (overlay) overlay.classList.remove('active');
    if (sheet) sheet.classList.remove('open');
}

// ── Start Scan ─────────────────────────────────────────────────────────────
async function startScan() {
    maxDuration = parseInt(document.getElementById('h-duration').value);
    renderMode = document.getElementById('h-render').value;
    lastDetection = null;
    detectedLabels.clear();
    recentFrameLabels = [];
    capturedFrames = [];
    frameCount = 0;
    scanStats = { totalMs: 0, inferenceSum: 0, inferenceCount: 0 };
    lastCaptureTime = Date.now() + 2000; // skip first 2s to avoid blank frames
    paused = false;
    document.getElementById('s-pause').textContent = '⏸ Pause';
    document.getElementById('s-detecting').textContent = 'Detecting...';
    _warmupDone = false;
    resetBubbles();
    hideSettings();

    try {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            alert('Camera requires HTTPS. Please access this site via https://scan.bundlebox.ai');
            return;
        }
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
        });
        video.srcObject = stream;
        await video.play();
        initCameraControls();
    } catch (e) {
        alert('Camera access denied: ' + e.message);
        return;
    }

    showScreen('scan');
    scanning = true;
    startTime = Date.now();
    if (renderMode === 'smooth') {
        startOverlaySync();
    }

    // Start recording video
    recordedChunks = [];
    recordedBlob = null;
    lastInventoryResult = null;
    try {
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp8' });
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
        mediaRecorder.onstop = () => { recordedBlob = new Blob(recordedChunks, { type: 'video/webm' }); };
        mediaRecorder.start(1000); // chunk every 1s
    } catch (e) {
        console.warn('MediaRecorder not supported:', e);
        mediaRecorder = null;
    }

    timerInterval = setInterval(updateTimer, 500);
    connectWebSocket();
}

function updateTimer() {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    document.getElementById('s-timer').textContent = `${mins}:${secs.toString().padStart(2, '0')}`;

    if (elapsed >= maxDuration) {
        finishScan();
    }
}

// ── Pause / Resume ─────────────────────────────────────────────────────────
function togglePause() {
    paused = !paused;
    document.getElementById('s-pause').textContent = paused ? '▶ Resume' : '⏸ Pause';
    // WS sending interval handles pause check automatically
}

// ── Detection via WebSocket ────────────────────────────────────────────────
// Two render modes:
//   "smooth"  — 30fps native <video> + transparent canvas overlay (boxes lag during pan)
//   "synced"  — coupled: canvas shows captured frame + boxes together (synced perfectly)

let lastDetection = null;
let renderMode = 'smooth';

function startOverlaySync() {
    // Only used in "smooth" mode — 30fps render loop for box overlay
    function renderLoop() {
        if (!scanning || renderMode !== 'smooth') return;
        const vw = video.videoWidth || 640;
        const vh = video.videoHeight || 480;
        if (canvas.width !== vw || canvas.height !== vh) {
            canvas.width = vw;
            canvas.height = vh;
        }

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        if (lastDetection) {
            const style = document.getElementById('h-style').value;
            if (lastDetection.masks) {
                drawMasks(ctx, lastDetection.masks, lastDetection.scaleX, lastDetection.scaleY);
            }
            drawDetections(ctx, lastDetection.bboxes, lastDetection.labels,
                lastDetection.scaleX, lastDetection.scaleY, style, lastDetection.masks);
        }

        requestAnimationFrame(renderLoop);
    }
    requestAnimationFrame(renderLoop);
}

function connectWebSocket() {
    // Show warmup overlay on connect (will hide once warmup completes)
    const warmupEl = document.getElementById('s-warmup');
    if (warmupEl) warmupEl.style.display = 'flex';

    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${location.host}${BASE_PATH}/ws/detect`;
    console.log('Connecting WebSocket:', wsUrl);

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log('WebSocket connected');
        // Start sending frames at ~100ms interval
        wsSendInterval = setInterval(sendFrame, 100);
    };

    ws.onmessage = (e) => {
        handleDetectionResult(JSON.parse(e.data));
    };

    ws.onclose = () => {
        console.log('WebSocket closed');
        clearInterval(wsSendInterval);
        wsSendInterval = null;
    };

    ws.onerror = (e) => {
        console.error('WebSocket error:', e);
    };
}

function disconnectWebSocket() {
    clearInterval(wsSendInterval);
    wsSendInterval = null;
    if (ws) {
        ws.close();
        ws = null;
    }
    frameStore.clear();
    wsFrameId = 0;
    // Hide warmup overlay
    const warmupEl = document.getElementById('s-warmup');
    if (warmupEl) { warmupEl.style.display = 'none'; warmupEl.classList.remove('ready'); }
}

// ── Warmup Overlay (torch.compile) ─────────────────────────────────────────
let _warmupDone = false;

function updateWarmupOverlay(count, total) {
    const overlay = document.getElementById('s-warmup');
    if (!overlay || _warmupDone) return;

    // Server didn't send warmup info (not CUDA / no torch.compile)
    if (count === undefined || total === undefined || total === 0) {
        overlay.style.display = 'none';
        _warmupDone = true;
        return;
    }

    if (count >= total) {
        // Warmup complete — show "Ready!" briefly then hide
        overlay.classList.add('ready');
        document.getElementById('s-warmup-title') ||
            (overlay.querySelector('.warmup-title').textContent = 'Ready!');
        overlay.querySelector('.warmup-title').textContent = 'Ready!';
        overlay.querySelector('.warmup-sub').textContent = 'Model optimized for maximum speed';
        overlay.querySelector('.warmup-step').textContent = '';
        const bar = document.getElementById('s-warmup-bar');
        if (bar) bar.style.width = '100%';

        setTimeout(() => {
            overlay.style.display = 'none';
            _warmupDone = true;
        }, 1200);
    } else {
        overlay.style.display = 'flex';
        overlay.classList.remove('ready');
        const step = document.getElementById('s-warmup-step');
        const bar = document.getElementById('s-warmup-bar');
        const sub = document.getElementById('s-warmup-sub');
        if (step) step.textContent = `${count} / ${total}`;
        if (bar) bar.style.width = `${(count / total) * 100}%`;
        if (sub) sub.textContent = count === 0
            ? 'Compiling for faster inference...'
            : `Compiling graph ${count} of ${total}...`;
    }
}

// ── Camera Controls (Zoom + Torch) ─────────────────────────────────────────

let torchOn = false;

function initCameraControls() {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    if (!track) return;

    const caps = track.getCapabilities ? track.getCapabilities() : {};

    // Zoom
    const zoomWrap = document.getElementById('s-zoom-wrap');
    const zoomSlider = document.getElementById('s-zoom');
    if (caps.zoom) {
        zoomWrap.style.display = 'flex';
        zoomSlider.min = caps.zoom.min;
        zoomSlider.max = Math.min(caps.zoom.max, 5); // cap at 5x
        zoomSlider.step = caps.zoom.step || 0.1;
        zoomSlider.value = caps.zoom.min;
        document.getElementById('s-zoom-label').textContent = `${parseFloat(caps.zoom.min).toFixed(1)}x`;
    } else {
        zoomWrap.style.display = 'none';
    }

    // Torch
    const torchBtn = document.getElementById('s-torch');
    if (caps.torch) {
        torchBtn.style.display = 'flex';
        torchBtn.classList.remove('active');
        torchOn = false;
    } else {
        torchBtn.style.display = 'none';
    }
}

function setZoom(val) {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    if (!track) return;
    const v = parseFloat(val);
    document.getElementById('s-zoom-label').textContent = `${v.toFixed(1)}x`;
    try {
        track.applyConstraints({ advanced: [{ zoom: v }] });
    } catch (e) {
        console.warn('Zoom not supported:', e);
    }
}

function toggleTorch() {
    if (!stream) return;
    const track = stream.getVideoTracks()[0];
    if (!track) return;
    torchOn = !torchOn;
    const btn = document.getElementById('s-torch');
    btn.classList.toggle('active', torchOn);
    try {
        track.applyConstraints({ advanced: [{ torch: torchOn }] });
    } catch (e) {
        console.warn('Torch not supported:', e);
        torchOn = false;
        btn.classList.remove('active');
    }
}

function sendFrame() {
    if (!scanning || paused || !ws || ws.readyState !== WebSocket.OPEN) return;

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Capture full-res to offscreen canvas
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    captureCtx.drawImage(video, 0, 0, vw, vh);

    const frameId = wsFrameId++;

    // In synced mode, store the full-res frame for later rendering
    if (renderMode === 'synced') {
        frameStore.set(frameId, captureCtx.getImageData(0, 0, vw, vh));
        // Keep only last 10 frames to limit memory
        if (frameStore.size > 10) {
            const oldest = frameStore.keys().next().value;
            frameStore.delete(oldest);
        }
    }

    // Resize for inference if needed
    const targetW = parseInt(document.getElementById('h-resolution').value);
    let sendCanvas = captureCanvas;
    if (targetW < vw) {
        const targetH = Math.round(targetW * vh / vw);
        resizeCanvas.width = targetW;
        resizeCanvas.height = targetH;
        resizeCtx.drawImage(captureCanvas, 0, 0, targetW, targetH);
        sendCanvas = resizeCanvas;
    }

    const base64 = sendCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];

    // Capture key frame for Gemini (always full-res)
    const now = Date.now();
    if (now - lastCaptureTime >= CAPTURE_INTERVAL) {
        capturedFrames.push(captureCanvas.toDataURL('image/jpeg', 0.7).split(',')[1]);
        lastCaptureTime = now;
    }

    const task = document.getElementById('h-task').value;
    const segmentor = document.getElementById('h-segmentor').value;
    ws.send(JSON.stringify({ frame_id: frameId, image: base64, task, segmentor }));
}

function handleDetectionResult(data) {
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Filter detections (normalize labels, remove noise/small boxes)
    const filtered = filterDetections(data.bboxes, data.labels, data.image_width, data.image_height);
    const bboxes = filtered.bboxes;
    const labels = filtered.labels;
    const masks = data.masks || null;

    // Handle warmup overlay (torch.compile)
    updateWarmupOverlay(data.warmup, data.warmup_total);

    if (renderMode === 'synced') {
        // Retrieve the stored frame that matches this result
        const stored = frameStore.get(data.frame_id);
        if (stored) {
            canvas.width = stored.width;
            canvas.height = stored.height;
            ctx.putImageData(stored, 0, 0);
            const scaleX = stored.width / data.image_width;
            const scaleY = stored.height / data.image_height;
            const style = document.getElementById('h-style').value;
            if (masks) drawMasks(ctx, masks, scaleX, scaleY);
            drawDetections(ctx, bboxes, labels, scaleX, scaleY, style, masks);
        }
        // Clean up frames older than this result
        for (const key of frameStore.keys()) {
            if (key <= data.frame_id) frameStore.delete(key);
        }
    } else {
        // Smooth: store result for the 30fps renderLoop
        lastDetection = {
            bboxes,
            labels,
            masks,
            scaleX: vw / data.image_width,
            scaleY: vh / data.image_height,
            timestamp: Date.now(),
        };
    }

    frameCount++;
    const totalInfMs = data.time_ms + (data.seg_time_ms || 0);
    scanStats.inferenceSum += totalInfMs;
    scanStats.inferenceCount++;

    const badge = document.getElementById('s-inference-ms');
    if (badge) {
        let text = `⚡ ${Math.round(data.time_ms)}ms`;
        if (data.seg_time_ms) text += ` +${Math.round(data.seg_time_ms)}ms seg`;
        badge.textContent = text;
    }

    detectedLabels = updateTemporalFilter(labels);
    updateTicker();
    document.getElementById('s-detecting').textContent = `${detectedLabels.size} items`;
}

function drawMasks(ctx, masks, scaleX, scaleY) {
    for (let i = 0; i < masks.length; i++) {
        const poly = masks[i];
        if (!poly || poly.length < 3) continue;

        const color = COLORS[i % COLORS.length];
        const r = parseInt(color.slice(1, 3), 16);
        const g = parseInt(color.slice(3, 5), 16);
        const b = parseInt(color.slice(5, 7), 16);

        // Filled mask
        ctx.fillStyle = `rgba(${r},${g},${b},0.3)`;
        ctx.beginPath();
        ctx.moveTo(poly[0][0] * scaleX, poly[0][1] * scaleY);
        for (let j = 1; j < poly.length; j++) {
            ctx.lineTo(poly[j][0] * scaleX, poly[j][1] * scaleY);
        }
        ctx.closePath();
        ctx.fill();

        // Contour outline
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.stroke();
    }
}

function drawDetections(ctx, bboxes, labels, scaleX, scaleY, style, masks) {
    ctx.font = 'bold 12px -apple-system, sans-serif';
    for (let i = 0; i < bboxes.length; i++) {
        const [x1, y1, x2, y2] = bboxes[i];
        const color = COLORS[i % COLORS.length];
        const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
        const w = (x2 - x1) * scaleX, h = (y2 - y1) * scaleY;

        // Draw box unless masks-only mode
        if (style !== 'masks') {
            if (style === 'filled') {
                const r = parseInt(color.slice(1, 3), 16), g = parseInt(color.slice(3, 5), 16), b = parseInt(color.slice(5, 7), 16);
                ctx.fillStyle = `rgba(${r},${g},${b},0.25)`;
                ctx.fillRect(sx1, sy1, w, h);
            }
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(sx1, sy1, w, h);
        }

        // Always draw label — use mask centroid when in masks-only mode with masks
        const label = labels[i] || '';
        if (label) {
            let lx = sx1, ly = sy1;
            const poly = masks && masks[i];
            if (style === 'masks' && poly && poly.length >= 3) {
                // Compute centroid of mask polygon
                let cx = 0, cy = 0;
                for (const pt of poly) { cx += pt[0]; cy += pt[1]; }
                cx = (cx / poly.length) * scaleX;
                cy = (cy / poly.length) * scaleY;
                lx = cx;
                ly = cy;
            }
            const tw = ctx.measureText(label).width + 6;
            ctx.fillStyle = color;
            ctx.fillRect(lx - tw / 2, ly - 8, tw, 16);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, lx - tw / 2 + 3, ly + 5);
        }
    }
}

// ── Floating Bubbles ───────────────────────────────────────────────────────
let _shownBubbles = new Set(); // labels already shown as bubbles
let _bubbleQueue = [];         // pending labels to show
let _bubbleTimer = null;

function updateTicker() {
    // Queue new labels as bubbles
    for (const label of detectedLabels) {
        if (!_shownBubbles.has(label)) {
            _shownBubbles.add(label);
            _bubbleQueue.push(label);
        }
    }
    // Stagger bubble creation
    if (!_bubbleTimer && _bubbleQueue.length > 0) {
        _bubbleTimer = setInterval(() => {
            if (_bubbleQueue.length === 0) {
                clearInterval(_bubbleTimer);
                _bubbleTimer = null;
                return;
            }
            spawnBubble(_bubbleQueue.shift());
        }, 300);
    }
    // Update count badge
    document.getElementById('s-detecting').textContent = `${detectedLabels.size} items`;
}

function spawnBubble(label) {
    const container = document.getElementById('s-bubbles');
    if (!container) return;
    const el = document.createElement('div');
    el.className = 'bubble';
    el.textContent = label;
    // Random horizontal jitter
    el.style.left = `${Math.random() * 40}px`;
    container.appendChild(el);
    // Remove after animation completes
    el.addEventListener('animationend', () => el.remove());
}

function resetBubbles() {
    _shownBubbles.clear();
    _bubbleQueue = [];
    if (_bubbleTimer) { clearInterval(_bubbleTimer); _bubbleTimer = null; }
    const container = document.getElementById('s-bubbles');
    if (container) container.innerHTML = '';
}

// ── Finish Scan ────────────────────────────────────────────────────────────
function finishScan() {
    scanning = false;
    clearInterval(timerInterval);
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    scanStats.totalMs = elapsed * 1000;

    // Stop scanning UI
    const scanline = document.getElementById('s-scanline');
    if (scanline) scanline.classList.remove('active');
    lastDetection = null;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Disconnect WebSocket
    disconnectWebSocket();

    // Stop camera
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;

    // Show results screen
    showScreen('results');
    const mins = Math.floor(elapsed / 60);
    const secs = elapsed % 60;
    const durationStr = `${mins}:${secs.toString().padStart(2, '0')}`;
    const avgMs = scanStats.inferenceCount > 0 ? Math.round(scanStats.inferenceSum / scanStats.inferenceCount) : 0;

    // Populate stats bar
    document.getElementById('r-item-count').textContent = '—';
    document.getElementById('r-duration').textContent = durationStr;
    document.getElementById('r-avg-ms').textContent = avgMs ? `${avgMs}ms` : '—';

    // Show processing state
    document.getElementById('r-processing').style.display = 'block';
    document.getElementById('r-inventory').style.display = 'none';
    document.getElementById('r-total').style.display = 'none';

    const fps = avgMs > 0 ? (1000 / avgMs).toFixed(1) : '—';
    document.getElementById('r-stats').innerHTML =
        `Florence-2 Live Detection<br>` +
        `Avg inference: ${avgMs}ms/frame | FPS: ${fps}<br>` +
        `Frames detected: ${frameCount} | Key frames: ${capturedFrames.length}<br>` +
        `Live labels found: ${detectedLabels.size} types`;

    // Store scan in history
    _currentScanMeta = {
        duration: durationStr,
        frameCount,
        avgMs,
        labelCount: detectedLabels.size,
        timestamp: Date.now(),
    };

    // Send key frames to Gemini for inventory
    runGeminiBatch();
}

function stopScan() {
    scanning = false;
    paused = false;
    clearInterval(timerInterval);
    disconnectWebSocket();
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
    }
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.srcObject = null;
}

// ── Gemini Batch Inventory ─────────────────────────────────────────────────
async function runGeminiBatch() {
    if (capturedFrames.length === 0) {
        showInventoryError('No frames captured. Try scanning longer.');
        return;
    }

    // Send max 15 evenly-spaced frames
    let frames = capturedFrames;
    if (frames.length > 15) {
        const step = frames.length / 15;
        frames = Array.from({ length: 15 }, (_, i) => capturedFrames[Math.floor(i * step)]);
    }

    try {
        const res = await fetch(API + '/api/inventory', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frames }),
        });

        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Server error ${res.status}`);
        }

        const data = await res.json();
        lastInventoryResult = data;
        showInventory(data);
    } catch (e) {
        showInventoryError(e.message);
    }
}

function showInventory(data) {
    document.getElementById('r-processing').style.display = 'none';
    const invEl = document.getElementById('r-inventory');
    invEl.style.display = 'block';

    const items = data.items || [];
    let totalCount = 0;

    invEl.innerHTML = items.map(item => {
        const count = item.count || 1;
        totalCount += count;
        const detail = [item.size, item.notes].filter(Boolean).join(' · ');
        return `<div class="inv-card">
      <div class="ic-icon">📦</div>
      <div class="ic-info">
        <div class="ic-name">${item.name}</div>
        ${detail ? `<div class="ic-detail">${detail}</div>` : ''}
      </div>
      <div class="ic-qty">×${count}</div>
    </div>`;
    }).join('');

    const totalEl = document.getElementById('r-total');
    totalEl.style.display = 'block';
    totalEl.textContent = `Total: ${items.length} unique items, ${totalCount} pieces`;

    // Update stats bar item count
    document.getElementById('r-item-count').textContent = totalCount;

    // Update stats with Gemini info
    const statsEl = document.getElementById('r-stats');
    statsEl.innerHTML += `<br><br>Gemini 2.0 Flash Inventory<br>` +
        `Items found: ${items.length} | Total pieces: ${totalCount}<br>` +
        `Cost: ~$${(data.cost || 0).toFixed(3)}`;

    // Save to scan history
    saveScanHistory(totalCount, items.length);
}

function showInventoryError(msg) {
    document.getElementById('r-processing').style.display = 'none';
    const invEl = document.getElementById('r-inventory');
    invEl.style.display = 'block';
    invEl.innerHTML = `<div style="padding:20px;text-align:center;color:#ff4757">
    <div style="font-size:16px;font-weight:600">Inventory generation failed</div>
    <div style="font-size:13px;margin-top:8px;color:#888">${msg}</div>
    <div style="font-size:12px;margin-top:12px;color:#666">Live detection found ${detectedLabels.size} item types: ${[...detectedLabels].sort().join(', ')}</div>
  </div>`;
}

// ── Share ───────────────────────────────────────────────────────────────────
function shareResults() {
    const text = document.getElementById('r-inventory').innerText + '\n' + document.getElementById('r-total').innerText;
    if (navigator.share) {
        navigator.share({ title: 'BundleBox Scan Results', text }).catch(() => { });
    } else {
        navigator.clipboard.writeText(text).then(() => alert('Copied to clipboard'));
    }
}

// ── Upload Artifacts ────────────────────────────────────────────────────────
async function uploadArtifacts() {
    const btn = document.getElementById('r-upload-btn');
    const origText = btn.textContent;
    btn.textContent = '⏳ Uploading...';
    btn.disabled = true;

    try {
        const formData = new FormData();

        // 1. Video recording
        if (recordedBlob) {
            formData.append('video', recordedBlob, 'scan.webm');
        }

        // 2. Key frames sent to Gemini
        let frames = capturedFrames;
        if (frames.length > 15) {
            const step = frames.length / 15;
            frames = Array.from({ length: 15 }, (_, i) => capturedFrames[Math.floor(i * step)]);
        }
        for (let i = 0; i < frames.length; i++) {
            const blob = await fetch('data:image/jpeg;base64,' + frames[i]).then(r => r.blob());
            formData.append('frames', blob, `frame_${i.toString().padStart(3, '0')}.jpg`);
        }

        // 3. Inventory result + stats
        const meta = {
            inventory: lastInventoryResult,
            stats: {
                scanDuration: scanStats.totalMs,
                framesProcessed: frameCount,
                keyFramesCaptured: capturedFrames.length,
                keyFramesSent: frames.length,
                avgInferenceMs: scanStats.inferenceCount > 0 ? Math.round(scanStats.inferenceSum / scanStats.inferenceCount) : 0,
                liveLabels: [...detectedLabels].sort(),
            },
            timestamp: new Date().toISOString(),
        };
        formData.append('metadata', JSON.stringify(meta));

        const res = await fetch(API + '/api/upload-artifacts', {
            method: 'POST',
            body: formData,
        });

        if (!res.ok) throw new Error(`Server error ${res.status}`);
        const result = await res.json();
        btn.textContent = '✅ Uploaded!';
        btn.style.background = '#16a34a';
        btn.style.color = '#fff';
        alert(`Artifacts saved to: ${result.path}`);
    } catch (e) {
        btn.textContent = '❌ Failed';
        alert('Upload failed: ' + e.message);
    } finally {
        setTimeout(() => {
            btn.textContent = origText;
            btn.disabled = false;
            btn.style.background = '';
            btn.style.color = '';
        }, 3000);
    }
}

// ── Scan History ────────────────────────────────────────────────────────────
let _currentScanMeta = null;

function saveScanHistory(totalPieces, uniqueItems) {
    const history = JSON.parse(localStorage.getItem('bb_history') || '[]');
    history.unshift({
        items: totalPieces,
        unique: uniqueItems,
        duration: _currentScanMeta?.duration || '—',
        avgMs: _currentScanMeta?.avgMs || 0,
        timestamp: Date.now(),
    });
    // Keep last 20
    if (history.length > 20) history.length = 20;
    localStorage.setItem('bb_history', JSON.stringify(history));
}

function renderScanHistory() {
    const history = JSON.parse(localStorage.getItem('bb_history') || '[]');

    // Home recent scans
    const recentLabel = document.getElementById('h-recent-label');
    const recentList = document.getElementById('h-recent-list');
    if (recentLabel && recentList) {
        if (history.length > 0) {
            recentLabel.style.display = '';
            recentList.innerHTML = history.slice(0, 3).map(s => {
                const ago = timeAgo(s.timestamp);
                return `<div class="recent-scan">
                    <div class="recent-thumb">📦</div>
                    <div class="recent-info">
                        <div class="ri-title">${s.unique} items · ${s.items} pieces</div>
                        <div class="ri-meta">${s.duration} scan · ${ago}</div>
                    </div>
                    <div class="recent-arrow">›</div>
                </div>`;
            }).join('');
        } else {
            recentLabel.style.display = 'none';
            recentList.innerHTML = '';
        }
    }

    // History screen
    const histList = document.getElementById('hist-list');
    if (histList) {
        if (history.length > 0) {
            histList.innerHTML = history.map(s => {
                const ago = timeAgo(s.timestamp);
                return `<div class="history-card">
                    <div class="hc-top">
                        <div class="hc-room">📦 Room Scan</div>
                        <div class="hc-date">${ago}</div>
                    </div>
                    <div class="hc-stats">
                        <div class="hc-stat"><strong>${s.items}</strong> pieces</div>
                        <div class="hc-stat"><strong>${s.duration}</strong> duration</div>
                        <div class="hc-stat"><strong>${s.avgMs}ms</strong> avg</div>
                    </div>
                </div>`;
            }).join('');
        } else {
            histList.innerHTML = `<div class="history-empty"><div class="he-icon">📋</div><p>No scans yet.<br>Start your first room scan!</p></div>`;
        }
    }
}

function timeAgo(ts) {
    const diff = Date.now() - ts;
    const mins = Math.floor(diff / 60000);
    if (mins < 1) return 'Just now';
    if (mins < 60) return `${mins}m ago`;
    const hrs = Math.floor(mins / 60);
    if (hrs < 24) return `${hrs}h ago`;
    const days = Math.floor(hrs / 24);
    return `${days}d ago`;
}

// Render history on load
renderScanHistory();
