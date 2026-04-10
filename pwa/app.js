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
checkServer();

async function checkServer() {
    try {
        const r = await fetch(API + '/api/health');
        const d = await r.json();
        document.getElementById('h-server-status').innerHTML =
            `<span class="status-dot ok"></span>Connected`;
        document.getElementById('h-device').textContent = d.device;
        document.getElementById('h-start').disabled = false;
    } catch {
        document.getElementById('h-server-status').innerHTML =
            `<span class="status-dot err"></span>Offline`;
        document.getElementById('h-start').disabled = true;
    }
}

// ── Screen navigation ──────────────────────────────────────────────────────
function showScreen(id) {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    document.getElementById(id).classList.add('active');
}

function goHome() {
    stopScan();
    showScreen('home');
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
    document.getElementById('s-ticker').innerHTML = '';
    document.getElementById('s-detecting').textContent = 'Detecting...';

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
            drawDetections(ctx, lastDetection.bboxes, lastDetection.labels,
                lastDetection.scaleX, lastDetection.scaleY, style);
        }

        requestAnimationFrame(renderLoop);
    }
    requestAnimationFrame(renderLoop);
}

function connectWebSocket() {
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
    ws.send(JSON.stringify({ frame_id: frameId, image: base64, task }));
}

function handleDetectionResult(data) {
    const bboxes = data.bboxes;
    const labels = data.labels;
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

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
            drawDetections(ctx, bboxes, labels, scaleX, scaleY, style);
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
            scaleX: vw / data.image_width,
            scaleY: vh / data.image_height,
            timestamp: Date.now(),
        };
    }

    frameCount++;
    scanStats.inferenceSum += data.time_ms;
    scanStats.inferenceCount++;

    const badge = document.getElementById('s-inference-ms');
    if (badge) badge.textContent = `⚡ ${Math.round(data.time_ms)}ms`;

    detectedLabels = new Set(labels);
    updateTicker();
    document.getElementById('s-detecting').textContent = `${detectedLabels.size} types found`;
}

function drawDetections(ctx, bboxes, labels, scaleX, scaleY, style) {
    ctx.font = 'bold 12px -apple-system, sans-serif';
    for (let i = 0; i < bboxes.length; i++) {
        const [x1, y1, x2, y2] = bboxes[i];
        const color = COLORS[i % COLORS.length];
        const sx1 = x1 * scaleX, sy1 = y1 * scaleY;
        const w = (x2 - x1) * scaleX, h = (y2 - y1) * scaleY;

        if (style === 'filled') {
            const r = parseInt(color.slice(1, 3), 16), g = parseInt(color.slice(3, 5), 16), b = parseInt(color.slice(5, 7), 16);
            ctx.fillStyle = `rgba(${r},${g},${b},0.25)`;
            ctx.fillRect(sx1, sy1, w, h);
        }
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(sx1, sy1, w, h);

        const label = labels[i] || '';
        if (label) {
            const tw = ctx.measureText(label).width + 6;
            ctx.fillStyle = color;
            ctx.fillRect(sx1, sy1 - 16, tw, 16);
            ctx.fillStyle = '#fff';
            ctx.fillText(label, sx1 + 3, sy1 - 3);
        }
    }
}

function updateTicker() {
    const ticker = document.getElementById('s-ticker');
    const sorted = [...detectedLabels].sort();
    ticker.innerHTML = sorted.map(l => `<div class="ticker-item">${l}</div>`).join('');
    // Auto-scroll to end
    ticker.scrollLeft = ticker.scrollWidth;
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
    document.getElementById('r-meta').textContent =
        `Scanned ${mins}:${secs.toString().padStart(2, '0')} | ${frameCount} frames processed | ${capturedFrames.length} key frames captured`;

    // Show processing state
    document.getElementById('r-processing').style.display = 'block';
    document.getElementById('r-inventory').style.display = 'none';
    document.getElementById('r-total').style.display = 'none';

    const avgMs = scanStats.inferenceCount > 0 ? Math.round(scanStats.inferenceSum / scanStats.inferenceCount) : 0;
    const fps = avgMs > 0 ? (1000 / avgMs).toFixed(1) : '—';
    document.getElementById('r-stats').innerHTML =
        `Florence-2 Live Detection<br>` +
        `Avg inference: ${avgMs}ms/frame | FPS: ${fps}<br>` +
        `Frames detected: ${frameCount} | Key frames: ${capturedFrames.length}<br>` +
        `Live labels found: ${detectedLabels.size} types`;

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
        const sizeStr = item.size ? `<span class="inv-size">${item.size}</span>` : '';
        return `<div class="inv-item">
      <div><span class="inv-name">${item.name}</span> ${sizeStr}</div>
      <span class="inv-count">×${count}</span>
    </div>`;
    }).join('');

    const totalEl = document.getElementById('r-total');
    totalEl.style.display = 'block';
    totalEl.textContent = `Total: ${items.length} unique items, ${totalCount} pieces`;

    // Update stats with Gemini info
    const statsEl = document.getElementById('r-stats');
    statsEl.innerHTML += `<br><br>Gemini 2.0 Flash Inventory<br>` +
        `Items found: ${items.length} | Total pieces: ${totalCount}<br>` +
        `Cost: ~$${(data.cost || 0).toFixed(3)}`;
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
