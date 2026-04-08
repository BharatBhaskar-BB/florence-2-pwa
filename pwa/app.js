// ── BundleBox PWA — Live Detection + Gemini Inventory ──────────────────────

const API = window.location.origin;

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

const video = document.getElementById('s-video');
const canvas = document.getElementById('s-canvas');
const ctx = canvas.getContext('2d');
const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

// ── Colors ─────────────────────────────────────────────────────────────────
const COLORS = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF',
    '#FF8000', '#8000FF', '#00FF80', '#FF0080', '#0080FF', '#80FF00',
];

// ── Init ───────────────────────────────────────────────────────────────────
if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/sw.js').catch(() => { });
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
    detectedLabels.clear();
    capturedFrames = [];
    frameCount = 0;
    scanStats = { totalMs: 0, inferenceSum: 0, inferenceCount: 0 };
    lastCaptureTime = 0;
    paused = false;
    document.getElementById('s-pause').textContent = '⏸ Pause';
    document.getElementById('s-ticker').innerHTML = '';
    document.getElementById('s-detecting').textContent = 'Detecting...';

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
        });
        video.srcObject = stream;
        await video.play();
    } catch (e) {
        alert('Camera access denied: ' + e.message);
        return;
    }

    showScreen('scan');
    scanning = true;
    startTime = Date.now();

    timerInterval = setInterval(updateTimer, 500);
    detectLoop();
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
    if (!paused && scanning) detectLoop();
}

// ── Detection Loop ─────────────────────────────────────────────────────────
async function detectLoop() {
    if (!scanning || paused) return;

    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;

    // Capture to offscreen canvas
    captureCanvas.width = vw;
    captureCanvas.height = vh;
    captureCtx.drawImage(video, 0, 0, vw, vh);

    const task = document.getElementById('h-task').value;
    const base64 = captureCanvas.toDataURL('image/jpeg', 0.7).split(',')[1];

    // Capture key frame for Gemini every N seconds
    const now = Date.now();
    if (now - lastCaptureTime >= CAPTURE_INTERVAL) {
        capturedFrames.push(base64);
        lastCaptureTime = now;
    }

    try {
        const t0 = performance.now();
        const res = await fetch(API + '/api/detect', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64, task }),
        });
        const data = await res.json();
        const elapsed = performance.now() - t0;

        // Draw to visible canvas
        canvas.width = vw;
        canvas.height = vh;
        ctx.drawImage(captureCanvas, 0, 0);

        const scaleX = vw / data.image_width;
        const scaleY = vh / data.image_height;
        const style = document.getElementById('h-style').value;
        drawDetections(ctx, data.bboxes, data.labels, scaleX, scaleY, style);

        frameCount++;
        scanStats.inferenceSum += data.time_ms;
        scanStats.inferenceCount++;

        // Update ticker with newly seen labels
        for (const l of data.labels) detectedLabels.add(l);
        updateTicker();
        document.getElementById('s-detecting').textContent = `${detectedLabels.size} types found`;

    } catch (e) {
        // silent — keep scanning
    }

    if (scanning && !paused) {
        requestAnimationFrame(detectLoop);
    }
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

    // Stop camera
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
