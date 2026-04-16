// ── BundleBox Service Worker — production-grade caching ──
// Strategy:
//   Navigation/HTML → network-first (fresh UI, offline fallback)
//   JS/CSS          → stale-while-revalidate (instant + background refresh)
//   Icons/images    → cache-first (rarely change)
//   API & mutations → network-only (never cache business data)

const CACHE_NAME = 'bundlebox-v7';

const PRECACHE = ['./', 'app.js', 'style.css', 'manifest.json'];

// ── Install: precache shell, but do NOT skipWaiting automatically ──
self.addEventListener('install', (e) => {
    e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(PRECACHE)));
    // Do NOT call self.skipWaiting() here.
    // Wait for the app to send SKIP_WAITING after user approves the update.
});

// ── Activate: clean old caches, claim clients ──
self.addEventListener('activate', (e) => {
    e.waitUntil(
        caches.keys().then(keys =>
            Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))
        )
    );
    self.clients.claim();
});

// ── Message: controlled skipWaiting on user approval ──
self.addEventListener('message', (e) => {
    if (e.data && e.data.type === 'SKIP_WAITING') {
        self.skipWaiting();
    }
});

// ── Fetch: per-resource caching strategy ──
self.addEventListener('fetch', (e) => {
    const { request } = e;

    // Never cache non-GET requests (POST, PUT, PATCH, DELETE)
    if (request.method !== 'GET') return;

    // API calls → network-only (never cache business data)
    if (request.url.includes('/api/')) {
        e.respondWith(fetch(request));
        return;
    }

    // WebSocket upgrade — let it pass through
    if (request.headers.get('upgrade') === 'websocket') return;

    // Navigation requests (HTML pages) → network-first with cache fallback
    if (request.mode === 'navigate') {
        e.respondWith(
            fetch(request)
                .then(res => {
                    const clone = res.clone();
                    caches.open(CACHE_NAME).then(c => c.put(request, clone));
                    return res;
                })
                .catch(() => caches.match(request).then(r => r || caches.match('./')))
        );
        return;
    }

    // Icons & images → cache-first (long-lived, rarely change)
    if (request.destination === 'image') {
        e.respondWith(
            caches.match(request).then(r => r || fetch(request).then(res => {
                const clone = res.clone();
                caches.open(CACHE_NAME).then(c => c.put(request, clone));
                return res;
            }))
        );
        return;
    }

    // JS, CSS, manifest → stale-while-revalidate
    // Serve from cache immediately, fetch fresh copy in background for next load
    e.respondWith(
        caches.open(CACHE_NAME).then(cache =>
            cache.match(request).then(cached => {
                const fetched = fetch(request).then(res => {
                    cache.put(request, res.clone());
                    return res;
                }).catch(() => cached);
                return cached || fetched;
            })
        )
    );
});
