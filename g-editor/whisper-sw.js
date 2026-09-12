/* whisper-sw.js · cache de modelos Whisper (Cache API).
   Misma carpeta que Grabadora-Subtitulos.html. */
const CACHE = 'gs-whisper-models-v1';
const HOSTS = [
  'huggingface.co',
  'cdn-lfs.huggingface.co',
  'cdn-lfs-us-1.huggingface.co',
  'cdn.jsdelivr.net',
  'github.com',
  'objects.githubusercontent.com',
  'release-assets.githubusercontent.com',
  'github-releases.githubusercontent.com'
];

function hostOk(u) {
  try {
    const url = new URL(u);
    if (HOSTS.some(h => url.hostname === h || url.hostname.endsWith('.' + h))) return true;
    if (/\.workers\.dev$/i.test(url.hostname) && url.searchParams.get('url')) return true;
    return false;
  } catch (_) { return false; }
}

self.addEventListener('install', ev => {
  ev.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', ev => {
  ev.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k.startsWith('gs-whisper-models-') && k !== CACHE).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', ev => {
  const req = ev.request;
  if (req.method !== 'GET') return;
  if (!hostOk(req.url)) return;
  ev.respondWith((async () => {
    const cache = await caches.open(CACHE);
    const hit = await cache.match(req);
    if (hit) return hit;
    const res = await fetch(req);
    if (res && res.ok && (res.type === 'basic' || res.type === 'cors')) {
      try { await cache.put(req, res.clone()); } catch (_) {}
    }
    return res;
  })());
});
