/* Damru Sovereign offline shell. Service workers are opportunistic, not guaranteed 24/7. */
const CACHE='damru-sovereign-shell-v1';
const ASSETS=['/','/index.html','/damru_sovereign_mind.js','/damru_model_vault.js','/damru_visualise.js','/damru_universe_sim.js','/damru_truth_forge.js','/damru_simlab.js','/damru_print_forge.js'];
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(ASSETS).catch(()=>{})).then(()=>self.skipWaiting())));
self.addEventListener('activate',e=>e.waitUntil(caches.keys().then(keys=>Promise.all(keys.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim())));
self.addEventListener('fetch',e=>{if(e.request.method!=='GET')return;const u=new URL(e.request.url);if(u.origin!==location.origin)return;e.respondWith(fetch(e.request).then(r=>{const c=r.clone();caches.open(CACHE).then(x=>x.put(e.request,c));return r}).catch(()=>caches.match(e.request).then(r=>r||caches.match('/index.html'))))});
