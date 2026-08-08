/* DAMRU BOOST v1 - client-side reliability layer.
 * Fixes the 'saare engines busy' error by making Damru brain resilient to
 * Hugging Face Space cold-starts: warm-on-load + warm-and-retry wrapper around
 * the global engine(). Purely additive and non-destructive: it only wraps
 * existing globals and never blocks the UI. Loaded by the damru_simlab.js
 * loader AFTER the main script has defined engine() and warmDamru().
 */
(function(){
'use strict';
if (window.__damruBoost) { return; }
window.__damruBoost = { v: 1 };

var API = 'https://damaru-ai-damru.hf.space';
var NL = String.fromCharCode(10);

function log(){ try { console.log.apply(console, ['[DamruBoost]'].concat([].slice.call(arguments))); } catch (e) {} }

function isBusy(s){
  if (!s || typeof s !== 'string') { return true; }
  var t = s.toLowerCase();
  return (t.indexOf('saare engine') >= 0) || (t.indexOf('busy/offline') >= 0) || (t.indexOf('engines ek saath') >= 0);
}

function sleep(ms){ return new Promise(function(r){ setTimeout(r, ms); }); }

function pingHealth(timeoutMs){
  return new Promise(function(resolve){
    var done = false, ctrl = null;
    try { ctrl = new AbortController(); } catch (e) { ctrl = null; }
    var t = setTimeout(function(){ if (!done) { done = true; try { if (ctrl) { ctrl.abort(); } } catch (x) {} resolve(false); } }, timeoutMs || 8000);
    var opt = ctrl ? { signal: ctrl.signal, cache: 'no-store' } : { cache: 'no-store' };
    fetch(API + '/health', opt)
      .then(function(r){ return r.ok ? r.json().catch(function(){ return {}; }) : null; })
      .then(function(d){
        if (done) { return; }
        done = true; clearTimeout(t);
        if (!d) { resolve(false); return; }
        var ready = (d.model_loaded === true) || (d.open_brain === true) || (d.brain_ready === true) || (d.ok === true);
        resolve(!!ready);
      })
      .catch(function(){ if (!done) { done = true; clearTimeout(t); resolve(false); } });
  });
}

function ensureReady(budgetMs){
  var start = Date.now();
  var budget = budgetMs || 40000;
  return (function loop(){
    return pingHealth(9000).then(function(ok){
      if (ok) { return true; }
      if ((Date.now() - start) >= budget) { return false; }
      return sleep(3000).then(loop);
    });
  })();
}

function warmOnLoad(){
  try { if (typeof window.warmDamru === 'function') { try { window.warmDamru(); } catch (e) {} } } catch (e) {}
  pingHealth(9000).then(function(ok){ log('warm-on-load ready:', ok); });
}

var FRIENDLY_BUSY = [
  '🐯 Damru abhi wake-up ho raha hai (cold start ho sakta hai). Bas 10-15 second ruk ke apna message dobara bhejo, main ready ho jaunga.',
  '',
  '⚡ Instant backup chahiye to Settings me apni free OpenRouter key daal do.'
].join(NL);

function installEngineGuard(){
  var orig = window.engine;
  if (typeof orig !== 'function') { return false; }
  if (orig.__damruGuarded) { return true; }
  var guarded = function(){
    var args = arguments, self = this;
    function tryOnce(){
      return Promise.resolve().then(function(){ return orig.apply(self, args); }).catch(function(e){ log('engine attempt failed:', e && e.message); return ''; });
    }
    return tryOnce().then(function(out){
      if (!isBusy(out)) { return out; }
      log('busy/empty on attempt 1 -> warm Space + retry once');
      return ensureReady(40000).catch(function(){ return false; }).then(function(){
        return tryOnce().then(function(out2){
          if (!isBusy(out2)) { return out2; }
          return FRIENDLY_BUSY;
        });
      });
    });
  };
  guarded.__damruGuarded = true;
  try { window.engine = guarded; } catch (e) { return false; }
  log('engine guard installed');
  return true;
}

function installWithRetry(n){
  if (installEngineGuard()) { return; }
  if (n <= 0) { log('gave up installing engine guard'); return; }
  setTimeout(function(){ installWithRetry(n - 1); }, 400);
}

function boot(){
  warmOnLoad();
  installWithRetry(20);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}

})();
