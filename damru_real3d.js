/* DAMRU REAL3D v1 - turn any Visualise model or any text prompt into a REAL
 * textured 3D GLB via the backend TRELLIS bridge (prompt -> image -> 3D).
 * Additive only: injects a Real 3D button into the Visualise toolbar and uses
 * the existing window.__damruPromptTo3D pipeline. Loaded by damru_boost.js.
 */
(function(){
'use strict';
if (window.__damruReal3D) { return; }
window.__damruReal3D = { v: 1 };

var ICE = String.fromCodePoint(0x1F9CA);

function log(){ try { console.log.apply(console, ['[DamruReal3D]'].concat([].slice.call(arguments))); } catch (e) {} }
function el(tag, css, txt){ var e = document.createElement(tag); if (css) { e.style.cssText = css; } if (txt != null) { e.textContent = txt; } return e; }

function currentPrompt(){
  try {
    var d = window.DamruVisualise;
    var sp = d && d.state && d.state.lastSpec;
    if (sp) { return sp.name || sp.title || sp.prompt || sp.label || ''; }
  } catch (e) {}
  return '';
}

var modal = null, input = null, statusEl = null, genBtn = null, busy = false;

function setStatus(msg, color){ if (statusEl) { statusEl.textContent = msg || ''; statusEl.style.color = color || '#9ba6bd'; } }

function buildModal(){
  if (modal) { return; }
  modal = el('div', 'position:fixed;inset:0;z-index:100460;background:rgba(3,5,9,.92);display:none;align-items:center;justify-content:center;padding:14px;font-family:Inter,system-ui;color:#eaf0fb');
  modal.id = 'dr3-modal';
  var card = el('div', 'width:min(560px,100%);max-height:92vh;overflow:auto;background:#0e131d;border:1px solid #2f3a52;border-radius:16px;padding:18px;box-shadow:0 25px 90px #000');

  var head = el('div', 'display:flex;gap:10px;align-items:center;margin-bottom:8px');
  head.appendChild(el('span', 'font-size:24px', ICE));
  head.appendChild(el('h2', 'margin:0;flex:1;font-size:18px', 'Real 3D Generator'));
  var x = el('button', 'background:#202a40;color:#fff;border:1px solid #3a4664;border-radius:9px;padding:6px 11px;cursor:pointer', 'Close');
  x.onclick = close;
  head.appendChild(x);
  card.appendChild(head);

  card.appendChild(el('div', 'font-size:12px;color:#9ba6bd;margin-bottom:10px', 'Koi bhi cheez likho (ya current model use karo). Backend image banata hai, phir TRELLIS real textured 3D GLB banata hai, jo yahin studio me load ho jaata hai.'));

  input = el('textarea', 'width:100%;min-height:64px;background:#151b27;color:#fff;border:1px solid #34405b;border-radius:9px;padding:10px;font-family:inherit;font-size:14px;resize:vertical;box-sizing:border-box');
  input.placeholder = 'e.g. F-22 Raptor fighter jet, a red sports car, a dragon statue...';
  card.appendChild(input);

  var row = el('div', 'display:flex;gap:8px;flex-wrap:wrap;margin-top:12px;align-items:center');
  genBtn = el('button', 'background:#e8623d;border:1px solid #e8623d;color:#fff;border-radius:9px;padding:10px 14px;cursor:pointer;font-weight:600', 'Generate Real 3D');
  genBtn.onclick = generate;
  row.appendChild(genBtn);
  var useCur = el('button', 'background:#202a40;border:1px solid #3a4664;color:#fff;border-radius:9px;padding:10px 12px;cursor:pointer', 'Use current model');
  useCur.onclick = function(){ var p = currentPrompt(); if (p) { input.value = p; setStatus('Loaded current model name.', '#56d692'); } else { setStatus('No current model detected. Type a prompt.', '#ffbb55'); } };
  row.appendChild(useCur);
  card.appendChild(row);

  statusEl = el('div', 'font-size:12px;color:#9ba6bd;margin-top:12px;line-height:1.5;word-break:break-word', '');
  card.appendChild(statusEl);

  card.appendChild(el('div', 'font-size:11px;color:#7b8699;margin-top:12px;line-height:1.5', 'Real 3D free ZeroGPU TRELLIS Spaces par banta hai. Cold start par 1-3 min lag sakte hain. Busy aaye to thodi der baad dobara try karo.'));

  modal.appendChild(card);
  modal.onclick = function(e){ if (e.target === modal && !busy) { close(); } };
  (document.body || document.documentElement).appendChild(modal);
}

function open(){
  buildModal();
  var p = currentPrompt();
  if (p && input && !input.value) { input.value = p; }
  if (modal) { modal.style.display = 'flex'; }
}
function close(){ if (modal && !busy) { modal.style.display = 'none'; } }

function generate(){
  if (busy) { return; }
  var prompt = ((input && input.value) || '').trim();
  if (!prompt) { setStatus('Pehle kuch likho ya current model use karo.', '#ef6471'); return; }
  if (typeof window.__damruPromptTo3D !== 'function') { setStatus('3D engine abhi ready nahi hai. Page refresh karke dobara try karo.', '#ef6471'); return; }
  busy = true;
  if (genBtn) { genBtn.disabled = true; genBtn.textContent = 'Generating...'; }
  var t0 = Date.now();
  setStatus('Image ban rahi hai, phir TRELLIS 3D bana raha hai. ZeroGPU cold start par 1-3 min. Please wait...', '#ffbd55');
  var tick = setInterval(function(){ if (busy) { setStatus('Still working... ' + Math.round((Date.now() - t0) / 1000) + 's elapsed (ZeroGPU queue ho sakta hai).', '#ffbd55'); } }, 5000);
  Promise.resolve().then(function(){ return window.__damruPromptTo3D(prompt, {}); })
    .then(function(url){
      clearInterval(tick); busy = false;
      if (genBtn) { genBtn.disabled = false; genBtn.textContent = 'Generate Real 3D'; }
      setStatus('Real 3D ready aur studio me load ho gaya! (' + Math.round((Date.now() - t0) / 1000) + 's) Doosra model bhi bana sakte ho.', '#56d692');
      log('generated', url);
    })
    .catch(function(e){
      clearInterval(tick); busy = false;
      if (genBtn) { genBtn.disabled = false; genBtn.textContent = 'Generate Real 3D'; }
      var m = (e && e.message) ? String(e.message) : String(e);
      setStatus('Real 3D abhi nahi ban paya (TRELLIS Space busy ya cold ho sakta hai). Thodi der baad dobara try karo. [' + m.slice(0, 140) + ']', '#ef6471');
      log('generate failed', e);
    });
}

function injectBtn(){
  var top = document.getElementById('dv-top');
  if (top && !document.getElementById('dr3-launch')) {
    var b = document.createElement('button');
    b.id = 'dr3-launch';
    b.className = 'dv-btn';
    b.textContent = ICE + ' Real 3D';
    b.onclick = open;
    top.insertBefore(b, top.firstChild);
  }
}

try { new MutationObserver(injectBtn).observe(document.documentElement, { childList: true, subtree: true }); } catch (e) {}
setTimeout(injectBtn, 1300);
setTimeout(injectBtn, 3000);

window.__damruReal3D.open = open;
window.__damruReal3D.generate = function(p){ buildModal(); if (input) { input.value = p || ''; } open(); generate(); };

})();
