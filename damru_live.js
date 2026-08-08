/* DAMRU LIVE v1 - real live VIDEO chat for Damru, added into index.html UI.
 * Additive only: injects a camera button into the composer input row and opens
 * a fullscreen live-video modal. Uses device camera (getUserMedia) + browser
 * SpeechRecognition (STT) -> Damru /chat brain -> speechSynthesis (TTS) for a
 * natural spoken video conversation. Loaded by damru_boost.js. No index.html edit.
 */
(function(){
'use strict';
if (window.__damruLive) { return; }
window.__damruLive = { v: 1 };

var API = 'https://damaru-ai-damru.hf.space';
var CAM = String.fromCodePoint(0x1F4F9);
var REC = String.fromCodePoint(0x1F534);
var SYS = 'You are Damru, a warm friendly Hinglish AI on a LIVE VIDEO call. Reply in short natural spoken-style Hinglish, 1 to 3 sentences, because your reply will be spoken aloud. Be conversational and human.';

function log(){ try { console.log.apply(console, ['[DamruLive]'].concat([].slice.call(arguments))); } catch (e) {} }
function el(tag, css, txt){ var e = document.createElement(tag); if (css) { e.style.cssText = css; } if (txt != null) { e.textContent = txt; } return e; }

var modal = null, video = null, statusEl = null, capYou = null, capDam = null;
var recog = null, mediaStream = null;
var active = false, speaking = false, muted = false, hist = [];

function setStatus(t){ if (statusEl) { statusEl.textContent = t || ''; } }
function setCap(elm, t){ if (elm) { elm.textContent = t || ''; } }

function ask(text){
  return fetch(API + '/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: text, system: SYS, history: hist }) })
    .then(function(r){ return r.json(); })
    .then(function(d){ return (d && (d.answer || d.response || d.text)) || ''; });
}

function speak(text){
  setStatus('Bol raha hoon...');
  try {
    speechSynthesis.cancel();
    var u = new SpeechSynthesisUtterance(text);
    u.lang = 'hi-IN';
    var vs = speechSynthesis.getVoices() || [];
    for (var i = 0; i < vs.length; i++) { if (vs[i].lang && vs[i].lang.indexOf('hi') === 0) { u.voice = vs[i]; break; } }
    u.onend = function(){ speaking = false; setStatus('Sun raha hoon...'); listen(); };
    u.onerror = function(){ speaking = false; setStatus('Sun raha hoon...'); listen(); };
    speechSynthesis.speak(u);
  } catch (e) { speaking = false; setStatus('Sun raha hoon...'); listen(); }
}

function handle(said){
  if (!said) { return; }
  setCap(capYou, 'Tum: ' + said);
  setCap(capDam, '');
  setStatus('Soch raha hoon...');
  speaking = true;
  try { if (recog) { recog.stop(); } } catch (e) {}
  ask(said).then(function(ans){
    if (!ans) { ans = 'Sorry, thodi der baad dobara bolo.'; }
    hist.push({ role: 'user', content: said });
    hist.push({ role: 'assistant', content: ans });
    if (hist.length > 12) { hist = hist.slice(-12); }
    setCap(capDam, 'Damru: ' + ans);
    speak(ans);
  }).catch(function(e){
    log('ask failed', e);
    setCap(capDam, 'Damru: connection issue, dobara bolo.');
    speak('Sorry, dobara bolo.');
  });
}

function makeRecog(){
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SR) { return null; }
  var r = new SR();
  r.lang = 'hi-IN';
  r.interimResults = false;
  r.maxAlternatives = 1;
  r.continuous = false;
  r.onresult = function(e){
    var said = '';
    try { said = e.results[0][0].transcript; } catch (x) { said = ''; }
    if (said) { handle(said); }
  };
  r.onend = function(){ if (active && !speaking && !muted) { try { r.start(); } catch (e) {} } };
  r.onerror = function(){};
  return r;
}

function listen(){ if (!recog || !active || speaking || muted) { return; } try { recog.start(); } catch (e) {} }

function buildModal(){
  if (modal) { return; }
  modal = el('div', 'position:fixed;inset:0;z-index:100470;background:#05070c;display:none;flex-direction:column;font-family:Inter,system-ui;color:#eaf0fb');
  modal.id = 'dlv-modal';

  video = document.createElement('video');
  video.autoplay = true; video.muted = true; video.playsInline = true;
  video.setAttribute('playsinline', '');
  video.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1);background:#000';
  modal.appendChild(video);

  var top = el('div', 'position:relative;z-index:2;display:flex;align-items:center;gap:10px;padding:14px 16px;background:linear-gradient(#000000aa,transparent)');
  var live = el('div', 'display:flex;align-items:center;gap:8px;font-weight:700;font-size:15px');
  live.appendChild(el('span', 'font-size:12px', REC));
  live.appendChild(el('span', null, 'LIVE - Damru'));
  top.appendChild(live);
  top.appendChild(el('div', 'flex:1'));
  var endBtn = el('button', 'background:#e5484d;color:#fff;border:none;border-radius:10px;padding:9px 16px;cursor:pointer;font-weight:700', 'End');
  endBtn.onclick = closeLive;
  top.appendChild(endBtn);
  modal.appendChild(top);

  modal.appendChild(el('div', 'flex:1'));

  var bot = el('div', 'position:relative;z-index:2;padding:16px;background:linear-gradient(transparent,#000000cc);display:flex;flex-direction:column;gap:8px');
  statusEl = el('div', 'align-self:center;background:rgba(14,19,29,.8);border:1px solid #2f3a52;border-radius:999px;padding:7px 16px;font-size:13px;color:#9fe8c8', 'Ready');
  bot.appendChild(statusEl);
  capYou = el('div', 'font-size:13px;color:#8fb7ff;text-align:center;min-height:18px;word-break:break-word', '');
  capDam = el('div', 'font-size:15px;color:#fff;text-align:center;min-height:20px;font-weight:600;word-break:break-word', '');
  bot.appendChild(capYou);
  bot.appendChild(capDam);
  var ctrls = el('div', 'display:flex;gap:10px;justify-content:center;margin-top:6px');
  var muteBtn = el('button', 'background:#202a40;color:#fff;border:1px solid #3a4664;border-radius:10px;padding:9px 16px;cursor:pointer', 'Mute mic');
  muteBtn.onclick = function(){
    muted = !muted;
    muteBtn.textContent = muted ? 'Unmute mic' : 'Mute mic';
    if (mediaStream) { try { mediaStream.getAudioTracks().forEach(function(t){ t.enabled = !muted; }); } catch (e) {} }
    if (muted) { try { if (recog) { recog.stop(); } } catch (e) {} setStatus('Mic muted'); }
    else { setStatus('Sun raha hoon...'); listen(); }
  };
  ctrls.appendChild(muteBtn);
  bot.appendChild(ctrls);
  modal.appendChild(bot);

  (document.body || document.documentElement).appendChild(modal);
}

function openLive(){
  buildModal();
  modal.style.display = 'flex';
  active = true; speaking = false; muted = false;
  setStatus('Camera on kar raha hoon...');
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('Is browser me camera support nahi hai.');
    return;
  }
  navigator.mediaDevices.getUserMedia({ video: true, audio: true }).then(function(stream){
    mediaStream = stream;
    if (video) { video.srcObject = stream; try { video.play(); } catch (e) {} }
    recog = makeRecog();
    if (!recog) { setStatus('Voice support nahi hai. Chrome me try karo.'); return; }
    setStatus('Sun raha hoon...');
    listen();
  }).catch(function(e){
    log('getUserMedia failed', e);
    setStatus('Camera/mic permission chahiye. Allow karke dobara try karo.');
  });
}

function closeLive(){
  active = false; speaking = false;
  try { if (recog) { recog.stop(); } } catch (e) {}
  try { speechSynthesis.cancel(); } catch (e) {}
  try { if (mediaStream) { mediaStream.getTracks().forEach(function(t){ t.stop(); }); } } catch (e) {}
  mediaStream = null;
  if (video) { try { video.srcObject = null; } catch (e) {} }
  if (modal) { modal.style.display = 'none'; }
  hist = [];
}

function injectBtn(){
  var row = document.querySelector('.inp-row');
  if (row && !document.getElementById('dlv-btn')) {
    var send = document.getElementById('send');
    var b = document.createElement('button');
    b.id = 'dlv-btn';
    b.className = 'icb';
    b.title = 'Live video chat';
    b.textContent = CAM;
    b.onclick = openLive;
    if (send) { row.insertBefore(b, send); } else { row.appendChild(b); }
  }
}

try { new MutationObserver(injectBtn).observe(document.documentElement, { childList: true, subtree: true }); } catch (e) {}
setTimeout(injectBtn, 1200);
setTimeout(injectBtn, 2600);

window.__damruLive.open = openLive;
window.__damruLive.close = closeLive;

})();
