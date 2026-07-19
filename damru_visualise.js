/* ============================================================================
 * DAMRU VISUALISE  --  AI CAD / 3D Simulation Studio
 * ----------------------------------------------------------------------------
 * World-class, self-contained 3D generator for Damru AI.
 *  - Fullscreen studio (modal overlay)
 *  - Three.js + three-bvh-csg (real boolean CAD) via importmap
 *  - Hybrid engine: AI outputs a JSON SCENE-SPEC (safe, deterministic,
 *    editable) with an optional procedural code escape-hatch. The heavy
 *    lifting is done by THIS code so Damru only needs to emit a spec ->
 *    complex builds (even a full city) become reliable.
 *  - Primitives + CSG (subtract/union/intersect) + procedural repeat/instancing
 *  - Orbit / pan / zoom + move on any axis (TransformControls: G/R/S, X/Y/Z)
 *  - Export GLTF (.glb) + STL (.stl)
 *  - Edit / duplicate / delete / regenerate individual parts
 *  - Animation loop + first-person walkthrough (desktop pointer-lock + mobile pad)
 *  - Grid + axes + measurement tool (real units)
 *  - Fully reactive (resize + touch)
 *
 * Integration (index.html):
 *   1) importmap in <head> mapping three / three/addons/ / three-bvh-csg
 *   2) <script src="damru_visualise.js"></script> before </body>
 *   3) window.__damruLLM(messages, maxTokens) bridge (Damru brain first)
 * The module self-injects a launcher (composer pill + sidebar item).
 * ==========================================================================*/
(function () {
  'use strict';
  if (window.DamruVisualise) return;

  // ---- lazy-loaded libs (resolved via <script type=importmap>) -------------
  var THREE, OrbitControls, TransformControls, PointerLockControls,
      GLTFExporter, STLExporter, GLTFLoader,
      CSG_Evaluator, CSG_Brush, CSG_ADD, CSG_SUB, CSG_INT;
  var libsReady = false;

  async function loadLibs() {
    if (libsReady) return;
    try { THREE = await import('three'); }
    catch (e) {
      console.warn('[visualise] primary three CDN failed, trying fallback', e);
      THREE = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
    }
    OrbitControls = (await import('three/addons/controls/OrbitControls.js')).OrbitControls;
    TransformControls = (await import('three/addons/controls/TransformControls.js')).TransformControls;
    try { PointerLockControls = (await import('three/addons/controls/PointerLockControls.js')).PointerLockControls; } catch (e) {}
    GLTFExporter = (await import('three/addons/exporters/GLTFExporter.js')).GLTFExporter;
    STLExporter = (await import('three/addons/exporters/STLExporter.js')).STLExporter;
    try { GLTFLoader = (await import('three/addons/loaders/GLTFLoader.js')).GLTFLoader; } catch (e) { console.warn('[visualise] GLTFLoader unavailable', e); }
    try {
      var csg = await import('three-bvh-csg');
      CSG_Evaluator = csg.Evaluator; CSG_Brush = csg.Brush;
      CSG_ADD = csg.ADDITION; CSG_SUB = csg.SUBTRACTION; CSG_INT = csg.INTERSECTION;
    } catch (e) { console.warn('[visualise] CSG lib unavailable; boolean ops will fallback to grouping', e); }
    libsReady = true;
  }

  // ---- studio state --------------------------------------------------------
  var S = {
    scene: null, camera: null, renderer: null, orbit: null, transform: null,
    root: null, grid: null, axes: null, ground: null,
    animated: [], selectable: [], selected: null,
    raycaster: null, clock: null, rafId: null,
    measureMode: false, measurePts: [], measureObjs: [],
    walk: null, walkActive: false, move: { f: 0, b: 0, l: 0, r: 0, u: 0, d: 0 },
    lastSpec: null, lastPrompt: '', units: 'm', built: false,
    generationSeq: 0, generating: false, research: []
  };

  // ---- Chakra identity bridge ---------------------------------------------
  function visualChakra(size, cls) {
    if (window.DamruChakra && typeof window.DamruChakra.svg === 'function') return window.DamruChakra.svg(size || 18, cls || 'spin');
    return '<svg class="chakra '+(cls||'spin')+'" width="'+(size||18)+'" height="'+(size||18)+'" viewBox="0 0 64 64"><circle cx="32" cy="32" r="25" fill="none" stroke="#4f86ff" stroke-width="4"/><circle cx="32" cy="32" r="5" fill="#4f86ff"/></svg>';
  }

  // ---- tiny DOM helper -----------------------------------------------------
  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    attrs = attrs || {};
    for (var k in attrs) {
      if (k === 'style') e.style.cssText = attrs[k];
      else if (k === 'html') e.innerHTML = attrs[k];
      else if (k === 'text') e.textContent = attrs[k];
      else if (k.slice(0, 2) === 'on' && typeof attrs[k] === 'function') e.addEventListener(k.slice(2), attrs[k]);
      else e.setAttribute(k, attrs[k]);
    }
    (kids || []).forEach(function (c) { if (c) e.appendChild(c); });
    return e;
  }

  // ---- styles --------------------------------------------------------------
  function injectStyles() {
    if (document.getElementById('dv-styles')) return;
    var css = ''
      + '#dv-modal{position:fixed;inset:0;z-index:99999;background:#0b0d12;display:none;flex-direction:column;font-family:Inter,system-ui,sans-serif;color:#e8eaf0}'
      + '#dv-modal.dv-open{display:flex}'
      + '#dv-top{display:flex;align-items:center;gap:8px;padding:8px 12px;background:#12151d;border-bottom:1px solid #232838;flex-wrap:wrap}'
      + '#dv-top .dv-title{font-weight:700;font-size:15px;margin-right:auto;display:flex;align-items:center;gap:8px}'
      + '.dv-btn{background:#1c2233;color:#dfe4f0;border:1px solid #2b3350;border-radius:9px;padding:7px 11px;font-size:13px;cursor:pointer;display:inline-flex;align-items:center;gap:5px}'
      + '.dv-btn:hover{background:#273154;border-color:#3a4573}'
      + '.dv-btn.on{background:#e8623d;border-color:#e8623d;color:#fff}'
      + '.dv-btn.icon{padding:7px 9px}'
      + '#dv-body{flex:1;display:flex;min-height:0}'
      + '#dv-canvas-wrap{flex:1;position:relative;min-width:0}'
      + '#dv-canvas-wrap canvas{display:block;width:100%;height:100%;touch-action:none}'
      + '#dv-side{width:270px;background:#0f121a;border-left:1px solid #232838;display:flex;flex-direction:column;overflow:hidden}'
      + '#dv-side h4{margin:0;padding:10px 12px;font-size:12px;letter-spacing:.5px;color:#8b93ab;text-transform:uppercase;border-bottom:1px solid #1c2130}'
      + '#dv-outliner{flex:1;overflow:auto;padding:6px}'
      + '.dv-node{padding:7px 9px;border-radius:8px;font-size:13px;cursor:pointer;display:flex;align-items:center;gap:6px;color:#cfd6e6}'
      + '.dv-node:hover{background:#1a2030}.dv-node.sel{background:#2a3252;color:#fff}'
      + '#dv-edit{padding:8px;border-top:1px solid #1c2130;display:flex;flex-wrap:wrap;gap:6px;align-items:center}'
      + '.dv-mat{display:flex;gap:5px;align-items:center;background:#171d2a;border:1px solid #2b3350;border-radius:9px;padding:4px 7px;font-size:10px;color:#9aa7c3}.dv-mat input[type=color]{width:30px;height:25px;border:0;background:transparent;padding:0}.dv-mat input[type=range]{width:58px}'
      + '#dv-prompt{display:flex;gap:8px;padding:10px 12px;background:#12151d;border-top:1px solid #232838}'
      + '#dv-prompt input{flex:1;background:#0b0d12;border:1px solid #2b3350;border-radius:10px;padding:11px 13px;color:#fff;font-size:14px;outline:none}'
      + '#dv-prompt input:focus{border-color:#e8623d}'
      + '#dv-go{background:#e8623d;color:#fff;border:none;border-radius:10px;padding:0 18px;font-weight:700;cursor:pointer;font-size:14px}'
      + '#dv-photoreal{background:#1c9c6b;color:#fff;border:none;border-radius:10px;padding:0 12px;font-weight:700;cursor:pointer;font-size:12px}'
      + '#dv-photoreal:disabled{opacity:.5;cursor:default}'
      + '#dv-go:disabled{opacity:.5;cursor:default}'
      + '#dv-cancel{background:#51232a;border:1px solid #8a3945;color:#ffdfe3;border-radius:10px;padding:0 13px;cursor:pointer;display:none}'
      + '#dv-status{position:absolute;top:12px;left:12px;background:rgba(10,12,18,.85);border:1px solid #2b3350;border-radius:10px;padding:8px 13px;font-size:13px;display:none;align-items:center;gap:8px;max-width:70%}'
      + '#dv-status.show{display:flex}'
      + '.dv-spin{width:20px;height:20px;display:grid;place-items:center;color:#4f86ff;flex:none}.dv-spin .chakra{display:block}'
      + '@keyframes dvspin{to{transform:rotate(360deg)}}'
      + '#dv-hud{position:absolute;bottom:12px;left:12px;font-size:11px;color:#7f8aa6;background:rgba(10,12,18,.7);padding:5px 9px;border-radius:8px;pointer-events:none;max-width:60%}'
      + '#dv-pad{position:absolute;bottom:16px;right:16px;display:none;grid-template-columns:repeat(3,44px);grid-gap:6px;z-index:5}'
      + '#dv-pad.show{display:grid}'
      + '#dv-pad button{width:44px;height:44px;border-radius:10px;background:rgba(28,34,51,.9);border:1px solid #2b3350;color:#fff;font-size:16px}'
      + '#dv-launch{position:fixed;bottom:88px;right:18px;z-index:9000;background:#e8623d;color:#fff;border:none;border-radius:50px;padding:12px 16px;font-weight:700;box-shadow:0 6px 20px rgba(0,0,0,.4);cursor:pointer;display:none}'
      + '@media(max-width:820px){#dv-side{position:absolute;right:0;top:0;bottom:0;transform:translateX(100%);transition:.25s;z-index:6}#dv-side.show{transform:none}#dv-status{max-width:86%}}';
    document.head.appendChild(el('style', { id: 'dv-styles', html: css }));
  }

  // ---- build modal DOM -----------------------------------------------------
  function buildModal() {
    if (document.getElementById('dv-modal')) return;
    injectStyles();

    var status = el('div', { id: 'dv-status' }, [el('div', { class: 'dv-spin', html: visualChakra(18, 'spin') }), el('span', { id: 'dv-status-txt', text: 'Working...' })]);
    var hud = el('div', { id: 'dv-hud', html: 'Drag = orbit &middot; two-finger/scroll = zoom &middot; select a part to move (G) rotate (R) scale (S), lock axis X/Y/Z' });
    var pad = el('div', { id: 'dv-pad' });
    [['\u2196', ''], ['\u2191', 'f'], ['\u2197', ''], ['\u2190', 'l'], ['\u23FA', ''], ['\u2192', 'r'], ['', ''], ['\u2193', 'b'], ['', '']].forEach(function (p) {
      if (!p[0]) { pad.appendChild(el('span')); return; }
      var b = el('button', { text: p[0] });
      if (p[1]) {
        var set = function (v) { return function (ev) { ev.preventDefault(); S.move[p[1]] = v; }; };
        b.addEventListener('touchstart', set(1)); b.addEventListener('touchend', set(0));
        b.addEventListener('mousedown', set(1)); b.addEventListener('mouseup', set(0));
      }
      pad.appendChild(b);
    });

    var canvasWrap = el('div', { id: 'dv-canvas-wrap' }, [status, hud, pad]);

    var outliner = el('div', { id: 'dv-outliner' });
    var editBar = el('div', { id: 'dv-edit' }, [
      mkBtn('\u2725 Move', function () { setTransformMode('translate'); }),
      mkBtn('\u21BB Rotate', function () { setTransformMode('rotate'); }),
      mkBtn('\u2921 Scale', function () { setTransformMode('scale'); }),
      mkBtn('\u29C9 Duplicate', duplicateSelected),
      mkBtn('\u267B Regenerate', regenerateSelected),
      mkBtn('\u2715 Delete', deleteSelected),
      el('label',{class:'dv-mat',html:'Color <input id="dv-color" type="color" value="#9aa7c7">'}),
      el('label',{class:'dv-mat',html:'Rough <input id="dv-rough" type="range" min="0" max="1" step=".02" value=".6">'})
    ]);
    setTimeout(function(){ var cp=document.getElementById('dv-color'),rp=document.getElementById('dv-rough'); if(cp)cp.oninput=function(){applySelectedMaterial('color',this.value)}; if(rp)rp.oninput=function(){applySelectedMaterial('roughness',+this.value)}; },0);
    var side = el('div', { id: 'dv-side' }, [el('h4', { text: 'Scene outliner' }), outliner, editBar]);

    var body = el('div', { id: 'dv-body' }, [canvasWrap, side]);

    var top = el('div', { id: 'dv-top' }, [
      el('div', { class: 'dv-title', html: visualChakra(20, 'pulse')+' Damru Visualise <span style="color:#7f8aa6;font-weight:500">&middot; AI CAD studio</span>' }),
      mkBtn('\u25A6 Grid', function (e) { toggleGrid(e.currentTarget); }, true),
      mkBtn('\u21BA Spin', function (e) { toggleAutoRotate(e.currentTarget); }),
      mkBtn('\uD83D\uDCCF Measure', function (e) { toggleMeasure(e.currentTarget); }),
      mkBtn('\uD83D\uDEB6 Walk', function (e) { toggleWalk(e.currentTarget); }),
      mkBtn('\u2913 GLB', function () { exportGLTF(); }),
      mkBtn('\u2913 STL', function () { exportSTL(); }),
      mkBtn('\uD83D\uDDD1 Clear', function () { clearScene(); refreshOutliner(); }),
      mkBtn('\u2630', function () { document.getElementById('dv-side').classList.toggle('show'); }, false, 'icon'),
      mkBtn('\u2715 Close', function () { close(); }, false, 'icon')
    ]);

    var promptInput = el('input', { id: 'dv-prompt-in', type: 'text', placeholder: 'Describe anything: "a modern 3-floor villa", "gear with 12 teeth", "a whole smart city"...' });
    promptInput.addEventListener('keydown', function (e) { if (e.key === 'Enter') runPrompt(); });
    var promptBar = el('div', { id: 'dv-prompt' }, [promptInput, el('button', { id: 'dv-cancel', text: 'Cancel', onclick: cancelGeneration }), el('button', { id: 'dv-photoreal', text: '\uD83C\uDF10 Photoreal 3D', title: 'AI image \u2192 real textured 3D (TRELLIS)', onclick: runPhotoreal }), el('button', { id: 'dv-go', text: 'Visualise', onclick: runPrompt })]);

    var modal = el('div', { id: 'dv-modal' }, [top, body, promptBar]);
    document.body.appendChild(modal);

    document.addEventListener('keydown', onKey);
  }

  function mkBtn(label, fn, on, extra) {
    return el('button', { class: 'dv-btn ' + (on ? 'on ' : '') + (extra || ''), text: label, onclick: fn });
  }

  // ---- three.js setup ------------------------------------------------------
  function setupThree() {
    if (S.renderer) return;
    var wrap = document.getElementById('dv-canvas-wrap');
    S.scene = new THREE.Scene();
    S.scene.background = new THREE.Color('#0b0d12');
    S.scene.fog = new THREE.Fog('#0b0d12', 200, 1200);

    S.camera = new THREE.PerspectiveCamera(55, wrap.clientWidth / wrap.clientHeight, 0.05, 5000);
    S.camera.position.set(14, 12, 18);

    S.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    S.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    S.renderer.setSize(wrap.clientWidth, wrap.clientHeight);
    S.renderer.shadowMap.enabled = true;
    S.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    wrap.appendChild(S.renderer.domElement);

    // lights
    S.scene.add(new THREE.HemisphereLight('#dfe9ff', '#2a2f3d', 0.9));
    var amb = new THREE.AmbientLight('#ffffff', 0.25); S.scene.add(amb);
    var sun = new THREE.DirectionalLight('#fff6e6', 1.15);
    sun.position.set(40, 70, 30); sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.near = 1; sun.shadow.camera.far = 400;
    sun.shadow.camera.left = -120; sun.shadow.camera.right = 120;
    sun.shadow.camera.top = 120; sun.shadow.camera.bottom = -120;
    S.scene.add(sun);

    // helpers
    S.grid = new THREE.GridHelper(400, 80, 0x3a4573, 0x1c2233);
    S.scene.add(S.grid);
    S.axes = new THREE.AxesHelper(6); S.scene.add(S.axes);

    var gMat = new THREE.MeshStandardMaterial({ color: '#0e1118', roughness: 1, metalness: 0 });
    S.ground = new THREE.Mesh(new THREE.PlaneGeometry(4000, 4000), gMat);
    S.ground.rotation.x = -Math.PI / 2; S.ground.position.y = -0.01;
    S.ground.receiveShadow = true; S.scene.add(S.ground);

    S.root = new THREE.Group(); S.root.name = 'root'; S.scene.add(S.root);

    S.orbit = new OrbitControls(S.camera, S.renderer.domElement);
    S.orbit.enableDamping = true; S.orbit.dampingFactor = 0.08;
    S.orbit.maxPolarAngle = Math.PI * 0.495; S.orbit.target.set(0, 2, 0);

    S.transform = new TransformControls(S.camera, S.renderer.domElement);
    S.transform.addEventListener('dragging-changed', function (e) { S.orbit.enabled = !e.value; });
    if (S.transform.getHelper) S.scene.add(S.transform.getHelper()); else S.scene.add(S.transform);

    S.raycaster = new THREE.Raycaster();
    S.clock = new THREE.Clock();

    S.renderer.domElement.addEventListener('pointerdown', onCanvasClick);
    window.addEventListener('resize', onResize);

    if (PointerLockControls) {
      S.walk = new PointerLockControls(S.camera, S.renderer.domElement);
      S.walk.addEventListener('unlock', function () { if (!isTouch()) endWalk(); });
    }
    animate();
  }

  function isTouch() { return 'ontouchstart' in window; }

  function onResize() {
    if (!S.renderer) return;
    var wrap = document.getElementById('dv-canvas-wrap');
    S.camera.aspect = wrap.clientWidth / wrap.clientHeight;
    S.camera.updateProjectionMatrix();
    S.renderer.setSize(wrap.clientWidth, wrap.clientHeight);
  }

  function animate() {
    S.rafId = requestAnimationFrame(animate);
    var dt = S.clock.getDelta();
    S.animated.forEach(function (a) {
      var o = a.obj;
      if (a.type === 'rotate' || a.type === 'orbit') o.rotation[a.axis || 'y'] += (a.speed || 1) * dt;
      else if (a.type === 'float') { a.t = (a.t || 0) + dt * (a.speed || 1); o.position.y = a.baseY + Math.sin(a.t) * (a.amp || 0.5); }
    });
    if (S.walkActive) walkStep(dt);
    if (S.orbit && !S.walkActive) S.orbit.update();
    S.renderer.render(S.scene, S.camera);
  }

  // ---- geometry factory ----------------------------------------------------
  function num(v, d) { return typeof v === 'number' && isFinite(v) ? v : d; }
  function arr3(a, d) { a = a || []; return [num(a[0], d[0]), num(a[1], d[1]), num(a[2], d[2])]; }

  function geometryFor(spec) {
    var s = spec.size || {};
    var t = (spec.type || 'box').toLowerCase();
    switch (t) {
      case 'box': case 'cube': case 'building': return new THREE.BoxGeometry(num(s.w || s.width, 1), num(s.h || s.height, 1), num(s.d || s.depth, 1));
      case 'sphere': case 'ball': return new THREE.SphereGeometry(num(s.r || s.radius, 0.5), 40, 28);
      case 'cylinder': case 'column': case 'pipe': return new THREE.CylinderGeometry(num(s.rt || s.radiusTop || s.r || s.radius, 0.5), num(s.rb || s.radiusBottom || s.r || s.radius, 0.5), num(s.h || s.height, 1), 40);
      case 'cone': return new THREE.ConeGeometry(num(s.r || s.radius, 0.5), num(s.h || s.height, 1), 40);
      case 'torus': case 'ring': return new THREE.TorusGeometry(num(s.r || s.radius, 0.6), num(s.tube, 0.2), 20, 48);
      case 'plane': case 'slab': case 'floor': return new THREE.BoxGeometry(num(s.w || s.width, 4), num(s.h || s.height, 0.1), num(s.d || s.depth, 4));
      case 'wedge': case 'prism': return prismGeo(num(s.w, 1), num(s.h, 1), num(s.d, 1));
      case 'capsule': return new THREE.CapsuleGeometry(num(s.r || s.radius, 0.4), num(s.h || s.height, 1), 8, 24);
      case 'dodecahedron': return new THREE.DodecahedronGeometry(num(s.r || s.radius, 0.7));
      case 'icosahedron': return new THREE.IcosahedronGeometry(num(s.r || s.radius, 0.7));
      case 'torusknot': return new THREE.TorusKnotGeometry(num(s.r || s.radius, 0.6), num(s.tube, 0.2), 100, 16);
      case 'lathe': return latheGeo(spec);
      case 'extrude': return extrudeGeo(spec);
      default: {
        // Robust: unknown/AI-invented type -> infer a sensible primitive from size keys
        // so a real shape renders instead of a tiny ugly 1x1x1 box.
        if (num(s.tube, 0) > 0) return new THREE.TorusGeometry(num(s.r || s.radius, 0.6), num(s.tube, 0.2), 20, 48);
        if (num(s.rt || s.radiusTop || s.rb || s.radiusBottom, 0) > 0) return new THREE.CylinderGeometry(num(s.rt || s.radiusTop || s.r || s.radius, 0.5), num(s.rb || s.radiusBottom || s.r || s.radius, 0.5), num(s.h || s.height, 1), 40);
        if (num(s.r || s.radius, 0) > 0 && num(s.h || s.height, 0) > 0) return new THREE.CapsuleGeometry(num(s.r || s.radius, 0.4), num(s.h || s.height, 1), 8, 24);
        if (num(s.r || s.radius, 0) > 0) return new THREE.SphereGeometry(num(s.r || s.radius, 0.5), 40, 28);
        var _w = num(s.w || s.width, 0), _h = num(s.h || s.height, 0), _d = num(s.d || s.depth, 0);
        if (_w || _h || _d) return new THREE.BoxGeometry(_w || 1, _h || 1, _d || 1);
        return new THREE.BoxGeometry(1, 1, 1);
      }
    }
  }

  function prismGeo(w, h, d) {
    var shape = new THREE.Shape();
    shape.moveTo(-w / 2, 0); shape.lineTo(w / 2, 0); shape.lineTo(-w / 2, h); shape.lineTo(-w / 2, 0);
    var g = new THREE.ExtrudeGeometry(shape, { depth: d, bevelEnabled: false });
    g.translate(0, 0, -d / 2); return g;
  }
  function latheGeo(spec) {
    var pts = (spec.points || [[0, 0], [0.6, 0], [0.4, 1], [0, 1.2]]).map(function (p) { return new THREE.Vector2(num(p[0], 0), num(p[1], 0)); });
    return new THREE.LatheGeometry(pts, 40);
  }
  function extrudeGeo(spec) {
    var path = spec.path || [[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]];
    var shape = new THREE.Shape();
    path.forEach(function (p, i) { i ? shape.lineTo(num(p[0], 0), num(p[1], 0)) : shape.moveTo(num(p[0], 0), num(p[1], 0)); });
    shape.closePath();
    return new THREE.ExtrudeGeometry(shape, { depth: num((spec.size || {}).d || (spec.size || {}).depth, 1), bevelEnabled: !!spec.bevel, bevelSize: 0.05, bevelThickness: 0.05 });
  }

  function materialFor(spec) {
    var m = spec.material || {};
    var mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(m.color || '#9aa7c7'),
      metalness: num(m.metalness, 0.15),
      roughness: num(m.roughness, 0.75),
      emissive: new THREE.Color(m.emissive || '#000000'),
      wireframe: !!m.wireframe,
      flatShading: !!m.flat
    });
    if (m.opacity != null && m.opacity < 1) { mat.transparent = true; mat.opacity = m.opacity; }
    return mat;
  }

  // ---- CSG -----------------------------------------------------------------
  function brushFrom(spec) {
    var b = new CSG_Brush(geometryFor(spec), materialFor(spec));
    applyTransform(b, spec); b.updateMatrixWorld(true); return b;
  }
  function applyCSG(baseMesh, csgSpec) {
    if (!CSG_Evaluator) { return baseMesh; }
    var ev = new CSG_Evaluator();
    var opMap = { subtract: CSG_SUB, cut: CSG_SUB, hole: CSG_SUB, add: CSG_ADD, union: CSG_ADD, intersect: CSG_INT, intersection: CSG_INT };
    var base = new CSG_Brush(baseMesh.geometry, baseMesh.material);
    base.position.copy(baseMesh.position); base.rotation.copy(baseMesh.rotation); base.scale.copy(baseMesh.scale); base.updateMatrixWorld(true);
    var op = opMap[(csgSpec.operation || 'subtract').toLowerCase()] || CSG_SUB;
    (csgSpec['with'] || csgSpec.tools || []).forEach(function (w) {
      try { base = ev.evaluate(base, brushFrom(w), op); } catch (e) { console.warn('[visualise] CSG op failed', e); }
    });
    base.material = baseMesh.material;
    base.castShadow = true; base.receiveShadow = true;
    return base;
  }

  // ---- transforms ----------------------------------------------------------
  function applyTransform(obj, spec) {
    var p = arr3(spec.position || spec.pos, [0, 0, 0]);
    var r = arr3(spec.rotation || spec.rot, [0, 0, 0]);
    var sc = spec.scale;
    obj.position.set(p[0], p[1], p[2]);
    obj.rotation.set(r[0] * Math.PI / 180, r[1] * Math.PI / 180, r[2] * Math.PI / 180);
    if (typeof sc === 'number') obj.scale.setScalar(sc);
    else if (sc) { var a = arr3(sc, [1, 1, 1]); obj.scale.set(a[0], a[1], a[2]); }
  }

  // ---- build one object (recursive) ---------------------------------------
  function buildObject(spec) {
    var obj;
    if (spec.type === 'group' || spec.children) {
      obj = new THREE.Group();
      applyTransform(obj, spec);
      (spec.children || []).forEach(function (c) { var ch = buildObject(c); if (ch) obj.add(ch); });
      if (spec.type !== 'group') { var self = buildLeaf(spec); if (self) obj.add(self); }
    } else {
      obj = buildLeaf(spec);
    }
    if (!obj) return null;
    obj.name = spec.name || spec.id || spec.type || 'object';
    if (spec.animation) registerAnim(obj, spec.animation);
    return spec.repeat ? applyRepeat(obj, spec.repeat) : obj;
  }

  function buildLeaf(spec) {
    var isBond=(spec.type||'').toLowerCase()==='bond', mesh;
    if(isBond){
      var a=arr3(spec.start,[-.5,0,0]),b=arr3(spec.end,[.5,0,0]),va=new THREE.Vector3(a[0],a[1],a[2]),vb=new THREE.Vector3(b[0],b[1],b[2]),dir=vb.clone().sub(va),len=Math.max(.0001,dir.length());
      mesh=new THREE.Mesh(new THREE.CylinderGeometry(num((spec.size||{}).r||spec.radius,.055),num((spec.size||{}).r||spec.radius,.055),len,num((spec.size||{}).segments,16)),materialFor(spec));
      mesh.position.copy(va.clone().add(vb).multiplyScalar(.5));mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir.normalize());
    }else{ mesh = new THREE.Mesh(geometryFor(spec), materialFor(spec)); applyTransform(mesh, spec); }
    mesh.castShadow = true; mesh.receiveShadow = true;
    if (spec.csg && (spec.csg['with'] || spec.csg.tools)) {
      var r = applyCSG(mesh, spec.csg);
      r.name = mesh.name; return r;
    }
    return mesh;
  }

  function registerAnim(obj, a) {
    S.animated.push({ obj: obj, type: (a.type || 'rotate').toLowerCase(), axis: a.axis || 'y', speed: num(a.speed, 1), amp: num(a.amp, 0.5), baseY: obj.position.y });
  }

  // ---- procedural repeat / instancing (cities!) ---------------------------
  function applyRepeat(obj, rep) {
    var c = arr3(rep.count, [1, 1, 1]).map(function (n) { return Math.max(1, Math.round(n)); });
    var g = arr3(rep.gap, [2, 2, 2]);
    var jitter = num(rep.jitter, 0);
    var vary = rep.vary || null;
    var total = c[0] * c[1] * c[2];
    var ox = (c[0] - 1) * g[0] / 2, oz = (c[2] - 1) * g[2] / 2, oy = 0;

    if (total > 400 && obj.isMesh) {
      var inst = new THREE.InstancedMesh(obj.geometry, obj.material, total);
      inst.castShadow = true; inst.receiveShadow = true;
      var m = new THREE.Matrix4(), q = new THREE.Quaternion(), v = new THREE.Vector3(), sv = new THREE.Vector3(), i = 0;
      for (var x = 0; x < c[0]; x++) for (var y = 0; y < c[1]; y++) for (var z = 0; z < c[2]; z++) {
        var jx = (Math.random() - 0.5) * jitter, jz = (Math.random() - 0.5) * jitter;
        var syv = vary && vary.scaleY ? (vary.scaleY[0] + Math.random() * (vary.scaleY[1] - vary.scaleY[0])) : 1;
        v.set(x * g[0] - ox + jx, oy + (syv - 1) * (obj.geometry.parameters ? (obj.geometry.parameters.height || 1) / 2 : 0), z * g[2] - oz + jz);
        sv.set(1, syv, 1);
        m.compose(v, q, sv); inst.setMatrixAt(i++, m);
      }
      inst.instanceMatrix.needsUpdate = true; inst.name = obj.name + ' x' + total;
      return inst;
    }
    var group = new THREE.Group(); group.name = obj.name + ' array';
    for (var X = 0; X < c[0]; X++) for (var Y = 0; Y < c[1]; Y++) for (var Z = 0; Z < c[2]; Z++) {
      var clone = obj.clone();
      var jX = (Math.random() - 0.5) * jitter, jZ = (Math.random() - 0.5) * jitter;
      clone.position.set(X * g[0] - ox + jX, Y * g[1] + obj.position.y, Z * g[2] - oz + jZ);
      if (vary && vary.scaleY) clone.scale.y *= (vary.scaleY[0] + Math.random() * (vary.scaleY[1] - vary.scaleY[0]));
      group.add(clone);
    }
    return group;
  }

  // ---- build whole spec ----------------------------------------------------
  function buildFromSpec(spec) {
    clearScene();
    S.lastSpec = spec;
    S.units = spec.units || 'm';
    if (spec.background) S.scene.background = new THREE.Color(spec.background);
    var env = spec.environment || {};
    S.grid.visible = env.grid !== false;
    S.ground.visible = env.ground !== false;
    var list = spec.objects || spec.parts || [];
    list.forEach(function (o) {
      try {
        var obj = buildObject(o);
        if (obj) { S.root.add(obj); collectSelectable(obj, o); }
      } catch (e) { console.warn('[visualise] object build failed', o, e); }
    });
    if (spec.camera) {
      var cp = arr3(spec.camera.position, [14, 12, 18]);
      S.camera.position.set(cp[0], cp[1], cp[2]);
      var ct = arr3(spec.camera.target, [0, 2, 0]); S.orbit.target.set(ct[0], ct[1], ct[2]);
    } else frameScene();
    S.built = true;
    refreshOutliner();
  }

  function collectSelectable(obj, spec) {
    obj.userData.spec = spec;
    S.selectable.push(obj);
  }

  function frameScene() {
    var box = new THREE.Box3().setFromObject(S.root);
    if (box.isEmpty()) return;
    var size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    var maxd = Math.max(size.x, size.y, size.z) || 5;
    var dist = maxd * 1.8;
    S.camera.position.set(center.x + dist * 0.7, center.y + dist * 0.6, center.z + dist * 0.9);
    S.orbit.target.copy(center); S.orbit.update();
    S.camera.near = maxd / 200; S.camera.far = maxd * 60; S.camera.updateProjectionMatrix();
  }

  function clearScene() {
    if (!S.root) return;
    detach();
    for (var i = S.root.children.length - 1; i >= 0; i--) disposeObj(S.root.children[i]);
    S.root.clear();
    S.animated = []; S.selectable = []; S.selected = null;
    clearMeasure();
  }
  function disposeObj(o) {
    o.traverse && o.traverse(function (c) { if (c.geometry) c.geometry.dispose(); if (c.material) { [].concat(c.material).forEach(function (m) { m.dispose(); }); } });
  }

  // ---- selection / transform ----------------------------------------------
  function onCanvasClick(e) {
    if (S.walkActive) return;
    var rect = S.renderer.domElement.getBoundingClientRect();
    var mx = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    var my = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    S.raycaster.setFromCamera(new THREE.Vector2(mx, my), S.camera);
    if (S.measureMode) { doMeasure(); return; }
    var hits = S.raycaster.intersectObjects(S.selectable, true);
    if (hits.length) { select(topLevel(hits[0].object)); } else { detach(); }
  }
  function topLevel(o) { while (o.parent && o.parent !== S.root) o = o.parent; return o; }
  function select(obj) {
    S.selected = obj; S.transform.attach(obj); highlightOutliner(obj); syncMaterialEditor(obj);
  }
  function eachMaterial(obj,fn){ if(!obj)return; obj.traverse(function(o){if(!o.material)return; [].concat(o.material).forEach(fn);}); }
  function applySelectedMaterial(key,value){ if(!S.selected)return toast('Select a part first'); eachMaterial(S.selected,function(m){if(key==='color'&&m.color)m.color.set(value);else if(key==='roughness')m.roughness=value;m.needsUpdate=true;}); var sp=S.selected.userData.spec||(S.selected.userData.spec={});sp.material=sp.material||{};sp.material[key]=value; }
  function syncMaterialEditor(obj){var c=document.getElementById('dv-color'),r=document.getElementById('dv-rough'),mat=null;eachMaterial(obj,function(m){if(!mat)mat=m});if(mat&&c&&mat.color)c.value='#'+mat.color.getHexString();if(mat&&r&&mat.roughness!=null)r.value=mat.roughness;}
  function detach() { S.selected = null; if (S.transform) S.transform.detach(); highlightOutliner(null); }
  function setTransformMode(m) { if (S.transform) S.transform.setMode(m); }
  function deleteSelected() { if (!S.selected) return; detach(); disposeObj(S.selected); S.root.remove(S.selected); S.selectable = S.selectable.filter(function (o) { return o !== S.selected; }); refreshOutliner(); }
  function duplicateSelected() {
    if (!S.selected) return;
    var c = S.selected.clone(); c.position.x += 2; S.root.add(c); c.userData.spec = S.selected.userData.spec;
    S.selectable.push(c); select(c); refreshOutliner();
  }

  // ---- outliner ------------------------------------------------------------
  function refreshOutliner() {
    var ol = document.getElementById('dv-outliner'); if (!ol) return;
    ol.innerHTML = '';
    if (!S.selectable.length) { ol.appendChild(el('div', { style: 'color:#5f6a86;padding:10px;font-size:13px', text: 'No objects yet. Type a prompt below.' })); return; }
    S.selectable.forEach(function (o) {
      var node = el('div', { class: 'dv-node' + (o === S.selected ? ' sel' : ''), text: '\u25C8 ' + (o.name || 'object') });
      node.addEventListener('click', function () { select(o); });
      o.userData._node = node; ol.appendChild(node);
    });
  }
  function highlightOutliner(sel) {
    S.selectable.forEach(function (o) { if (o.userData._node) o.userData._node.classList.toggle('sel', o === sel); });
  }

  // ---- measurement ---------------------------------------------------------
  function toggleMeasure(btn) { S.measureMode = !S.measureMode; btn.classList.toggle('on', S.measureMode); if (!S.measureMode) clearMeasure(); }
  function doMeasure() {
    var hits = S.raycaster.intersectObjects(S.selectable.concat(S.ground), true);
    if (!hits.length) return;
    S.measurePts.push(hits[0].point.clone());
    var dot = new THREE.Mesh(new THREE.SphereGeometry(0.12, 12, 12), new THREE.MeshBasicMaterial({ color: 0xe8623d }));
    dot.position.copy(hits[0].point); S.scene.add(dot); S.measureObjs.push(dot);
    if (S.measurePts.length === 2) {
      var a = S.measurePts[0], b = S.measurePts[1];
      var line = new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), new THREE.LineBasicMaterial({ color: 0xe8623d }));
      S.scene.add(line); S.measureObjs.push(line);
      var d = a.distanceTo(b);
      var lbl = makeLabel(d.toFixed(2) + ' ' + S.units); lbl.position.copy(a.clone().lerp(b, 0.5)); lbl.position.y += 0.4;
      S.scene.add(lbl); S.measureObjs.push(lbl);
      S.measurePts = [];
    }
  }
  function makeLabel(text) {
    var cv = document.createElement('canvas'); cv.width = 256; cv.height = 64;
    var ctx = cv.getContext('2d'); ctx.fillStyle = 'rgba(232,98,61,.92)'; ctx.fillRect(0, 0, 256, 64);
    ctx.fillStyle = '#fff'; ctx.font = 'bold 30px Inter,sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(text, 128, 34);
    var tex = new THREE.CanvasTexture(cv);
    var sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, depthTest: false }));
    sp.scale.set(2.4, 0.6, 1); return sp;
  }
  function clearMeasure() { S.measureObjs.forEach(function (o) { S.scene.remove(o); disposeObj(o); }); S.measureObjs = []; S.measurePts = []; }

  // ---- walkthrough ---------------------------------------------------------
  function toggleWalk(btn) { if (S.walkActive) endWalk(); else startWalk(); btn.classList.toggle('on', S.walkActive); }
  function startWalk() {
    S.walkActive = true; S.orbit.enabled = false;
    S.camera.position.y = 1.7;
    if (isTouch()) document.getElementById('dv-pad').classList.add('show');
    else if (S.walk) { try { S.walk.lock(); } catch (e) {} }
    document.getElementById('dv-hud').innerHTML = 'Walkthrough: WASD / arrows to move &middot; drag to look &middot; press Walk again to exit';
  }
  function endWalk() {
    S.walkActive = false; S.orbit.enabled = true;
    document.getElementById('dv-pad').classList.remove('show');
    if (S.walk && S.walk.isLocked) { try { S.walk.unlock(); } catch (e) {} }
    var b = [].slice.call(document.querySelectorAll('#dv-top .dv-btn')).find(function (x) { return x.textContent.indexOf('Walk') >= 0; });
    if (b) b.classList.remove('on');
  }
  function walkStep(dt) {
    var sp = 12 * dt, m = S.move;
    var dir = new THREE.Vector3();
    S.camera.getWorldDirection(dir); dir.y = 0; dir.normalize();
    var right = new THREE.Vector3().crossVectors(dir, new THREE.Vector3(0, 1, 0)).normalize();
    if (m.f) S.camera.position.addScaledVector(dir, sp);
    if (m.b) S.camera.position.addScaledVector(dir, -sp);
    if (m.l) S.camera.position.addScaledVector(right, -sp);
    if (m.r) S.camera.position.addScaledVector(right, sp);
  }

  // ---- grid / spin ---------------------------------------------------------
  function toggleGrid(btn) { S.grid.visible = !S.grid.visible; S.axes.visible = S.grid.visible; btn.classList.toggle('on', S.grid.visible); }
  function toggleAutoRotate(btn) { S.orbit.autoRotate = !S.orbit.autoRotate; S.orbit.autoRotateSpeed = 1.2; btn.classList.toggle('on', S.orbit.autoRotate); }

  // ---- external GLB loader (TRELLIS / image->3D output) --------------------
  async function loadExternalGLB(url) {
    await open();            // build + open the studio (also mounts canvas)
    await loadLibs();
    setupThree();
    if (!GLTFLoader) { toast('GLB loader unavailable'); return; }
    showStatus('Loading photoreal 3D model…', true);
    clearScene();
    return new Promise(function (resolve, reject) {
      new GLTFLoader().load(url, function (gltf) {
        var obj = gltf.scene || (gltf.scenes && gltf.scenes[0]);
        if (!obj) { hideStatus(); toast('Empty GLB'); return reject(new Error('empty glb')); }
        obj.name = 'trellis-model';
        obj.traverse(function (o) { if (o.isMesh) { o.castShadow = true; o.receiveShadow = true; if (o.material) o.material.needsUpdate = true; } });
        S.root.add(obj);
        S.selectable.push(obj);
        S.built = true; S.lastSpec = { name: 'trellis-model' };
        try { if (typeof refreshOutliner === 'function') refreshOutliner(); } catch (e) {}
        try { frameScene(); } catch (e) {}
        hideStatus();
        toast('Photoreal 3D loaded');
        resolve(obj);
      }, undefined, function (err) {
        hideStatus(); toast('GLB load failed'); reject(err);
      });
    });
  }
  window.__damruLoadGLB = loadExternalGLB;

  // prompt -> photoreal textured 3D (backend /model3d -> image -> TRELLIS -> GLB)
  async function runPhotoreal() {
    var inp = document.getElementById('dv-prompt-in');
    var v = (inp && inp.value || '').trim();
    if (!v) { toast('Type a prompt first'); return; }
    var btn = document.getElementById('dv-photoreal'); if (btn) btn.disabled = true;
    try {
      if (typeof window.__damruPromptTo3D !== 'function') throw new Error('bridge missing (update index.html)');
      try { await open(); } catch (e0) {}
      showStatus('Photoreal 3D ban raha hai\u2026 AI image \u2192 real 3D (~1\u20132 min)', true);
      await window.__damruPromptTo3D(v);
    } catch (e) {
      console.warn('photoreal fail -> parametric fallback', e);
      hideStatus();
      // JUGAAD: free 3D server busy/asleep? auto-fall back to the always-free parametric engine.
      toast('Photoreal server busy \u2014 building parametric 3D instead');
      try { if (typeof runPrompt === 'function') await runPrompt(); }
      catch (e2) { hideStatus(); toast('3D build failed: ' + (e2 && e2.message || e2)); }
    } finally { if (btn) btn.disabled = false; }
  }

  // ---- export --------------------------------------------------------------
  function exportGLTF() {
    if (!S.built) return toast('Nothing to export yet');
    new GLTFExporter().parse(S.root, function (res) {
      var blob = new Blob([res], { type: 'model/gltf-binary' });
      download(blob, (S.lastSpec && S.lastSpec.name || 'damru-model') + '.glb');
    }, function (e) { toast('GLB export failed'); }, { binary: true });
  }
  function getSTLBlob() {
    if (!S.built || !STLExporter) return null;
    // Visualise is Y-up; slicers are Z-up. Export a clone rotated +90° on X
    // so the manufactured part has the same upright orientation seen on screen.
    var exportRoot = S.root.clone(true);
    exportRoot.rotation.x += Math.PI / 2;
    exportRoot.updateMatrixWorld(true);
    var stl = new STLExporter().parse(exportRoot, { binary: false });
    return new Blob([stl], { type: 'model/stl' });
  }
  function exportSTL() {
    var blob = getSTLBlob();
    if (!blob) return toast('Nothing to export yet');
    download(blob, (S.lastSpec && S.lastSpec.name || 'damru-model') + '.stl');
  }
  function download(blob, name) {
    var a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; a.click();
    setTimeout(function () { URL.revokeObjectURL(a.href); }, 4000);
  }
  function toast(m) { showStatus(m, false); setTimeout(hideStatus, 1600); }

  // ---- keyboard ------------------------------------------------------------
  function onKey(e) {
    if (!document.getElementById('dv-modal').classList.contains('dv-open')) return;
    if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
    if (S.walkActive) {
      if (e.key === 'w' || e.key === 'ArrowUp') S.move.f = 1; if (e.key === 's' || e.key === 'ArrowDown') S.move.b = 1;
      if (e.key === 'a' || e.key === 'ArrowLeft') S.move.l = 1; if (e.key === 'd' || e.key === 'ArrowRight') S.move.r = 1;
    }
    switch (e.key.toLowerCase()) {
      case 'g': setTransformMode('translate'); break;
      case 'r': setTransformMode('rotate'); break;
      case 's': if (!S.walkActive) setTransformMode('scale'); break;
      case 'x': case 'y': case 'z': if (S.transform) { S.transform.showX = e.key === 'x'; S.transform.showY = e.key === 'y'; S.transform.showZ = e.key === 'z'; } break;
      case 'delete': case 'backspace': deleteSelected(); break;
      case 'escape': if (S.walkActive) endWalk(); else if (S.transform) { S.transform.showX = S.transform.showY = S.transform.showZ = true; } break;
    }
  }
  window.addEventListener('keyup', function (e) {
    if (!S.walkActive) return;
    if (e.key === 'w' || e.key === 'ArrowUp') S.move.f = 0; if (e.key === 's' || e.key === 'ArrowDown') S.move.b = 0;
    if (e.key === 'a' || e.key === 'ArrowLeft') S.move.l = 0; if (e.key === 'd' || e.key === 'ArrowRight') S.move.r = 0;
  });

  // ---- AI generation -------------------------------------------------------
  var SCHEMA = [
    'You are DAMRU-CAD, a world-class 3D/CAD design engine. Convert the user request into a STRICT JSON scene-spec.',
    'Output ONLY one JSON object in a ```json fenced block. No prose outside it.',
    'SCHEMA:',
    '{',
    '  "name": string, "units": "m", "background": "#hex"(optional),',
    '  "environment": {"ground": bool, "grid": bool},',
    '  "camera": {"position":[x,y,z], "target":[x,y,z]} (optional),',
    '  "objects": [ Object, ... ]',
    '}',
    'Object = {',
    '  "id": string, "name": string,',
    '  "type": one of box|sphere|cylinder|cone|torus|plane|wedge|capsule|lathe|extrude|dodecahedron|icosahedron|torusknot|group,',
    '  "size": { box:{w,h,d}, sphere:{r}, cylinder:{rt,rb,h}, cone:{r,h}, torus:{r,tube}, plane:{w,h,d}, capsule:{r,h} },',
    '  "position":[x,y,z], "rotation":[degX,degY,degZ], "scale": number|[x,y,z],',
    '  "material": {"color":"#hex","metalness":0..1,"roughness":0..1,"opacity":0..1,"emissive":"#hex","wireframe":bool},',
    '  "csg": {"operation":"subtract|union|intersect", "with":[Object,...]} (boolean CAD e.g. windows/holes),',
    '  "children": [Object,...] (for type group),',
    '  "repeat": {"count":[nx,ny,nz], "gap":[dx,dy,dz], "jitter":n, "vary":{"scaleY":[min,max]}} (arrays/cities),',
    '  "animation": {"type":"rotate|float","axis":"y","speed":n,"amp":n}',
    '}',
    'RULES: Ground plane is y=0, up is +Y. Place objects ABOVE ground (positive Y for their center). Use real proportions in metres.',
    'For holes/windows/doors use csg subtract. For repeated structures (city blocks, columns, fences, windows grid) use repeat.',
    'Build rich, complete, realistic scenes. Never represent a vehicle, spacecraft, machine or building as one primitive. Use named assemblies and visible functional subsystems. For spacecraft require hull, nose, propulsion, control surfaces, TPS, avionics, RCS and landing/launch hardware. A city = roads + repeated blocks + landmarks. Keep meshes sensible (<5000).'
  ].join('\n');

  async function llm(messages, mt) {
    if (typeof window.__damruLLM === 'function') return await window.__damruLLM(messages, mt);
    if (typeof window.engine === 'function') return await window.engine(messages, mt || 6000, 0.35, 'code');
    throw new Error('No LLM bridge (window.__damruLLM) found');
  }

  async function refinePrompt(raw) {
    try {
      var msgs = [
        { role: 'system', content: 'You are Damru multidisciplinary 3D architect across industrial design, mechanics, architecture, aerospace, physics, chemistry and biology. Expand the request into a precise buildable scene brief: classify the domain, list semantic parts, approximate dimensions in appropriate units, materials/colors, topology, spatial relationships, governing constraints and simulation assumptions. Be concrete and compact (bullet points). Do NOT output JSON.' },
        { role: 'user', content: raw }
      ];
      var brief = await llm(msgs, 1400);
      return (brief && brief.trim().length > 20) ? (raw + '\n\nDESIGN BRIEF:\n' + brief.trim()) : raw;
    } catch (e) { return raw; }
  }

  function extractJSON(text) {
    if (!text) throw new Error('empty response');
    var m = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
    var body = m ? m[1] : text;
    var s = body.indexOf('{'), e = body.lastIndexOf('}');
    if (s < 0 || e < 0) throw new Error('no JSON object found');
    var raw = body.slice(s, e + 1).replace(/,\s*([}\]])/g, '$1');
    return JSON.parse(raw);
  }

  function withTimeout(promise, ms, label) {
    var timer;
    return Promise.race([promise, new Promise(function (_, reject) {
      timer = setTimeout(function () { reject(new Error((label || 'operation') + ' timeout')); }, ms);
    })]).finally(function () { clearTimeout(timer); });
  }

  async function webResearch(prompt) {
    var key = 'damru-cad-research:' + (prompt || '').trim().toLowerCase().slice(0, 180);
    try {
      var cached = JSON.parse(sessionStorage.getItem(key) || 'null');
      if (cached && Date.now() - cached.at < 3600000) return cached.results || [];
    } catch (e) {}
    try {
      if (typeof window.__damruWebSearch !== 'function') return [];
      var query = prompt + ' real dimensions material engineering manufacturable CAD design';
      var data = await withTimeout(window.__damruWebSearch(query, 5), 6500, 'web research');
      var results = Array.isArray(data) ? data.slice(0, 5) : ((data && data.results) || []).slice(0, 5);
      try { sessionStorage.setItem(key, JSON.stringify({ at: Date.now(), results: results })); } catch (e2) {}
      return results;
    } catch (e) { return []; }
  }

  function isSpaceVehiclePrompt(prompt) {
    return /star\s*ship|spaceship|space\s*ship|spacecraft|rocket|launch\s*vehicle|orbital\s*vehicle/.test((prompt || '').toLowerCase());
  }

  // Deterministic aerospace compiler: a rich model exists even when web/LLM is offline.
  function starshipSpec(prompt) {
    var O = [];
    var steel = { color:'#b9c2c9', metalness:.88, roughness:.23 };
    var steel2 = { color:'#7f8b94', metalness:.78, roughness:.31 };
    var black = { color:'#141920', metalness:.35, roughness:.72 };
    var tile = { color:'#20262d', metalness:.10, roughness:.92 };
    var glass = { color:'#071b2d', metalness:.55, roughness:.12, emissive:'#0b426b', opacity:.92 };
    var hot = { color:'#ff7a1a', metalness:.05, roughness:.28, emissive:'#ff3b00', opacity:.78 };
    var blue = { color:'#2bd9ff', metalness:.08, roughness:.18, emissive:'#007ea8', opacity:.72 };
    var pad = { color:'#3d4652', metalness:.42, roughness:.68 };
    function add(id,type,size,pos,mat,rot,extra) {
      var q={id:id,name:id,type:type,size:size||{},position:pos||[0,0,0],rotation:rot||[0,0,0],material:mat||steel};
      if(extra) for(var k in extra) q[k]=extra[k]; O.push(q); return q;
    }
    function box(id,w,h,d,pos,mat,rot){return add(id,'box',{w:w,h:h,d:d},pos,mat,rot);}
    function cyl(id,rt,rb,h,pos,mat,rot,extra){return add(id,'cylinder',{rt:rt,rb:rb,h:h},pos,mat,rot,extra);}
    function ring(id,r,t,y,mat){return add(id,'torus',{r:r,tube:t},[0,y,0],mat,[90,0,0]);}

    // Orbital launch mount and flame trench.
    box('launch-pad',22,.55,22,[0,.275,0],pad);
    box('flame-trench',7,.16,19,[0,.62,0],black);
    ring('launch-mount-ring',4.8,.48,1.15,steel2);
    for(var pi=0;pi<6;pi++) {
      var pa=pi*Math.PI/3, px=Math.cos(pa)*4.8, pz=Math.sin(pa)*4.8;
      box('mount-pillar-'+(pi+1),.62,2.3,.62,[px,1.55,pz],steel2,[0,-pi*60,0]);
    }

    // Pressure vessel, aft skirt and aerodynamic nose.
    cyl('aft-engine-skirt',3.35,3.62,3.2,[0,5.1,0],steel2);
    cyl('primary-pressure-hull',3.05,3.18,24,[0,18.7,0],steel);
    add('ogive-nose','lathe',{},[0,30.7,0],steel,[0,0,0],{points:[[3.05,0],[3.08,.8],[2.92,2.2],[2.55,4.3],[1.95,6.1],[1.18,7.7],[.42,8.8],[0,9.15]]});
    ring('aft-weld-ring',3.28,.10,6.75,steel2);
    [9.2,12.2,15.2,18.2,21.2,24.2,27.2,30.1].forEach(function(y,i){ring('hull-seam-'+(i+1),3.07,.045,y,steel2);});

    // Windward thermal protection system: visible tessellated heat-shield spine.
    for(var tr=0;tr<10;tr++) for(var tc=-2;tc<=2;tc++) {
      var ty=8.2+tr*2.05, tx=tc*.78+(tr%2?.39:0);
      if(Math.abs(tx)<2.25) box('heat-tile-'+tr+'-'+(tc+2),.69,1.74,.105,[tx,ty,3.055],tile,[0,0,0]);
    }

    // Six forward observation windows with emissive cockpit interior.
    [[-1.15,34.15,2.25],[-.38,34.42,2.55],[.38,34.42,2.55],[1.15,34.15,2.25],[-.72,35.25,2.08],[.72,35.25,2.08]].forEach(function(v,i){
      box('flight-window-'+(i+1),.62,.48,.12,v,glass,[i<4?0:7,0,0]);
    });

    // Four actuated aerodynamic flaps, deliberately separate for transforms/animation.
    add('forward-flap-port','wedge',{w:2.0,h:5.7,d:.38},[-3.35,27.0,0],black,[0,90,-8]);
    add('forward-flap-starboard','wedge',{w:2.0,h:5.7,d:.38},[3.35,27.0,0],black,[0,-90,8]);
    add('aft-flap-port','wedge',{w:2.7,h:7.0,d:.50},[-3.48,7.7,0],black,[0,90,-12]);
    add('aft-flap-starboard','wedge',{w:2.7,h:7.0,d:.50},[3.48,7.7,0],black,[0,-90,12]);
    box('dorsal-stabilizer',.36,5.2,2.3,[0,27.2,-3.0],black,[0,0,0]);

    // RCS pods, navigation lights and dorsal avionics.
    [[-2.45,31.1,1.9],[2.45,31.1,1.9],[-2.45,31.1,-1.9],[2.45,31.1,-1.9]].forEach(function(v,i){
      cyl('rcs-pod-'+(i+1),.25,.34,1.05,v,steel2,[90,0,0]);
      cyl('rcs-nozzle-'+(i+1),.11,.19,.42,[v[0],v[1],v[2]+(v[2]>0?.65:-.65)],black,[90,0,0]);
    });
    add('nav-light-port','sphere',{r:.16},[-3.18,28.9,0],{color:'#ff284c',emissive:'#ff002f',roughness:.15});
    add('nav-light-starboard','sphere',{r:.16},[3.18,28.9,0],{color:'#4cff8c',emissive:'#00ff55',roughness:.15});
    box('avionics-spine',.42,10.5,.35,[0,21.4,-3.02],steel2);

    // Seven-engine cluster (one center + six ring), throat glow and exhaust plumes.
    var engines=[[0,0]];
    for(var ei=0;ei<6;ei++){var ea=ei*Math.PI/3;engines.push([Math.cos(ea)*1.62,Math.sin(ea)*1.62]);}
    engines.forEach(function(v,i){
      cyl('raptor-nozzle-'+(i+1),.40,.73,1.55,[v[0],3.15,v[1]],black);
      ring('engine-throat-ring-'+(i+1),.39,.085,3.58,hot).position=[v[0],3.58,v[1]];
      add('engine-plume-'+(i+1),'cone',{r:.48,h:2.6},[v[0],1.72,v[1]],i===0?blue:hot,[0,0,180],{animation:{type:'float',axis:'y',speed:3.2+i*.12,amp:.12}});
    });

    // Deployable landing legs and foot pads.
    [[-2.8,0,-2.8],[2.8,0,-2.8],[-2.8,0,2.8],[2.8,0,2.8]].forEach(function(v,i){
      box('landing-strut-'+(i+1),.34,4.2,.34,[v[0],4.35,v[2]],steel2,[i<2?-13:13,0,i%2?-13:13]);
      box('landing-foot-'+(i+1),1.15,.22,.88,[v[0]*1.12,2.18,v[2]*1.12],pad);
    });

    // Service tower gives scale and a believable launch-site scene.
    box('tower-foundation',5,.7,7,[-9,.35,0],pad);
    [-10.5,-7.5].forEach(function(x,ix){[-2.3,2.3].forEach(function(z,iz){box('tower-column-'+ix+'-'+iz,.42,37,.42,[x,18.8,z],steel2);});});
    for(var lv=0;lv<9;lv++) {
      var ly=2.6+lv*4.15;
      box('tower-deck-'+lv,4.1,.24,5.2,[-9,ly,0],pad);
      box('tower-brace-a-'+lv,.20,4.35,6.0,[-9,ly+1.8,0],steel2,[42,0,0]);
      box('tower-brace-b-'+lv,.20,4.35,6.0,[-9,ly+1.8,0],steel2,[-42,0,0]);
    }
    box('crew-arm',6.4,.38,1.15,[-4.8,31.8,0],steel2,[0,0,-4]);
    add('tracking-dish','sphere',{r:.72},[-9,39.0,0],steel2,[0,0,0]);

    return {
      name:'DAMRU-Starship-Mk-I', units:'m', background:'#050811',
      environment:{ground:true,grid:true},
      camera:{position:[31,26,48],target:[-1.2,19,0]},
      metadata:{generator:'Damru Aerospace Compiler',designIntent:(prompt||'orbital starship'),minimumAssembly:true},
      objects:O
    };
  }


  function isDNAPrompt(p){return /\bdna\b|double\s*helix|deoxyribonucleic|genome/.test((p||'').toLowerCase());}
  function isMoleculePrompt(p){return /molecule|molecular|\batom\b|benzene|methane|water\s*(?:h2o)?|carbon\s*dioxide|co2|caffeine/.test((p||'').toLowerCase());}
  function isProteinPrompt(p){return /protein|peptide|amino\s*acid|alpha\s*helix/.test((p||'').toLowerCase());}
  function bondSpec(id,a,b,color,r,kind){return{id:id,name:id,type:'bond',start:a,end:b,size:{r:r||.055,segments:16},material:{color:color||'#aab5c6',metalness:.05,roughness:.55},metadata:{bondType:kind||'covalent'}};}
  function dnaSpec(prompt){
    var O=[],N=28,R=3.35,rise=.56,turn=10.5,seq='ATGCGTACGATTGCCATGCAATCGGTAC';
    var C={phosphate:'#ff9d2e',sugar:'#e8e5d8',A:'#42d67d',T:'#f4cf4f',G:'#6aa8ff',C:'#ef6477',bond:'#aab7ca',hydrogen:'#62d5ef'};
    function sphere(id,r,pos,color,role){O.push({id:id,name:id,type:'sphere',size:{r:r},position:pos,material:{color:color,metalness:.04,roughness:.48},metadata:{domain:'biology',role:role}});}
    var prevA=null,prevB=null;
    for(var i=0;i<N;i++){
      var a=i*Math.PI*2/turn,y=.7+i*rise,pa=[Math.cos(a)*R,y,Math.sin(a)*R],pb=[-pa[0],y,-pa[2]],sa=[Math.cos(a)*R*.82,y,Math.sin(a)*R*.82],sb=[-sa[0],y,-sa[2]],ba=[Math.cos(a)*R*.48,y,Math.sin(a)*R*.48],bb=[-ba[0],y,-ba[2]];
      var A=seq[i%seq.length],B=A==='A'?'T':A==='T'?'A':A==='G'?'C':'G';
      sphere('strand-A-phosphate-'+(i+1),.18,pa,C.phosphate,'phosphate');sphere('strand-B-phosphate-'+(i+1),.18,pb,C.phosphate,'phosphate');
      sphere('strand-A-sugar-'+(i+1),.22,sa,C.sugar,'deoxyribose');sphere('strand-B-sugar-'+(i+1),.22,sb,C.sugar,'deoxyribose');
      sphere('base-'+A+'-'+(i+1),.28,ba,C[A],'nucleobase-'+A);sphere('base-'+B+'-'+(i+1),.28,bb,C[B],'nucleobase-'+B);
      O.push(bondSpec('A-phosphate-sugar-'+(i+1),pa,sa,C.bond,.065,'phosphodiester'),bondSpec('A-sugar-base-'+(i+1),sa,ba,C[A],.075,'glycosidic'),bondSpec('B-phosphate-sugar-'+(i+1),pb,sb,C.bond,.065,'phosphodiester'),bondSpec('B-sugar-base-'+(i+1),sb,bb,C[B],.075,'glycosidic'));
      var hb=A==='G'||A==='C'?3:2;for(var h=0;h<hb;h++){var off=(h-(hb-1)/2)*.15;O.push(bondSpec('hydrogen-'+A+B+'-'+(i+1)+'-'+(h+1),[ba[0],y+off,ba[2]],[bb[0],y+off,bb[2]],C.hydrogen,.025,'hydrogen'));}
      if(prevA){O.push(bondSpec('backbone-A-'+i,prevA,pa,C.phosphate,.085,'phosphodiester'),bondSpec('backbone-B-'+i,prevB,pb,C.phosphate,.085,'phosphodiester'));}
      prevA=pa;prevB=pb;
    }
    return{name:'DAMRU-B-DNA-Double-Helix',units:'nm',background:'#050813',environment:{ground:false,grid:true},camera:{position:[19,11,24],target:[0,8,0]},metadata:{generator:'Damru BioGeometry Compiler',structure:'B-DNA',basePairs:N,basePairsPerTurn:10.5,riseAngstrom:3.4,educationalCoarseGrain:true,prompt:prompt},simulation:{domain:'biology',temperatureK:310,bondModel:'coarse harmonic'},objects:O};
  }
  function moleculeSpec(prompt){
    var x=(prompt||'').toLowerCase(),O=[],atoms=[],bonds=[];var col={H:'#f4f6ff',C:'#444c5c',O:'#ef4d5b',N:'#4c79e8',S:'#f4d13b'};
    function atom(id,e,pos,r){O.push({id:id,name:e+' atom '+id,type:'sphere',size:{r:r||({H:.32,C:.48,O:.46,N:.46,S:.55}[e]||.45)},position:pos,material:{color:col[e]||'#c987e8',metalness:.02,roughness:.42},metadata:{domain:'chemistry',element:e}});atoms.push({id:id,element:e,position:pos});}
    function bond(id,a,b,order){O.push(bondSpec(id,a,b,order===2?'#e1b45a':'#aeb8c8',order===2?.09:.07,order===2?'double':'single'));bonds.push({id:id,order:order||1});}
    if(/water|h2o/.test(x)){atom('O1','O',[0,2,0]);atom('H1','H',[-1.15,1.25,0]);atom('H2','H',[1.15,1.25,0]);bond('O-H1',[0,2,0],[-1.15,1.25,0]);bond('O-H2',[0,2,0],[1.15,1.25,0]);}
    else if(/carbon\s*dioxide|co2/.test(x)){atom('C1','C',[0,2,0]);atom('O1','O',[-1.7,2,0]);atom('O2','O',[1.7,2,0]);bond('C-O1',[0,2,0],[-1.7,2,0],2);bond('C-O2',[0,2,0],[1.7,2,0],2);}
    else if(/benzene/.test(x)){for(var i=0;i<6;i++){var a=i*Math.PI/3,p=[Math.cos(a)*2,2,Math.sin(a)*2],hp=[Math.cos(a)*3.05,2,Math.sin(a)*3.05];atom('C'+(i+1),'C',p);atom('H'+(i+1),'H',hp);bond('C-H'+(i+1),p,hp);var j=(i+1)%6,q=[Math.cos(j*Math.PI/3)*2,2,Math.sin(j*Math.PI/3)*2];bond('ring-'+(i+1),p,q,i%2?1:2);}}
    else {atom('C1','C',[0,2,0]);var pts=[[-1.35,3.15,1.1],[1.35,3.15,1.1],[-1.35,.85,-1.1],[1.35,.85,-1.1]];pts.forEach(function(p,i){atom('H'+(i+1),'H',p);bond('C-H'+(i+1),[0,2,0],p);});}
    return{name:'DAMRU-Molecule',units:'angstrom',background:'#050914',environment:{ground:false,grid:true},camera:{position:[10,8,13],target:[0,2,0]},metadata:{generator:'Damru Molecular Geometry Compiler',representation:'ball-and-stick',educationalCoarseGrain:true,prompt:prompt,atoms:atoms,bonds:bonds},simulation:{domain:'chemistry',forceModel:'coarse harmonic + thermal'},objects:O};
  }
  function proteinSpec(prompt){var O=[],N=52,prev=null;for(var i=0;i<N;i++){var a=i*1.745,p=[Math.cos(a)*2.1,.7+i*.34,Math.sin(a)*2.1],side=[Math.cos(a)*3.05,.7+i*.34,Math.sin(a)*3.05],c=i%3===0?'#56d692':i%3===1?'#6b9cff':'#d986e8';O.push({id:'residue-'+(i+1),name:'amino-acid-'+(i+1),type:'sphere',size:{r:.25},position:p,material:{color:c,roughness:.48},metadata:{domain:'biology',role:'alpha-carbon'}},{id:'side-chain-'+(i+1),name:'side-chain-'+(i+1),type:'sphere',size:{r:.18+(i%4)*.025},position:side,material:{color:'#f2b45e',roughness:.5},metadata:{domain:'biology',role:'side-chain'}},bondSpec('side-bond-'+(i+1),p,side,'#aeb8ca',.045,'covalent'));if(prev)O.push(bondSpec('peptide-'+i,prev,p,'#e7e9ed',.065,'peptide'));prev=p;}return{name:'DAMRU-Protein-Alpha-Helix',units:'angstrom',background:'#050813',environment:{ground:false,grid:true},camera:{position:[17,12,22],target:[0,9,0]},metadata:{generator:'Damru Protein Geometry Compiler',representation:'coarse alpha helix',residues:N,notFoldPrediction:true,prompt:prompt},simulation:{domain:'biology',temperatureK:310,bondModel:'coarse harmonic'},objects:O};}
  // ---- ORGANIC SCULPTURE COMPILER (Damru native, free-forever, always builds) ----
  function materialFromPrompt(prompt){
    var x=(prompt||'').toLowerCase();
    if(/gold|golden|swarn/.test(x)) return {color:'#ffcf47',metalness:.95,roughness:.22};
    if(/bronze|copper|brass/.test(x)) return {color:'#b3702a',metalness:.9,roughness:.35};
    if(/silver|steel|chrome|platinum/.test(x)) return {color:'#cfd6de',metalness:.95,roughness:.2};
    if(/marble|granite|stone/.test(x)) return {color:'#eae6df',metalness:.03,roughness:.5};
    if(/jade|emerald/.test(x)) return {color:'#3fae7a',metalness:.2,roughness:.35};
    if(/obsidian|onyx|black/.test(x)) return {color:'#20242c',metalness:.45,roughness:.4};
    return {color:'#c9a24a',metalness:.55,roughness:.4};
  }
  function isOrganicPrompt(p){return /statue|sculpture|idol|murti|murthi|deity|goddess|dragon|serpent|naga|creature|monster|beast|dinosaur|dino|lion|tiger|horse|elephant|eagle|griffin|phoenix|unicorn|gargoyle|figurine|totem|angel|demon|warrior|snake|fish|bird/.test((p||'').toLowerCase());}
  function creatureSpec(prompt){
    var M=materialFromPrompt(prompt), x=(prompt||'').toLowerCase();
    var winged=/dragon|phoenix|griffin|angel|eagle|bird|wing/.test(x);
    var serpent=/dragon|serpent|snake|naga/.test(x);
    var base={color:'#8a8f98',metalness:.15,roughness:.78}, eyeMat={color:'#141821',metalness:.25,roughness:.3};
    var O=[], path=[];
    function P(id,type,size,pos,rot,mat){O.push({id:id,name:id,type:type,size:size||{},position:pos||[0,0,0],rotation:rot||[0,0,0],material:mat||M});}
    function link(id,a,b,r,mat){O.push({id:id,name:id,type:'bond',start:a,end:b,size:{r:r||.2,segments:16},material:mat||M});}
    P('pedestal-base','cylinder',{rt:3.1,rb:3.6,h:1.0},[0,.5,0],[0,0,0],base);
    P('pedestal-top','cylinder',{rt:2.7,rb:3.0,h:.45},[0,1.22,0],[0,0,0],base);
    P('pedestal-trim','torus',{r:2.8,tube:.16},[0,1.45,0],[90,0,0],M);
    var y0=2.0, segs=7, i;
    for(i=0;i<segs;i++){
      var t=i/(segs-1), r=1.3-0.72*t;
      var px=Math.sin(t*Math.PI*0.85)*1.7-0.4, py=y0+1.3+t*2.6, pz=serpent?Math.sin(t*Math.PI*1.5)*0.7:0;
      path.push([px,py,pz,r]);
      P('body-'+(i+1),'sphere',{r:r},[px,py,pz],[0,0,0],M);
      if(i>0) link('spine-'+i,[path[i-1][0],path[i-1][1],path[i-1][2]],[px,py,pz],r*0.8);
      P('spike-'+(i+1),'cone',{r:0.18*(1.1-t),h:0.6*(1.1-t)},[px,py+r+0.15,pz],[0,0,0],M);
    }
    var hh=path[segs-1], nx=hh[0]+0.7, ny=hh[1]+0.7, nz=hh[2];
    link('neck',[hh[0],hh[1],hh[2]],[nx,ny,nz],0.5);
    P('head','sphere',{r:0.8},[nx,ny,nz],[0,0,0],M);
    P('snout','cone',{r:0.46,h:1.15},[nx+0.85,ny-0.05,nz],[0,0,-72],M);
    P('eye-L','sphere',{r:0.13},[nx+0.42,ny+0.3,nz+0.42],[0,0,0],eyeMat);
    P('eye-R','sphere',{r:0.13},[nx+0.42,ny+0.3,nz-0.42],[0,0,0],eyeMat);
    P('horn-L','cone',{r:0.15,h:1.0},[nx-0.2,ny+0.85,nz+0.35],[25,0,18],M);
    P('horn-R','cone',{r:0.15,h:1.0},[nx-0.2,ny+0.85,nz-0.35],[-25,0,18],M);
    [[1.0,0.95],[1.0,-0.95],[-0.7,0.95],[-0.7,-0.95]].forEach(function(c,li){
      var lx=c[0],lz=c[1],hipY=y0+1.15;
      P('hip-'+(li+1),'sphere',{r:0.5},[lx,hipY,lz],[0,0,0],M);
      link('thigh-'+(li+1),[lx,hipY,lz],[lx+0.25,hipY-1.1,lz*1.05],0.4);
      P('knee-'+(li+1),'sphere',{r:0.34},[lx+0.25,hipY-1.1,lz*1.05],[0,0,0],M);
      link('shin-'+(li+1),[lx+0.25,hipY-1.1,lz*1.05],[lx+0.35,y0-0.15,lz*1.08],0.3);
      P('foot-'+(li+1),'sphere',{r:0.4},[lx+0.4,y0-0.2,lz*1.08],[0,0,0],M);
    });
    var tx=path[0][0]-0.6, ty=path[0][1]-0.2, tz=path[0][2], prev=[path[0][0],path[0][1],path[0][2]], k;
    for(k=0;k<8;k++){
      var ang=k*0.55; tx-=0.55; ty+=Math.sin(ang)*0.28-0.02*k; tz+=Math.cos(ang)*0.28;
      var pr=Math.max(0.08,0.55-0.06*k), cur=[tx,Math.max(y0-0.1,ty),tz];
      P('tail-'+(k+1),'sphere',{r:pr},cur,[0,0,0],M);
      link('tail-link-'+(k+1),prev,cur,pr*0.8); prev=cur;
    }
    P('tail-tip','cone',{r:0.2,h:0.9},[prev[0]-0.4,prev[1],prev[2]],[0,0,80],M);
    if(winged){
      [1,-1].forEach(function(s,wi){
        var wy=path[Math.floor(segs*0.6)][1]+0.6, w;
        for(w=0;w<4;w++){ link('wing-rib-'+(wi+1)+'-'+(w+1),[-0.2,wy,s*0.5],[-0.2-w*0.4,wy+2.2-w*0.3,s*(1.6+w*0.9)],0.09); }
        P('wing-membrane-'+(wi+1),'box',{w:0.1,h:2.4,d:3.0},[-0.9,wy+0.9,s*2.0],[0,s*22,s*20],{color:M.color,metalness:M.metalness,roughness:M.roughness+0.1,opacity:0.95});
      });
    }
    return {name:'DAMRU-Sculpture',units:'m',background:'#0a0c12',environment:{ground:true,grid:true},camera:{position:[10,7,12],target:[0,4.5,0]},metadata:{generator:'Damru Organic Sculpture Compiler',designIntent:(prompt||'sculpture'),winged:winged,serpentine:serpent,parts:O.length,minimumAssembly:true},objects:O};
  }
  function universalCategory(p){return isDNAPrompt(p)?'DNA/biology':isProteinPrompt(p)?'protein/biology':isMoleculePrompt(p)?'molecular chemistry':isSpaceVehiclePrompt(p)?'aerospace':/car|bike|vehicle|aircraft|jet|ship/.test((p||'').toLowerCase())?'vehicle':/robot|machine|engine|gear|mechanism/.test((p||'').toLowerCase())?'mechanical system':/house|building|city|bridge|room|factory/.test((p||'').toLowerCase())?'architecture':(isOrganicPrompt(p)&&!/robot|android|humanoid/.test((p||'').toLowerCase()))?'organic sculpture':'general object';}
  function universalContract(p){var c=universalCategory(p);var base='\n\nUNIVERSAL FORGE CONTRACT: Category='+c+'. Decompose into named semantic parts with realistic proportions, spatial relationships, functional materials and deliberately different editable colours. Never answer with base+concept-core or one primitive. Use groups and multiple geometry types. For arbitrary links/bonds use type bond with start:[x,y,z], end:[x,y,z], size:{r:number}. Ground is Y=0 unless scientific molecular scene. Add metadata with assumptions and simulation domain. Preserve scientific honesty; do not claim engineering/biological/chemical certification.';if(c==='organic sculpture'){base+='\n\nORGANIC SCULPTURE MODE: This is a living/artistic form on a pedestal, NOT a machine. Use ONLY smooth organic primitives - sphere, capsule, cone, tapered cylinder (rt!=rb), lathe, torus, and bond (start/end) for limbs/neck/tail - never plain boxes for the body. Build: pedestal/base; a curved chain of shrinking body/spine segments (offset each position AND rotation so the silhouette flows, never a straight stack); a head with eyes, snout and horns/ears; four limbs each from tapered bonds + sphere joints (hip, knee, foot); a long tapering tail from many shrinking spheres linked by bonds; dorsal spikes/scales via small cones. Add wings from bond ribs + a thin membrane if it is a dragon/bird/griffin/angel. Minimum 26 named parts. If a material is named (gold, bronze, silver, marble, jade, obsidian) apply ONE coherent metallic/stone material family to every structural part: gold=#ffcf47 metalness .95 roughness .22; bronze=#b3702a .9/.35; silver=#cfd6de .95/.2; marble=#eae6df .03/.5; jade=#3fae7a .2/.35.';}return base;}

  function sceneQuality(spec, prompt) {
    var list=(spec&&(spec.objects||spec.parts))||[], flat=[];
    function walk(a){(a||[]).forEach(function(o){flat.push(o);if(o.children)walk(o.children);});}
    walk(list);
    var score=flat.length, names=flat.map(function(o){return ((o.name||o.id||'')+' '+(o.type||'')).toLowerCase();}).join(' ');
    if(isSpaceVehiclePrompt(prompt)) {
      ['hull','nose','engine','nozzle','flap','fin','window','heat','tile','rcs','landing','avionics','launch','tower'].forEach(function(k){if(names.indexOf(k)>=0)score+=7;});
      var types={};flat.forEach(function(o){types[o.type||'box']=1;});score+=Object.keys(types).length*2;
    }
    var types={},colors={};flat.forEach(function(o){types[(o.type||'box').toLowerCase()]=1;var c=o.material&&o.material.color;if(c)colors[c]=1;});score+=Object.keys(types).length*2+Math.min(8,Object.keys(colors).length);
    if(isDNAPrompt(prompt)){['phosphate','sugar','base','hydrogen','backbone'].forEach(function(k){if(names.indexOf(k)>=0)score+=8;});}
    return score;
  }

  function deepBuildPrompt(prompt) { return (prompt||'').trim().length>2; }

  function fallbackSpec(prompt) {
    var x = (prompt || '').toLowerCase();
    var vaultSpec = window.DamruModelVault && window.DamruModelVault.build ? window.DamruModelVault.build(prompt) : null;
    if (vaultSpec) return vaultSpec;
    if (isDNAPrompt(x)) return dnaSpec(prompt);
    if (isProteinPrompt(x)) return proteinSpec(prompt);
    if (isMoleculePrompt(x)) return moleculeSpec(prompt);
    if (isSpaceVehiclePrompt(x)) return starshipSpec(prompt);
    if (isOrganicPrompt(x) && !/robot|android|humanoid/.test(x)) return creatureSpec(prompt);
    var objects = [], mat = { color:'#d8734f', metalness:.15, roughness:.55 };
    function box(id,w,h,d,pos,color){ objects.push({id:id,name:id,type:'box',size:{w:w,h:h,d:d},position:pos,material:{color:color||'#d8734f',metalness:.12,roughness:.58}}); }
    if (/robot|android|humanoid/.test(x)) {
      box('pelvis',2.6,1.1,1.5,[0,4.3,0],'#3e536f');box('torso',3.5,4,1.8,[0,6.8,0],'#557ea8');box('chest-core',1.1,1.1,.25,[0,7.2,.98],'#39c9ff');
      objects.push({id:'head',name:'sensor-head',type:'sphere',size:{r:1.15},position:[0,9.7,0],material:{color:'#9aa9bd',metalness:.55,roughness:.28}},{id:'eye-L',name:'eye-L',type:'sphere',size:{r:.16},position:[-.4,9.85,1.05],material:{color:'#32e2ff',emissive:'#0088aa'}},{id:'eye-R',name:'eye-R',type:'sphere',size:{r:.16},position:[.4,9.85,1.05],material:{color:'#32e2ff',emissive:'#0088aa'}});
      [-1,1].forEach(function(side){var sx=side*2.4;objects.push({id:'shoulder-'+side,name:'shoulder-joint-'+side,type:'sphere',size:{r:.55},position:[sx,7.8,0],material:{color:'#e1a64b',metalness:.45,roughness:.3}},{id:'upper-arm-'+side,name:'upper-arm-'+side,type:'capsule',size:{r:.38,h:1.7},position:[sx,6.5,0],material:{color:'#6d87a6',metalness:.4,roughness:.35}},{id:'elbow-'+side,name:'elbow-'+side,type:'sphere',size:{r:.42},position:[sx,5.1,0],material:{color:'#e1a64b'}},{id:'forearm-'+side,name:'forearm-'+side,type:'capsule',size:{r:.34,h:1.6},position:[sx,3.9,0],material:{color:'#8b9db3',metalness:.4}},{id:'hand-'+side,name:'gripper-'+side,type:'box',size:{w:.75,h:.7,d:.6},position:[sx,2.7,0],material:{color:'#38485f'}});});
      [-1,1].forEach(function(side){var sx=side*.85;objects.push({id:'hip-'+side,name:'hip-'+side,type:'sphere',size:{r:.48},position:[sx,3.7,0],material:{color:'#e1a64b'}},{id:'thigh-'+side,name:'thigh-'+side,type:'capsule',size:{r:.48,h:2.0},position:[sx,2.35,0],material:{color:'#637f9e',metalness:.35}},{id:'knee-'+side,name:'knee-'+side,type:'sphere',size:{r:.46},position:[sx,1.15,0],material:{color:'#e1a64b'}},{id:'foot-'+side,name:'foot-'+side,type:'box',size:{w:1.1,h:.38,d:1.8},position:[sx,.25,.35],material:{color:'#2c394c'}});});
    } else if (/car|automobile|sports car|truck/.test(x)) {
      box('chassis',4.2,.6,7.2,[0,1.05,0],'#272f3b');box('body-shell',3.8,1.3,5.8,[0,1.75,0],'#d94f43');box('cabin',3.25,1.35,2.9,[0,2.85,-.35],'#263b55');box('windshield',3.0,.9,.12,[0,2.95,1.12],'#63b8da');box('rear-glass',2.9,.8,.12,[0,2.9,-1.83],'#63b8da');box('front-bumper',3.9,.42,.45,[0,.95,3.65],'#161b22');box('rear-bumper',3.9,.42,.45,[0,.95,-3.65],'#161b22');
      [-1,1].forEach(function(sx){[-1,1].forEach(function(sz){objects.push({id:'wheel-'+sx+'-'+sz,name:'wheel-'+sx+'-'+sz,type:'torus',size:{r:.72,tube:.25},position:[sx*2,1,sz*2.35],rotation:[0,90,0],material:{color:'#111318',metalness:.15,roughness:.85}},{id:'hub-'+sx+'-'+sz,name:'hub-'+sx+'-'+sz,type:'cylinder',size:{r:.32,h:.25},position:[sx*2,1,sz*2.35],rotation:[0,0,90],material:{color:'#b5bdc8',metalness:.8,roughness:.2}});});});
    } else if (/aircraft|airplane|aeroplane|jet/.test(x)) {
      objects.push({id:'fuselage',name:'fuselage',type:'capsule',size:{r:1.15,h:8.5},position:[0,3,0],rotation:[90,0,0],material:{color:'#c9d2db',metalness:.55,roughness:.28}},{id:'nose',name:'nose-cone',type:'cone',size:{r:1.12,h:2.7},position:[0,3,5.6],rotation:[90,0,0],material:{color:'#aeb9c5',metalness:.5}},{id:'main-wing',name:'main-wing',type:'wedge',size:{w:13,h:.35,d:4.2},position:[0,3,-.1],rotation:[0,0,0],material:{color:'#72849a',metalness:.5}},{id:'tail-wing',name:'tail-wing',type:'wedge',size:{w:5,h:.22,d:2},position:[0,3,-4.35],material:{color:'#72849a'}},{id:'vertical-tail',name:'vertical-tail',type:'wedge',size:{w:2.4,h:3,d:.28},position:[0,4.5,-4.35],rotation:[0,90,0],material:{color:'#e05c47'}});[-1,1].forEach(function(sx){objects.push({id:'engine-'+sx,name:'turbofan-'+sx,type:'cylinder',size:{r:.65,h:2.2},position:[sx*2.6,2.35,.3],rotation:[90,0,0],material:{color:'#444f5d',metalness:.7}},{id:'intake-'+sx,name:'intake-'+sx,type:'torus',size:{r:.65,tube:.09},position:[sx*2.6,2.35,1.4],material:{color:'#20252c',metalness:.75}});});
    } else if (/tree|plant|forest/.test(x)) {
      objects.push({id:'trunk',name:'trunk',type:'cylinder',size:{rt:.55,rb:.9,h:7},position:[0,3.5,0],material:{color:'#70482d',roughness:.95}});for(var i=0;i<14;i++){var a=i*2.399,y=3+i*.32,start=[0,y,0],end=[Math.cos(a)*(2.1+i*.035),y+1.2,Math.sin(a)*(2.1+i*.035)];objects.push(bondSpec('branch-'+i,start,end,'#70482d',.16,'wood'),{id:'foliage-'+i,name:'leaf-cluster-'+i,type:'icosahedron',size:{r:1.05+(i%3)*.18},position:end,material:{color:i%2?'#2f8f4e':'#45ae62',roughness:.9}});}
    } else if (/solar system|planetary system|planets/.test(x)) {
      objects.push({id:'sun',name:'Sun',type:'sphere',size:{r:2},position:[0,3,0],material:{color:'#ffba35',emissive:'#e85b00',roughness:.35}});var pc=['#8f7b6b','#e5b66b','#4e8ed0','#c55239','#d8ad79','#d6c79c','#85bccc','#426cc0'];for(var i=0;i<8;i++){var r=3.2+i*1.35,a=i*.87;objects.push({id:'orbit-'+i,name:'orbit-'+(i+1),type:'torus',size:{r:r,tube:.018},position:[0,3,0],rotation:[90,0,0],material:{color:'#33415b',roughness:.8}},{id:'planet-'+i,name:'planet-'+(i+1),type:'sphere',size:{r:.2+i*.045},position:[Math.cos(a)*r,3,Math.sin(a)*r],material:{color:pc[i],roughness:.65},animation:{type:'orbit',axis:'y',speed:.06+i*.02}});}
    } else if (/city|building|villa|house|tower/.test(x)) {
      box('foundation',18,.35,14,[0,.18,0],'#777f91');
      for (var i=0;i<5;i++) box('block-'+i,3,3+i*1.35,3,[i*3-6,(3+i*1.35)/2+.35,(i%2)*4-2],i%2?'#6687aa':'#c7835c');
      box('road',22,.08,3,[0,.05,7],'#252b35');
    } else if (/gear|cog/.test(x)) {
      objects.push({id:'gear',name:'gear',type:'cylinder',size:{rt:4,rb:4,h:1},position:[0,.5,0],rotation:[0,0,0],material:{color:'#aeb7c7',metalness:.8,roughness:.25},csg:{operation:'subtract',with:[{id:'bore',name:'bore',type:'cylinder',size:{rt:1,rb:1,h:2},position:[0,0,0],material:mat}]}});
    } else if (/table|desk/.test(x)) {
      box('top',8,.5,4,[0,4,0],'#9b633d'); [[-3,2,-1.4],[3,2,-1.4],[-3,2,1.4],[3,2,1.4]].forEach(function(p,i){box('leg-'+i,.5,4,.5,p,'#68442c')});
    } else if (/chair/.test(x)) {
      box('seat',4,.5,4,[0,2.5,0]); box('back',4,4,.5,[0,4.5,1.75]); [[-1.5,1,-1.5],[1.5,1,-1.5],[-1.5,1,1.5],[1.5,1,1.5]].forEach(function(p,i){box('leg-'+i,.4,2.2,.4,p)});
    } else {
      box('reference-platform',9,.35,9,[0,.175,0],'#3b465a');box('primary-body',4.8,3.2,4.8,[0,2.0,0],'#d8734f');
      objects.push({id:'upper-form',name:'upper-form',type:'capsule',size:{r:1.65,h:2.2},position:[0,4.75,0],material:{color:'#7696bd',metalness:.22,roughness:.42}},{id:'central-feature',name:'central-feature',type:'sphere',size:{r:.82},position:[0,3.1,2.55],material:{color:'#55d7c1',emissive:'#063a35',roughness:.3}});
      for(var i=0;i<8;i++){var a=i*Math.PI/4,px=Math.cos(a)*2.75,pz=Math.sin(a)*2.75;objects.push({id:'semantic-module-'+(i+1),name:'semantic-module-'+(i+1),type:i%2?'box':'cylinder',size:i%2?{w:.8,h:1.8,d:.8}:{r:.44,h:1.8},position:[px,2.25,pz],material:{color:i%2?'#e0ad54':'#6e8de0',metalness:.25,roughness:.45}},bondSpec('support-link-'+(i+1),[px*.55,2.3,pz*.55],[px,2.3,pz],'#aab6c8',.07,'structural'));}
      objects.push({id:'top-ring',name:'top-interface',type:'torus',size:{r:1.75,tube:.16},position:[0,6.25,0],rotation:[90,0,0],material:{color:'#d5dbe5',metalness:.65,roughness:.22}});
    }
    return {name:'instant-'+(prompt||'model').slice(0,32).replace(/[^a-z0-9]+/ig,'-'),units:'m',environment:{ground:true,grid:true},objects:objects};
  }

  function cancelGeneration() {
    S.generationSeq += 1; S.generating = false;
    var c = document.getElementById('dv-cancel'); if (c) c.style.display = 'none';
    var g = document.getElementById('dv-go'); if (g) g.disabled = false;
    showStatus('AI refinement cancelled — instant draft kept.', false); setTimeout(hideStatus, 2200);
  }

  async function generate(rawPrompt) {
    await loadLibs(); setupThree();
    var seq = ++S.generationSeq; S.generating = true; S.lastPrompt = rawPrompt;
    var cancel = document.getElementById('dv-cancel'); if (cancel) cancel.style.display = '';
    var deep = deepBuildPrompt(rawPrompt), baseline = fallbackSpec(rawPrompt), baselineScore = sceneQuality(baseline, rawPrompt);

    // Zero-blank + no-downgrade invariant: a complete deterministic assembly appears first.
    buildFromSpec(baseline);
    showStatus('Universal '+universalCategory(rawPrompt)+' kernel ready · Damru reasoning pass 1/3…', true);

    var hits = await webResearch(rawPrompt); if (seq !== S.generationSeq) return;
    S.research = hits;
    var context = hits.map(function(h,i){return '['+(i+1)+'] '+(h.title||'')+': '+(h.snippet||'').slice(0,700)+' '+(h.url||'');}).join('\n');
    var engineered = rawPrompt;
    if (deep) {
      showStatus('Deep Build 1/3 · architecture and subsystem reasoning…', true);
      try { engineered = await withTimeout(refinePrompt(rawPrompt), 38000, 'design brief'); } catch (briefErr) { engineered = rawPrompt; }
      if (seq !== S.generationSeq) return;
    }
    var contract = universalContract(rawPrompt) + (isSpaceVehiclePrompt(rawPrompt) ? '\nAEROSPACE MINIMUM: hull, nose, propulsion, engine cluster, TPS, control surfaces, avionics, RCS, landing/launch hardware; minimum 32 named parts.' : '');
    var user = engineered + contract + (context ? '\n\nFRESH WEB RESEARCH (reference only; reconcile dimensions):\n'+context : '') + '\n\nReturn a manufacturable, dimensioned STRICT JSON scene.';
    showStatus('Deep Build 2/3 · synthesising precision CAD with '+hits.length+' sources…', true);

    var spec = null, err = null;
    try {
      var out = await withTimeout(llm([{role:'system',content:SCHEMA},{role:'user',content:user}], deep ? 6200 : 3000), deep ? 90000 : 24000, 'Damru CAD');
      spec = extractJSON(out);
    } catch (e) { err = e; }
    if (!spec && seq === S.generationSeq) {
      try {
        showStatus('Deep Build 3/3 · repairing and validating scene graph…', true);
        var repairInstruction = SCHEMA+'\nReturn compact valid JSON. '+(deep?'Maximum 90 objects; preserve every required subsystem.':'Maximum 24 objects.');
        var repaired = await withTimeout(llm([{role:'system',content:repairInstruction},{role:'user',content:user}], deep ? 4200 : 1600), deep ? 42000 : 8000, 'CAD repair');
        spec = extractJSON(repaired);
      } catch (e2) { err = e2; }
    }
    if (seq !== S.generationSeq) return;

    if (spec) {
      var candidateScore=sceneQuality(spec,rawPrompt), threshold=Math.max(8,baselineScore*.72);
      if (candidateScore >= threshold) {
        showStatus('Validated researched model · building scene…', true); buildFromSpec(spec);
        toast('Deep Build ready · quality '+candidateScore+' · '+hits.length+' sources');
      } else {
        showStatus('Weak AI proposal rejected · strong local '+universalCategory(rawPrompt)+' model preserved.', false);
        toast('No-downgrade gate protected '+baseline.objects.length+' semantic parts');
        setTimeout(hideStatus,3600);
      }
    } else {
      showStatus('Cloud refinement unavailable · '+baseline.objects.length+'-part '+universalCategory(rawPrompt)+' model preserved.', false);
      setTimeout(hideStatus, 3800);
    }
    S.generating = false; if (cancel) cancel.style.display = 'none';
  }

  async function regenerateSelected() {
    if (!S.selected) return toast('Select a part first');
    var name = S.selected.name;
    var p = prompt('Regenerate "' + name + '" as:', name);
    if (!p) return;
    showStatus('Regenerating part...', true);
    try {
      var msgs = [{ role: 'system', content: SCHEMA + '\nReturn a scene with EXACTLY ONE object replacing the described part.' }, { role: 'user', content: p }];
      var out = await llm(msgs, 3000);
      var spec = extractJSON(out); var o = (spec.objects || [])[0];
      if (o) { var pos = S.selected.position.clone(); var nu = buildObject(o); nu.position.copy(pos); deleteSelected(); S.root.add(nu); collectSelectable(nu, o); select(nu); refreshOutliner(); }
    } catch (e) { toast('Regenerate failed'); }
    hideStatus();
  }

  function runPrompt() {
    var inp = document.getElementById('dv-prompt-in');
    var v = (inp.value || '').trim(); if (!v) return;
    document.getElementById('dv-go').disabled = true;
    generate(v).finally(function () { document.getElementById('dv-go').disabled = false; });
  }

  function showStatus(t, spin) { var s = document.getElementById('dv-status'); document.getElementById('dv-status-txt').textContent = t; s.querySelector('.dv-spin').style.display = spin ? '' : 'none'; s.classList.add('show'); if(window.DamruChakra) window.DamruChakra.setState(spin?'building':'ready', spin?'VISUALISE':'READY'); }
  function hideStatus() { document.getElementById('dv-status').classList.remove('show'); if(window.DamruChakra) window.DamruChakra.setState('ready','READY'); }

  // ---- open / close --------------------------------------------------------
  async function open(initialPrompt) {
    buildModal();
    document.getElementById('dv-modal').classList.add('dv-open');
    try { showStatus('Loading 3D engine...', true); await loadLibs(); setupThree(); onResize(); hideStatus(); }
    catch (e) { showStatus('3D engine load failed: ' + (e && e.message), false); return; }
    if (initialPrompt) { document.getElementById('dv-prompt-in').value = initialPrompt; runPrompt(); }
  }
  function close() {
    if (S.walkActive) endWalk();
    var m = document.getElementById('dv-modal'); if (m) m.classList.remove('dv-open');
  }

  // ---- launcher injection into Damru UI ------------------------------------
  function injectLauncher() {
    try {
      var pills = [].slice.call(document.querySelectorAll('button, .tpill'));
      var dev = pills.find(function (b) { return /developer/i.test(b.textContent || ''); });
      if (dev && dev.parentNode && !document.getElementById('dv-pill')) {
        var pill = dev.cloneNode(true); pill.id = 'dv-pill';
        pill.innerHTML = '<span class="ic">'+visualChakra(18,'pulse')+'</span> Visualise';
        pill.onclick = function () { open(); };
        dev.parentNode.insertBefore(pill, dev.nextSibling);
      }
      var items = [].slice.call(document.querySelectorAll('*'));
      var brain = items.find(function (n) { return n.children.length <= 2 && /^\s*\S*\s*Brain Lab\s*$/.test(n.textContent || '') && n.offsetParent; });
      if (brain && !document.getElementById('dv-nav')) {
        var nav = brain.cloneNode(true); nav.id = 'dv-nav';
        nav.innerHTML = nav.innerHTML.replace(/Brain Lab/i, 'Visualise 3D');
        nav.onclick = function () { open(); };
        brain.parentNode.insertBefore(nav, brain.nextSibling);
      }
    } catch (e) { console.warn('[visualise] launcher inject skipped', e); }
    if (!document.getElementById('dv-pill') && !document.getElementById('dv-nav')) {
      var fab = document.getElementById('dv-launch') || el('button', { id: 'dv-launch', html: visualChakra(18,'pulse')+' Visualise', onclick: function () { open(); } });
      fab.style.display = 'block';
      if (!fab.parentNode) document.body.appendChild(fab);
    }
  }

  window.DamruVisualise = { open: open, close: close, generate: generate, getSTLBlob: getSTLBlob, cancel: cancelGeneration, state: S, setSelectedMaterial:applySelectedMaterial, category:universalCategory, dnaSpec:dnaSpec, moleculeSpec:moleculeSpec, proteinSpec:proteinSpec, loadGLB: loadExternalGLB };
  window.openVisualise = function (p) { open(p); };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { setTimeout(injectLauncher, 800); });
  else setTimeout(injectLauncher, 800);
})();
