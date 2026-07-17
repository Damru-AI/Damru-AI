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
      GLTFExporter, STLExporter,
      CSG_Evaluator, CSG_Brush, CSG_ADD, CSG_SUB, CSG_INT;
  var libsReady = false;

  async function loadLibs() {
    if (libsReady) return;
    THREE = await import('three');
    OrbitControls = (await import('three/addons/controls/OrbitControls.js')).OrbitControls;
    TransformControls = (await import('three/addons/controls/TransformControls.js')).TransformControls;
    try { PointerLockControls = (await import('three/addons/controls/PointerLockControls.js')).PointerLockControls; } catch (e) {}
    GLTFExporter = (await import('three/addons/exporters/GLTFExporter.js')).GLTFExporter;
    STLExporter = (await import('three/addons/exporters/STLExporter.js')).STLExporter;
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
      + '#dv-edit{padding:8px;border-top:1px solid #1c2130;display:flex;flex-wrap:wrap;gap:6px}'
      + '#dv-prompt{display:flex;gap:8px;padding:10px 12px;background:#12151d;border-top:1px solid #232838}'
      + '#dv-prompt input{flex:1;background:#0b0d12;border:1px solid #2b3350;border-radius:10px;padding:11px 13px;color:#fff;font-size:14px;outline:none}'
      + '#dv-prompt input:focus{border-color:#e8623d}'
      + '#dv-go{background:#e8623d;color:#fff;border:none;border-radius:10px;padding:0 18px;font-weight:700;cursor:pointer;font-size:14px}'
      + '#dv-go:disabled{opacity:.5;cursor:default}'
      + '#dv-cancel{background:#51232a;border:1px solid #8a3945;color:#ffdfe3;border-radius:10px;padding:0 13px;cursor:pointer;display:none}'
      + '#dv-status{position:absolute;top:12px;left:12px;background:rgba(10,12,18,.85);border:1px solid #2b3350;border-radius:10px;padding:8px 13px;font-size:13px;display:none;align-items:center;gap:8px;max-width:70%}'
      + '#dv-status.show{display:flex}'
      + '.dv-spin{width:15px;height:15px;border:2px solid #3a4573;border-top-color:#e8623d;border-radius:50%;animation:dvspin .8s linear infinite}'
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

    var status = el('div', { id: 'dv-status' }, [el('div', { class: 'dv-spin' }), el('span', { id: 'dv-status-txt', text: 'Working...' })]);
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
      mkBtn('\u2715 Delete', deleteSelected)
    ]);
    var side = el('div', { id: 'dv-side' }, [el('h4', { text: 'Scene outliner' }), outliner, editBar]);

    var body = el('div', { id: 'dv-body' }, [canvasWrap, side]);

    var top = el('div', { id: 'dv-top' }, [
      el('div', { class: 'dv-title', html: '\uD83E\uDDCA Damru Visualise <span style="color:#7f8aa6;font-weight:500">&middot; AI CAD studio</span>' }),
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
    var promptBar = el('div', { id: 'dv-prompt' }, [promptInput, el('button', { id: 'dv-cancel', text: 'Cancel', onclick: cancelGeneration }), el('button', { id: 'dv-go', text: 'Visualise', onclick: runPrompt })]);

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
      default: return new THREE.BoxGeometry(1, 1, 1);
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
    var mesh = new THREE.Mesh(geometryFor(spec), materialFor(spec));
    applyTransform(mesh, spec);
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
    S.selected = obj; S.transform.attach(obj); highlightOutliner(obj);
  }
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
    'Build rich, complete, realistic scenes. A city = ground roads (thin boxes) + building block via repeat with vary.scaleY + landmarks. Keep total meshes sensible (<5000).'
  ].join('\n');

  async function llm(messages, mt) {
    if (typeof window.__damruLLM === 'function') return await window.__damruLLM(messages, mt);
    if (typeof window.engine === 'function') return await window.engine(messages, mt || 6000, 0.35, 'code');
    throw new Error('No LLM bridge (window.__damruLLM) found');
  }

  async function refinePrompt(raw) {
    try {
      var msgs = [
        { role: 'system', content: 'You are a senior industrial/architectural designer. Expand the user request into a precise, buildable design brief for a 3D/CAD engine: list key parts, approximate dimensions in metres, materials/colors, layout and spatial relationships, and any repetition. Be concrete and compact (bullet points). Do NOT output JSON.' },
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

  function fallbackSpec(prompt) {
    var x = (prompt || '').toLowerCase(), objects = [], mat = { color:'#d8734f', metalness:.15, roughness:.55 };
    function box(id,w,h,d,pos,color){ objects.push({id:id,name:id,type:'box',size:{w:w,h:h,d:d},position:pos,material:{color:color||'#d8734f',metalness:.12,roughness:.58}}); }
    if (/city|building|villa|house|tower/.test(x)) {
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
      box('base',8,.5,8,[0,.25,0],'#4f596d');
      objects.push({id:'core',name:'concept-core',type:/ball|sphere/.test(x)?'sphere':'cylinder',size:/ball|sphere/.test(x)?{r:2.5}:{rt:2.6,rb:3.2,h:6},position:[0,3.5,0],material:mat});
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
    // Zero-blank UX: deterministic draft appears immediately, independent of network/LLM.
    buildFromSpec(fallbackSpec(rawPrompt));
    showStatus('Instant draft ready · researching real-world dimensions…', true);
    var hits = await webResearch(rawPrompt); if (seq !== S.generationSeq) return;
    S.research = hits;
    var context = hits.map(function(h,i){return '['+(i+1)+'] '+(h.title||'')+': '+(h.snippet||'').slice(0,700)+' '+(h.url||'');}).join('\n');
    var user = rawPrompt + (context ? '\n\nFRESH WEB RESEARCH (use as reference, do not blindly copy):\n'+context : '') + '\n\nReturn a manufacturable, dimensioned STRICT JSON scene.';
    showStatus('Instant draft visible · Damru refining with '+hits.length+' web sources…', true);
    var spec = null, err = null;
    try {
      var out = await withTimeout(llm([{role:'system',content:SCHEMA},{role:'user',content:user}], 3000), 24000, 'Damru CAD');
      spec = extractJSON(out);
    } catch (e) { err = e; }
    if (!spec && seq === S.generationSeq) {
      try {
        var repaired = await withTimeout(llm([{role:'system',content:SCHEMA+'\nReturn compact valid JSON. Maximum 24 objects.'},{role:'user',content:rawPrompt}], 1600), 8000, 'CAD repair');
        spec = extractJSON(repaired);
      } catch (e2) { err = e2; }
    }
    if (seq !== S.generationSeq) return;
    if (spec) { showStatus('Building researched precision model…', true); buildFromSpec(spec); toast('Research-refined model ready · '+hits.length+' sources'); }
    else { showStatus('Instant model kept — refinement unavailable ('+(err&&err.message||'offline')+').', false); setTimeout(hideStatus, 3200); }
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

  function showStatus(t, spin) { var s = document.getElementById('dv-status'); document.getElementById('dv-status-txt').textContent = t; s.querySelector('.dv-spin').style.display = spin ? '' : 'none'; s.classList.add('show'); }
  function hideStatus() { document.getElementById('dv-status').classList.remove('show'); }

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
        pill.innerHTML = '<span class="ic">\uD83E\uDDCA</span> Visualise';
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
      var fab = document.getElementById('dv-launch') || el('button', { id: 'dv-launch', html: '\uD83E\uDDCA Visualise', onclick: function () { open(); } });
      fab.style.display = 'block';
      if (!fab.parentNode) document.body.appendChild(fab);
    }
  }

  window.DamruVisualise = { open: open, close: close, generate: generate, getSTLBlob: getSTLBlob, cancel: cancelGeneration, state: S };
  window.openVisualise = function (p) { open(p); };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { setTimeout(injectLauncher, 800); });
  else setTimeout(injectLauncher, 800);
})();
