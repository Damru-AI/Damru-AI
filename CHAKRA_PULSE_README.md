# DAMRU — CHAKRA PULSE UI

## Included upgrade

### 1. Ashoka Chakra identity
- Mathematically generated SVG with exactly 24 spokes.
- Replaces the old Damru logo across brand, welcome, avatars, settings and onboarding.
- Navy/blue styling designed for the existing dark interface.
- Accessible SVG label and reduced-motion support.

### 2. Cursor-style AI responses
- Damru answers now appear through a frame-synchronised progressive renderer.
- A glowing cursor remains visible while the response is being written.
- Long answers dynamically increase chunk size to avoid an excessively slow animation.
- Markdown, code blocks and action buttons hydrate only after typing finishes, preventing broken partial HTML.
- Loaded chat history remains instant rather than replaying old typing animations.

### 3. Chakra thinking pipeline
Thinking no longer displays three generic dots. A rotating Chakra and visible stages show:

- SANKALP — intent mapping
- KHOJ — memory and web context
- TARK — reasoning paths
- SATYA — self-verification
- UTTAR — response shaping

The same system is wired into normal chat, vision, image generation and video generation loaders.

### 4. Reactive animated startup
A full-screen Chakra startup sequence includes:

- 24-spoke rotating core
- orbit rings
- perspective grid
- pointer/touch-reactive light field
- staged boot messages
- tap-anywhere skip
- automatic close after about three seconds

### 5. Surprise: Chakra Pulse HUD
The top bar contains a live intelligence-state display:

`READY → THINKING → RESEARCH → VERIFYING → ANSWERING → BUILDING`

Tapping the HUD toggles Damru Deep Think mode.

### 6. Visualise integration
Visualise title, launcher and progress status now use the same Chakra identity. Visualise build/refinement updates the global Chakra Pulse HUD and returns it to READY when finished.

### 7. Starship Deep Build retained
The included `damru_visualise.js` also retains the previously completed 156-part deterministic Starship aerospace compiler and no-downgrade quality gate.

## Upload map

Upload/replace these files together in the root of `Damru-AI/Damru-AI`:

1. `index.html`
2. `damru_visualise.js`
3. `damru_simlab.js`
4. `damru_print_forge.js`

Keep script order:

```html
<script src="damru_visualise.js"></script>
<script src="damru_simlab.js"></script>
<script src="damru_print_forge.js"></script>
```

After Vercel deploys, test once in Chrome Incognito or clear cached files.

## Verification completed

- All four inline JavaScript blocks parsed successfully.
- `damru_visualise.js` syntax passed.
- 24-spoke generation contract passed.
- Typewriter/cursor function and final Markdown hydration hooks passed.
- Chakra state engine and all five reasoning-stage hooks passed.
- Startup, HUD and Visualise bridge hooks passed.
- Starship compiler remains present.
- ZIP integrity test passed during packaging.

The sandbox did not expose a usable Chromium/Playwright runtime despite listing one, so the final visual Vercel/mobile render remains a user-side check after deployment.
