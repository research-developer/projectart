// p5.js sketch — black canvas, listens to backend PointerEvents, draws strokes.
// In M0 we draw plain lines. M6 swaps in p5.brush.js for organic textures.

let canvasW = 1920;
let canvasH = 1080;
let last = null;            // {x, y} for the previous contact point
let hud = { x: 0, y: 0, visible: false };
let dragging = false;       // local mouse drag state for --input mouse

function setup() {
  createCanvas(windowWidth, windowHeight);
  background(0);
  noFill();
  stroke(255);
  strokeWeight(4);
  strokeCap(ROUND);
  strokeJoin(ROUND);

  window.PA_WS.on('message', onMessage);

  // Local mouse pumping for --input mouse: forward events upstream over WS.
  const fwd = (ev, contact) => {
    const sx = (ev.clientX / window.innerWidth) * canvasW;
    const sy = (ev.clientY / window.innerHeight) * canvasH;
    window.PA_WS.send({
      type: 'mouse',
      x: sx,
      y: sy,
      contact,
      ts_ms: Math.round(performance.now()),
    });
  };
  window.addEventListener('mousedown', (e) => { dragging = true; fwd(e, true); });
  window.addEventListener('mouseup', (e) => { dragging = false; fwd(e, false); });
  window.addEventListener('mousemove', (e) => { fwd(e, dragging); });
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function onMessage(msg) {
  if (msg.type === 'hello') {
    canvasW = msg.canvas_w || canvasW;
    canvasH = msg.canvas_h || canvasH;
  }
  if (msg.type === 'pointer') {
    handlePointer(msg);
    return;
  }
  if (msg.type === 'hud_anchor') {
    hud = { x: msg.x, y: msg.y, visible: !!msg.visible };
    return;
  }
  // Forward to the entity overlay layer
  if (window.PA_SCENE) {
    window.PA_SCENE.onMessage(msg);
  }
}

function handlePointer(p) {
  // Map backend canvas-space to current window-space.
  const x = (p.x / canvasW) * width;
  const y = (p.y / canvasH) * height;
  if (p.contact) {
    if (last) {
      stroke(255);
      line(last.x, last.y, x, y);
    }
    last = { x, y };
  } else {
    last = null;
  }
}

function draw() {
  // Entity overlays (cats, people, etc.) on top of the canvas
  if (window.PA_SCENE) {
    const now = performance.now();
    window.PA_SCENE.tick(now);
    window.PA_SCENE.draw(window);
  }

  // HUD placeholder — small reticle that follows the active hand. M4 expands.
  if (hud.visible) {
    push();
    noFill();
    stroke(255, 80);
    strokeWeight(1);
    const hx = (hud.x / canvasW) * width;
    const hy = (hud.y / canvasH) * height;
    circle(hx, hy, 32);
    pop();
  }
}
