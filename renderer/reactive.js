// renderer/reactive.js
// Consumes SceneFrame snapshots and maintains one persistent visual per object id,
// morphing (tweening) position/size/color toward the latest snapshot. Objects are
// created on first sight and removed only after their id is absent for a grace period.
(function () {
  const GRACE_MS = 400;          // keep a visual this long after its id disappears
  const TWEEN = 0.35;            // per-frame easing toward target [0..1]
  const visuals = new Map();     // id -> {x,y,r,color,alpha,shape, tx,ty,tr,tcolor,talpha, lastSeen}

  function onMessage(msg) {
    if (msg.type !== 'scene_frame') return;
    const now = performance.now();
    for (const o of msg.objects) {
      let v = visuals.get(o.id);
      if (!v) {
        v = { x: o.x, y: o.y, r: o.r, color: o.color, alpha: o.alpha, shape: o.shape };
        visuals.set(o.id, v);
      }
      v.tx = o.x; v.ty = o.y; v.tr = o.r; v.tcolor = o.color;
      v.talpha = o.alpha; v.shape = o.shape; v.lastSeen = now;
    }
  }

  function setup() {
    createCanvas(windowWidth, windowHeight);
    rectMode(CENTER);
    window.PA_WS.on('message', onMessage);
  }
  function windowResized() { resizeCanvas(windowWidth, windowHeight); }

  function draw() {
    background(10, 10, 10);
    const now = performance.now();
    const W = width, H = height;
    for (const [id, v] of visuals) {
      if (now - v.lastSeen > GRACE_MS) { visuals.delete(id); continue; }
      v.x += (v.tx - v.x) * TWEEN;
      v.y += (v.ty - v.y) * TWEEN;
      v.r += (v.tr - v.r) * TWEEN;
      v.alpha += ((v.talpha ?? 1) - v.alpha) * TWEEN;
      if (v.tcolor) v.color = v.tcolor;
      const px = v.x * W, py = v.y * H, pr = v.r * Math.min(W, H);
      push();
      noFill();
      const c = color(v.color);
      c.setAlpha(255 * Math.max(0, Math.min(1, v.alpha)));
      stroke(c); strokeWeight(3);
      if (v.shape === 'box') rect(px, py, pr * 2, pr * 2, 6);
      else circle(px, py, pr * 2);
      pop();
    }
  }

  window.setup = setup; window.draw = draw; window.windowResized = windowResized;
})();
