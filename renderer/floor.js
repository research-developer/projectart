// renderer/floor.js
// Renders the stage (moles + HUD) to an offscreen buffer in stage space, then
// warps it onto the floor through a 4-corner homography using a FINE GRID MESH
// (perspective-correct — a two-triangle quad would bow interior lines). The 4
// projector corners are draggable (output calibration) and persisted upstream.
(function () {
  const GRID = 24;                 // mesh resolution (perspective-correct warp)
  const STAGE_W = 1000, STAGE_H = 1000;
  let stageBuf;
  const moles = new Map();         // id -> {x,y,r,color}
  let game = { score: 0, round_ms_left: 0, phase: 'playing' };
  let calibrate = false;
  let corners = null;              // projector px, stage order TL,TR,BR,BL
  let dragging = -1;

  function loadCorners() {
    try { const s = localStorage.getItem('PA_FLOOR_CORNERS'); if (s) return JSON.parse(s); } catch (e) {}
    return [[0, 0], [windowWidth, 0], [windowWidth, windowHeight], [0, windowHeight]];
  }
  function saveCorners() {
    try { localStorage.setItem('PA_FLOOR_CORNERS', JSON.stringify(corners)); } catch (e) {}
    window.PA_WS.send({ type: 'calib_output', corners });
  }

  // Homography mapping the unit square (stage TL,TR,BR,BL) -> the 4 dst px corners.
  function homography(dst) {
    const [p0, p1, p2, p3] = dst; // (0,0),(1,0),(1,1),(0,1)
    const x0 = p0[0], y0 = p0[1], x1 = p1[0], y1 = p1[1];
    const x2 = p2[0], y2 = p2[1], x3 = p3[0], y3 = p3[1];
    const dx1 = x1 - x2, dx2 = x3 - x2, dx3 = x0 - x1 + x2 - x3;
    const dy1 = y1 - y2, dy2 = y3 - y2, dy3 = y0 - y1 + y2 - y3;
    const den = dx1 * dy2 - dx2 * dy1;
    const g = (dx3 * dy2 - dx2 * dy3) / den;
    const h = (dx1 * dy3 - dx3 * dy1) / den;
    return [
      [x1 - x0 + g * x1, x3 - x0 + h * x3, x0],
      [y1 - y0 + g * y1, y3 - y0 + h * y3, y0],
      [g, h, 1],
    ];
  }
  function applyH(H, u, v) {
    const x = H[0][0] * u + H[0][1] * v + H[0][2];
    const y = H[1][0] * u + H[1][1] * v + H[1][2];
    const w = H[2][0] * u + H[2][1] * v + H[2][2];
    return [x / w, y / w];
  }

  function onMessage(msg) {
    if (msg.type === 'scene_frame') {
      const seen = new Set();
      for (const o of msg.objects) { if (o.kind === 'mole') { moles.set(o.id, o); seen.add(o.id); } }
      for (const id of [...moles.keys()]) if (!seen.has(id)) moles.delete(id);
    } else if (msg.type === 'game_state') {
      game = msg;
    }
  }

  window.setup = function () {
    createCanvas(windowWidth, windowHeight, WEBGL);
    stageBuf = createGraphics(STAGE_W, STAGE_H);
    corners = loadCorners();
    window.PA_WS.on('message', onMessage);
  };
  window.windowResized = function () { resizeCanvas(windowWidth, windowHeight); };

  function drawStage() {
    stageBuf.push();
    stageBuf.background(0);
    stageBuf.noStroke();
    for (const o of moles.values()) {
      stageBuf.fill(o.color || '#cc3333');
      stageBuf.circle(o.x * STAGE_W, o.y * STAGE_H, (o.r || 0.08) * 2 * STAGE_W);
    }
    stageBuf.fill(255);
    stageBuf.textSize(40);
    stageBuf.text('Score ' + game.score, 30, 50);
    stageBuf.text((game.round_ms_left / 1000 | 0) + 's', STAGE_W - 140, 50);
    if (game.phase === 'over') { stageBuf.textSize(90); stageBuf.text('TIME!', STAGE_W / 2 - 130, STAGE_H / 2); }
    stageBuf.pop();
  }

  window.draw = function () {
    drawStage();
    background(0);
    const H = homography(corners);
    texture(stageBuf);
    noStroke();
    translate(-width / 2, -height / 2);  // WEBGL origin -> top-left pixels
    for (let i = 0; i < GRID; i++) {
      for (let j = 0; j < GRID; j++) {
        const u0 = i / GRID, u1 = (i + 1) / GRID, v0 = j / GRID, v1 = (j + 1) / GRID;
        const a = applyH(H, u0, v0), b = applyH(H, u1, v0);
        const c = applyH(H, u1, v1), d = applyH(H, u0, v1);
        beginShape();
        vertex(a[0], a[1], 0, u0 * STAGE_W, v0 * STAGE_H);
        vertex(b[0], b[1], 0, u1 * STAGE_W, v0 * STAGE_H);
        vertex(c[0], c[1], 0, u1 * STAGE_W, v1 * STAGE_H);
        vertex(d[0], d[1], 0, u0 * STAGE_W, v1 * STAGE_H);
        endShape(CLOSE);
      }
    }
    if (calibrate) drawHandles();
  };

  function drawHandles() {
    noFill(); stroke(0, 255, 0); strokeWeight(2);
    beginShape();
    for (const c of corners) vertex(c[0], c[1]);
    endShape(CLOSE);
    fill(0, 255, 0); noStroke();
    for (let i = 0; i < 4; i++) circle(corners[i][0], corners[i][1], 18);
  }

  window.keyPressed = function () {
    if (key === 'c' || key === 'C') calibrate = !calibrate;
    if (['1', '2', '3', '4'].includes(key)) {
      window.PA_WS.send({ type: 'calib_input_capture', corner: parseInt(key, 10) - 1 });
    }
  };
  window.mousePressed = function () {
    if (!calibrate) return;
    for (let i = 0; i < 4; i++) {
      if (dist(mouseX, mouseY, corners[i][0], corners[i][1]) < 16) { dragging = i; return; }
    }
  };
  window.mouseDragged = function () { if (dragging >= 0) corners[dragging] = [mouseX, mouseY]; };
  window.mouseReleased = function () { if (dragging >= 0) { dragging = -1; saveCorners(); } };
})();
