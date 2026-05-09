// Entity overlay layer. Subscribes to EntityEvent (`{type:"entity", phase, track_id, class_name, bbox_*, ...}`)
// and draws a fade-in / fade-out labeled box per tracked entity.
//
// Stacks on top of the strokes + HUD layers. Cleared every frame; entities
// know their own animation state (alpha + age) and self-prune when faded.
//
// Coordinates from the backend are in canvas-px (1920x1080 by default).
// We rescale to actual window dimensions before drawing.

(function () {
  const FADE_IN_MS = 220;
  const FADE_OUT_MS = 600;
  const STALE_MS = 1500;          // if no update for this long, treat as gone

  /** @type {Map<number, EntityState>} */
  const entities = new Map();

  // Default backend canvas size; overwritten by Hello.
  let canvasW = 1920;
  let canvasH = 1080;

  /** @typedef {{
   *    track_id: number,
   *    class_name: string,
   *    x: number, y: number, w: number, h: number,
   *    confidence: number,
   *    enteredAt: number,
   *    lastSeen: number,
   *    leavingAt: number | null,
   *    alpha: number,
   *  }} EntityState
   */

  function colorFor(name) {
    // Stable HSL by class hash → distinct color per class.
    let h = 0;
    for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) | 0;
    return `hsl(${(h % 360 + 360) % 360}, 75%, 65%)`;
  }

  function applyEntityEvent(msg, now) {
    const { track_id, phase, class_name, bbox_x, bbox_y, bbox_w, bbox_h, confidence } = msg;
    if (phase === 'leave') {
      const ent = entities.get(track_id);
      if (ent) ent.leavingAt = now;
      return;
    }
    let ent = entities.get(track_id);
    if (!ent) {
      ent = {
        track_id,
        class_name,
        x: bbox_x, y: bbox_y, w: bbox_w, h: bbox_h,
        confidence: confidence ?? 1,
        enteredAt: now,
        lastSeen: now,
        leavingAt: null,
        alpha: 0,
      };
      entities.set(track_id, ent);
    } else {
      ent.x = bbox_x; ent.y = bbox_y; ent.w = bbox_w; ent.h = bbox_h;
      ent.confidence = confidence ?? ent.confidence;
      ent.lastSeen = now;
      ent.leavingAt = null;
    }
  }

  function tick(now) {
    for (const ent of entities.values()) {
      if (ent.leavingAt !== null) {
        const t = (now - ent.leavingAt) / FADE_OUT_MS;
        ent.alpha = Math.max(0, 1 - t);
      } else if (now - ent.lastSeen > STALE_MS) {
        // No leave event but updates stopped — implicit fade.
        ent.leavingAt = now;
      } else {
        const t = (now - ent.enteredAt) / FADE_IN_MS;
        ent.alpha = Math.min(1, t);
      }
    }
    // Prune fully-faded entities.
    for (const [tid, ent] of entities) {
      if (ent.leavingAt !== null && ent.alpha <= 0.01) entities.delete(tid);
    }
  }

  function draw(p) {
    const sx = p.width / canvasW;
    const sy = p.height / canvasH;
    p.push();
    p.noFill();
    p.textFont('ui-monospace, SFMono-Regular, Menlo, monospace');
    p.textSize(14);
    p.strokeWeight(2);
    for (const ent of entities.values()) {
      const a = Math.round(255 * ent.alpha);
      if (a <= 0) continue;
      const col = colorFor(ent.class_name);
      const x = ent.x * sx;
      const y = ent.y * sy;
      const w = ent.w * sx;
      const h = ent.h * sy;
      p.stroke(p.color(col + ''));
      p.fill(p.color(0, 0, 0, 0));
      p.rect(x, y, w, h, 6);
      // label background
      p.noStroke();
      p.fill(0, 0, 0, Math.round(180 * ent.alpha));
      const label = `${ent.class_name} ${(ent.confidence * 100 | 0)}%`;
      const tw = p.textWidth(label) + 12;
      p.rect(x, y - 22, tw, 20, 4);
      p.fill(255, 255, 255, a);
      p.text(label, x + 6, y - 8);
    }
    p.pop();
  }

  function applyHello(msg) {
    if (msg.canvas_w) canvasW = msg.canvas_w;
    if (msg.canvas_h) canvasH = msg.canvas_h;
  }

  // Public API
  window.PA_SCENE = {
    onMessage(msg) {
      const now = performance.now();
      if (msg.type === 'hello') return applyHello(msg);
      if (msg.type === 'entity') return applyEntityEvent(msg, now);
    },
    draw,
    tick,
  };
})();
