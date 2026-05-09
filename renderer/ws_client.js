// Minimal WS client. The Python backend serves this static file on :8000 and
// the websocket on :8765. We compute the WS URL from the page host so the
// renderer works whether opened locally or from the projector display.

(function () {
  const status = document.getElementById('status');
  const wsUrl = `ws://${location.hostname}:8765/`;
  let socket = null;
  let backoff = 250;
  const listeners = { open: [], close: [], message: [] };

  function setStatus(t) { if (status) status.textContent = t; }

  function connect() {
    setStatus(`connecting ${wsUrl}…`);
    socket = new WebSocket(wsUrl);
    socket.addEventListener('open', () => {
      backoff = 250;
      setStatus('connected');
      listeners.open.forEach((fn) => fn());
    });
    socket.addEventListener('message', (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      listeners.message.forEach((fn) => fn(msg));
    });
    socket.addEventListener('close', () => {
      setStatus(`disconnected — retry in ${backoff}ms`);
      listeners.close.forEach((fn) => fn());
      setTimeout(connect, backoff);
      backoff = Math.min(backoff * 2, 4000);
    });
    socket.addEventListener('error', () => {
      try { socket.close(); } catch {}
    });
  }

  function send(obj) {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(obj));
    }
  }

  function on(name, fn) { listeners[name].push(fn); }

  window.PA_WS = { connect, send, on };
  connect();
})();
