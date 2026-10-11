// Small DOM helpers shared by every section.
window.LLM = window.LLM || {};

(function (L) {
  const U = (L.ui = {});
  L.onTheme = [];  // redraw hooks for canvas/SVG drawings that read theme colors
  U.$ = (sel, root = document) => root.querySelector(sel);
  U.esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  U.pct = (p) => (p >= 0.995 ? "100%" : p < 0.0005 ? "0.0%" : (p * 100).toFixed(1) + "%");

  // Background color for a value in [0,1], blended from the panel surface toward the accent.
  U.heat = (v, varName = "--accent") => `color-mix(in oklab, var(${varName}) ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, var(--sunk))`;

  // Horizontal bar list. rows: [{label, value (0..1), text, cls, ghost}]
  U.bars = function (el, rows) {
    el.innerHTML = rows
      .map((r) => `<div class="bar ${r.cls || ""}"><span class="lbl" title="${U.esc(r.label)}">${U.esc(r.label)}</span><span class="track">${r.ghost != null ? `<span class="ghost" style="width:${(r.ghost * 100).toFixed(1)}%"></span>` : ""}<span class="fill" style="width:${(Math.max(0, r.value) * 100).toFixed(1)}%"></span></span><span class="val">${r.text}</span></div>`)
      .join("");
  };

  U.seg = function (el, items, current, onPick) {
    el.innerHTML = items.map(([k, label]) => `<button role="tab" data-k="${k}" aria-selected="${k === current}">${U.esc(label)}</button>`).join("");
    el.onclick = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-selected", String(x === b)));
      onPick(b.dataset.k);
    };
  };

  // Line chart in an SVG with a fixed viewBox. series: [{points:[[x,y]], cls, color}], hlines: [{y,label}]
  U.lineChart = function (svg, { xMax, yMin, yMax, series, hlines = [], xLabel = "", yTicks = [] }) {
    const vb = svg.viewBox.baseVal;
    const W = vb.width, H = vb.height, padL = 30, padR = 8, padT = 8, padB = 20;
    const X = (x) => padL + (x / xMax) * (W - padL - padR);
    const Y = (y) => padT + (1 - (y - yMin) / (yMax - yMin)) * (H - padT - padB);
    let s = `<line x1="${padL}" y1="${H - padB}" x2="${W - padR}" y2="${H - padB}" stroke="var(--line)"/>`;
    yTicks.forEach((t) => {
      s += `<line x1="${padL}" x2="${W - padR}" y1="${Y(t)}" y2="${Y(t)}" stroke="var(--line)" stroke-dasharray="2 3"/><text x="${padL - 4}" y="${Y(t) + 3}" text-anchor="end">${t}</text>`;
    });
    hlines.forEach((h) => {
      s += `<line x1="${padL}" x2="${W - padR}" y1="${Y(h.y)}" y2="${Y(h.y)}" stroke="${h.color || "var(--muted)"}" stroke-dasharray="4 3"/><text x="${W - padR - 2}" y="${Y(h.y) - 4}" text-anchor="end" style="fill:${h.color || "var(--muted)"}">${h.label}</text>`;
    });
    series.forEach((se) => {
      if (!se.points.length) return;
      const d = se.points.map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(Math.max(yMin, Math.min(yMax, y))).toFixed(1)}`).join("");
      if (se.area) s += `<path d="${d}L${X(se.points[se.points.length - 1][0])},${H - padB}L${X(se.points[0][0])},${H - padB}Z" fill="${se.color}" opacity="0.12"/>`;
      s += `<path d="${d}" fill="none" stroke="${se.color}" stroke-width="2" stroke-linejoin="round"/>`;
      if (se.dots) se.points.forEach(([x, y]) => (s += `<circle cx="${X(x)}" cy="${Y(y)}" r="2.6" fill="${se.color}"/>`));
      const [lx, ly] = se.points[se.points.length - 1];
      s += `<circle cx="${X(lx)}" cy="${Y(ly)}" r="3.5" fill="${se.color}"/>`;
    });
    s += `<text x="${W - padR}" y="${H - 6}" text-anchor="end">${xLabel}</text>`;
    svg.innerHTML = s;
  };

  U.copy = function (btn, text) {
    btn.addEventListener("click", () => {
      const done = () => { const t = btn.textContent; btn.textContent = "Copied"; setTimeout(() => (btn.textContent = t), 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text()).then(done, () => selectFallback());
      } else selectFallback();
      function selectFallback() {
        const pre = btn.closest(".panel").querySelector("pre");
        const r = document.createRange();
        r.selectNodeContents(pre);
        const sel = getSelection();
        sel.removeAllRanges();
        sel.addRange(r);
        btn.textContent = "Selected: press Ctrl/⌘+C";
      }
    });
  };
})(window.LLM);
