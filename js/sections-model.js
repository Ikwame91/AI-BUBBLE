// Part 1: the forward pass — hero loop, tokenizer, embeddings, attention, prediction.
window.LLM = window.LLM || {};

(function (L) {
  const U = L.ui, M = L.math;
  const S = (L.sections = L.sections || {});
  const ngram = () => (L._ngram = L._ngram || new L.NGram(L.CORPUS));
  const joinWords = (ws) => ws.join(" ").replace(/ ([.,!?])/g, "$1");
  const tokColor = (id) => `color-mix(in oklab, hsl(${(id * 47) % 360} 70% 55%) 24%, var(--surface))`;

  // ================= HERO: the generation loop =================
  S.loop = function () {
    const input = U.$("#loop-prompt"), stagesEl = U.$("#loop-stages");
    const T = 0.9;
    let generated = [];
    let lastPick = null;
    const out = document.createElement("div");
    out.className = "out-text";
    stagesEl.after(out);

    function render(step) {
      const ng = ngram();
      const promptWords = L.words(input.value);
      const all = promptWords.concat(generated);
      const ids = all.map((w) => (ng.index.has(w) ? ng.index.get(w) : "?"));
      const d = ng.distribution(joinWords(all));
      const s = L.applySampling(d.logits, { temperature: T });
      const top = s.order.slice(0, 5);
      const last2 = all.slice(-2).join(" ");
      const st = [];
      const tail = all.slice(-6), tailIds = ids.slice(-6);
      st.push(["①", "Tokenize", "#tokenize", (all.length > 6 ? "… " : "") + tail.map((w, i) => `<span class="chip" style="background:${tokColor(typeof tailIds[i] === "number" ? tailIds[i] : 3)}">${U.esc(w)}<small>${tailIds[i]}</small></span>`).join(" ") + `<div class="muted" style="margin-top:6px">${all.length} tokens in context (this model uses whole words as tokens)</div>`]);
      st.push(["②", "Read context", "#attend", `Looks at <b>"${U.esc(last2)}"</b>.<br>${U.esc(d.used)}.<div class="muted" style="margin-top:6px">In a transformer: embeddings + attention layers.</div>`]);
      st.push(["③", "Score (logits)", "#predict", top.map((i) => `${U.esc(ng.vocab[i]).padEnd(8, " ").replace(/ /g, "&nbsp;")} ${d.logits[i].toFixed(2)}`).join("<br>") + `<div class="muted" style="margin-top:6px">+ ${ng.vocab.length - 5} more</div>`]);
      st.push(["④", "Softmax", "#predict", `<div class="bars mini"></div><div class="muted" style="margin-top:6px">temperature ${T}</div>`]);
      st.push(["⑤", "Sample", "#predict", lastPick ? `dice = <b>${lastPick.r.toFixed(3)}</b><br>→ picked <b style="color:var(--pick)">"${U.esc(lastPick.w)}"</b> (${U.pct(lastPick.p)})<div class="muted" style="margin-top:6px">appended; loop repeats</div>` : `<span class="muted">Press Next token to roll the dice.</span>`]);
      stagesEl.innerHTML = st.map(([n, h, href, body]) => `<div class="stage${step ? " flash" : ""}"><h4><i>${n}</i>${h}<a href="${href}">how</a></h4><div class="body">${body}</div></div>`).join("");
      U.bars(stagesEl.querySelector(".bars.mini"), top.map((i) => ({ label: ng.vocab[i], value: s.final[i], text: U.pct(s.final[i]) })));
      stagesEl.querySelectorAll(".bars.mini .bar").forEach((b) => (b.style.gridTemplateColumns = "4.5em 1fr 3.4em"));
      out.innerHTML = U.esc(joinWords(promptWords)) + (generated.length ? " " : "") + generated.map((w, i) => `<span class="gen${i === generated.length - 1 ? " last" : ""}">${U.esc(w)}</span>`).join(" ").replace(/ <span class="gen( last)?">([.,!?])/g, '<span class="gen$1">$2');
      return { d, s };
    }

    function step() {
      const ng = ngram();
      const all = L.words(input.value).concat(generated);
      const d = ng.distribution(joinWords(all));
      const s = L.applySampling(d.logits, { temperature: T });
      const r = Math.random();
      const i = M.sampleIndex(s.final, r);
      lastPick = { w: ng.vocab[i], p: s.final[i], r };
      generated.push(ng.vocab[i]);
      render(true);
    }
    U.$("#loop-step").onclick = step;
    U.$("#loop-run").onclick = () => {
      let n = 0;
      const t = setInterval(() => { step(); if (++n >= 12) clearInterval(t); }, 260);
    };
    U.$("#loop-reset").onclick = () => { generated = []; lastPick = null; render(); };
    input.addEventListener("input", () => { generated = []; lastPick = null; render(); });
    render();
  };

  // ================= 1 TOKENIZE =================
  S.tokenize = function () {
    const bpe = (L._bpe = L._bpe || new L.BPE(L.CORPUS, 400));
    const inp = U.$("#tok-input"), sl = U.$("#tok-merges"), out = U.$("#tok-merges-out");
    sl.max = bpe.merges.length;
    const list = U.$("#tok-mergelist");
    list.innerHTML = bpe.merges.map((m, i) => `<span data-i="${i}">${i + 1}. ${U.esc(m.a)}+${U.esc(m.b)}</span>`).join("");
    function render() {
      const n = +sl.value;
      out.textContent = n;
      const toks = bpe.encode(inp.value, n);
      U.$("#tok-chips").innerHTML = toks.map((t) => `<span class="chip${t.known ? "" : " unk"}" style="background:${tokColor(t.id)}" title="${t.known ? "id " + t.id : "not in vocabulary: byte fallback"}">${U.esc(t.text).replace("▁", '<span class="sp">▁</span>')}<small>${t.id}</small></span>`).join("");
      const chars = inp.value.replace(/\s+/g, " ").trim().length;
      U.$("#tok-stats").textContent = `${chars} characters → ${toks.length} tokens (${(chars / Math.max(1, toks.length)).toFixed(2)} characters per token) · vocabulary size ${bpe.baseCount + n} · ▁ marks the start of a word`;
      list.querySelectorAll("span").forEach((s) => s.classList.toggle("on", +s.dataset.i < n));
    }
    sl.addEventListener("input", render);
    inp.addEventListener("input", render);
    render();
  };

  // ================= 2 EMBED =================
  S.embed = function () {
    const words = Object.keys(L.EMB);
    const pca = L.pca2(words.map((w) => L.EMB[w]));
    const pts = words.map((w) => ({ w, p: pca.project(L.EMB[w]) }));
    const xs = pts.map((o) => o.p[0]), ys = pts.map((o) => o.p[1]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const X = (x) => 20 + ((x - x0) / (x1 - x0 || 1)) * 340, Y = (y) => 22 + ((y - y0) / (y1 - y0 || 1)) * 296;
    // Words with near-identical vectors land on the same spot; push their labels apart (layout only).
    pts.forEach((o) => (o.xy = [X(o.p[0]), Y(o.p[1])]));
    const jit = M.rng(3);
    for (let it = 0; it < 300; it++) {
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const a = pts[i].xy, b = pts[j].xy, dx = b[0] - a[0], dy = b[1] - a[1];
        if (Math.abs(dx) < 50 && Math.abs(dy) < 14) {
          const sy = dy === 0 ? (jit() < 0.5 ? -1 : 1) : Math.sign(dy), sx = dx === 0 ? (jit() < 0.5 ? -1 : 1) : Math.sign(dx);
          a[1] -= sy * 0.8; b[1] += sy * 0.8; a[0] -= sx * 0.4; b[0] += sx * 0.4;
        }
      }
      pts.forEach((o) => { o.xy[0] = Math.max(8, Math.min(380, o.xy[0])); o.xy[1] = Math.max(12, Math.min(332, o.xy[1])); });
    }
    const placed = (w) => pts.find((o) => o.w === w).xy;
    const catColor = (v) => {
      const k = v.indexOf(Math.max(...v.slice(0, 8)));
      return `hsl(${[275, 330, 210, 30, 50, 170, 120, 0][k] || 0} 60% 50%)`;
    };
    const svg = U.$("#emb-map");
    let selected = "queen", resultVec = null;
    function drawMap() {
      let s = "";
      pts.forEach(({ w, xy }) => {
        const on = w === selected;
        s += `<g data-w="${w}"><circle cx="${xy[0]}" cy="${xy[1]}" r="${on ? 5.5 : 3.5}" fill="${catColor(L.EMB[w])}" ${on ? 'stroke="var(--ink)" stroke-width="1.5"' : ""}/><text x="${xy[0] + 6}" y="${xy[1] + 3.5}" ${on ? 'style="font-weight:600"' : ""}>${w}</text></g>`;
      });
      if (resultVec) {
        // Mark the result at the position of its nearest word, since that is where it lands.
        const near = L.nearest(resultVec, 1)[0];
        const [rx, ry] = placed(near.w);
        s += `<g><circle cx="${rx}" cy="${ry}" r="8" fill="none" stroke="var(--pick)" stroke-width="2"/><text x="${rx - 10}" y="${ry - 9}" style="fill:var(--pick);font-weight:600" text-anchor="end">result</text></g>`;
      }
      svg.innerHTML = s;
    }
    svg.addEventListener("click", (e) => {
      const g = e.target.closest("g[data-w]");
      if (!g) return;
      selected = g.dataset.w;
      drawMap();
      drawVec(L.EMB[selected], selected);
    });
    function drawVec(v, name) {
      U.$("#emb-sel-name").textContent = "· " + name;
      U.$("#emb-vec").innerHTML = v.map((x, i) => `<div class="vec-cell"><div class="sw" style="background:${x >= 0 ? U.heat(x) : U.heat(-x, "--bad")}">${x.toFixed(1)}</div>${L.EMB_DIMS[i]}</div>`).join("");
    }
    function arith() {
      const a = U.$("#emb-a").value.trim().toLowerCase(), b = U.$("#emb-b").value.trim().toLowerCase(), c = U.$("#emb-c").value.trim().toLowerCase();
      const missing = [a, b, c].filter((w) => !L.EMB[w]);
      if (missing.length) {
        resultVec = null;
        U.$("#emb-result").innerHTML = `<p class="muted">"${U.esc(missing[0])}" is not in this page's ${words.length}-word vocabulary. Click the map to see the words it knows.</p>`;
      } else {
        resultVec = M.add(M.sub(L.EMB[a], L.EMB[b]), L.EMB[c]);
        const near = L.nearest(resultVec, 4, [a, b, c]);
        U.$("#emb-result").innerHTML = `<div class="muted" style="margin-bottom:4px">closest words (cosine similarity)</div>` + near.map((n) => `<div class="row"><span>${n.w}</span><span class="num">${n.s.toFixed(3)}</span></div>`).join("");
        drawVec(resultVec.map((x) => Math.round(x * 100) / 100), `${a} − ${b} + ${c}`);
      }
      drawMap();
    }
    ["#emb-a", "#emb-b", "#emb-c"].forEach((s) => U.$(s).addEventListener("input", arith));
    arith();
  };

  // ================= 3 ATTEND =================
  S.attend = function () {
    let head = "previous", sel = null;
    const inp = U.$("#att-input");
    U.seg(U.$("#att-heads"), Object.entries(L.HEADS).map(([k, h]) => [k, h.name]), head, (k) => { head = k; render(); });
    function render() {
      const toks = L.words(inp.value).slice(0, 14);
      if (toks.length < 2) { U.$("#att-heat").innerHTML = '<p class="muted">Type at least two words.</p>'; U.$("#att-detail").innerHTML = ""; return; }
      if (sel == null || sel >= toks.length) sel = toks.length - 1;
      const A = L.attention(toks, head);
      U.$("#att-blurb").textContent = L.HEADS[head].blurb.replace(/\*/g, "");
      const n = toks.length;
      const grid = U.$("#att-heat");
      grid.style.gridTemplateColumns = `auto repeat(${n}, 38px)`;
      let h = `<div></div>` + toks.map((t, j) => `<div class="lab top">${U.esc(t)}</div>`).join("");
      A.weights.forEach((row, i) => {
        h += `<div class="lab${i === sel ? " on" : ""}" data-i="${i}">${U.esc(toks[i])} →</div>`;
        row.forEach((w, j) => {
          const masked = A.raw[i][j] === -Infinity;
          h += masked ? `<div class="cell masked" title="masked"></div>` : `<div class="cell${i === sel ? " sel" : ""}" data-i="${i}" style="background:${U.heat(w)};color:${w > 0.55 ? "var(--surface)" : "var(--ink)"}" title="${toks[i]} → ${toks[j]}: ${(w * 100).toFixed(1)}%">${w >= 0.995 ? "1" : w.toFixed(2).replace(/^0/, "")}</div>`;
        });
      });
      grid.innerHTML = h;
      // Detail for the selected query row
      const i = sel;
      const rows = toks.map((t, j) => ({ t, j, raw: A.raw[i][j], sc: A.scaled[i][j], w: A.weights[i][j] })).filter((r) => r.raw !== -Infinity);
      const best = rows.reduce((a, b) => (b.w > a.w ? b : a));
      U.$("#att-detail").innerHTML = `<p class="mini-h">Query: <b style="color:var(--pick)">"${U.esc(toks[i])}"</b> (position ${i}) <span class="muted">· d = ${A.d}, so scores are divided by √${A.d} = ${Math.sqrt(A.d).toFixed(2)}</span></p>
        <table><thead><tr><th>key</th><th>q·k</th><th>÷√d</th><th>softmax weight</th></tr></thead><tbody>${rows.map((r) => `<tr><td>${r.j}: ${U.esc(r.t)}</td><td>${r.raw.toFixed(2)}</td><td>${r.sc.toFixed(2)}</td><td>${(r.w * 100).toFixed(1)}%<span class="wbar" style="width:${Math.round(r.w * 80)}px"></span></td></tr>`).join("")}</tbody></table>
        <p class="hint">"${U.esc(toks[i])}" takes ${(best.w * 100).toFixed(0)}% of its new vector from "${U.esc(best.t)}". Click any row label or cell to switch the query.</p>`;
    }
    U.$("#att-heat").addEventListener("click", (e) => {
      const c = e.target.closest("[data-i]");
      if (c) { sel = +c.dataset.i; render(); }
    });
    inp.addEventListener("input", () => { sel = null; render(); });
    render();
  };

  // ================= 4 PREDICT =================
  S.predict = function () {
    const inp = U.$("#pred-input"), t = U.$("#pred-temp"), k = U.$("#pred-topk"), p = U.$("#pred-topp");
    let hit = null;
    const params = () => ({ temperature: +t.value, topK: +k.value, topP: +p.value });
    function dist() {
      const d = ngram().distribution(inp.value);
      const pr = params();
      if (pr.temperature === 0) {
        // Greedy: all probability on the top token.
        const top = d.logits.indexOf(Math.max(...d.logits));
        const s = L.applySampling(d.logits, { ...pr, temperature: 1 });
        s.final = s.final.map((_, i) => (i === top ? 1 : 0));
        s.scaled = s.final.slice();
        s.keep = s.final.map((x) => x > 0);
        return { d, s };
      }
      return { d, s: L.applySampling(d.logits, pr) };
    }
    function render() {
      U.$("#pred-temp-out").textContent = (+t.value).toFixed(2) + ((+t.value) === 0 ? " greedy" : "");
      U.$("#pred-topk-out").textContent = +k.value ? k.value : "off";
      U.$("#pred-topp-out").textContent = (+p.value).toFixed(2);
      const { d, s } = dist();
      const ng = ngram();
      U.$("#pred-used").textContent = `Model used: ${d.used}. Solid bar = final probability, dashed outline = probability at temperature 1.`;
      const order = [...s.base.keys()].sort((a, b) => s.base[b] - s.base[a]).slice(0, 12);
      U.bars(U.$("#pred-bars"), order.map((i) => ({
        label: ng.vocab[i], value: s.final[i], ghost: s.base[i], text: U.pct(s.final[i]),
        cls: (!s.keep[i] || s.final[i] === 0 ? "cut" : "") + (i === hit ? " hit" : ""),
      })));
    }
    function sampleOne() {
      const { s } = dist();
      const i = M.sampleIndex(s.final);
      hit = i;
      render();
      return ngram().vocab[i];
    }
    U.$("#pred-sample").onclick = () => {
      const w = sampleOne();
      U.$("#pred-samples").innerHTML = `<span>→ ${U.esc(w)}</span>`;
    };
    U.$("#pred-sample20").onclick = () => {
      const counts = new Map();
      const { s } = dist();
      for (let n = 0; n < 20; n++) {
        const w = ngram().vocab[M.sampleIndex(s.final)];
        counts.set(w, (counts.get(w) || 0) + 1);
      }
      hit = null;
      render();
      U.$("#pred-samples").innerHTML = [...counts].sort((a, b) => b[1] - a[1]).map(([w, c]) => `<span>${U.esc(w)} ×${c}</span>`).join("");
    };
    [t, k, p].forEach((x) => x.addEventListener("input", () => { hit = null; render(); }));
    inp.addEventListener("input", () => { hit = null; U.$("#pred-samples").innerHTML = ""; render(); });
    render();
  };
})(window.LLM);
