// Core math and the toy models. No libraries: every number on screen is computed here.
window.LLM = window.LLM || {};

(function (L) {
  // ---------- small math helpers ----------
  const M = (L.math = {});
  M.softmax = function (xs, temperature = 1) {
    const t = Math.max(temperature, 1e-6);
    let max = -Infinity;
    for (const x of xs) if (x > max) max = x;
    const ex = xs.map((x) => (x === -Infinity ? 0 : Math.exp((x - max) / t)));
    const s = ex.reduce((a, b) => a + b, 0) || 1;
    return ex.map((e) => e / s);
  };
  M.dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  M.norm = (a) => Math.sqrt(M.dot(a, a));
  M.cos = (a, b) => {
    const n = M.norm(a) * M.norm(b);
    return n ? M.dot(a, b) / n : 0;
  };
  M.add = (a, b) => a.map((x, i) => x + b[i]);
  M.sub = (a, b) => a.map((x, i) => x - b[i]);
  M.scale = (a, k) => a.map((x) => x * k);
  M.sigmoid = (x) => 1 / (1 + Math.exp(-x));
  M.mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
  M.std = (a) => {
    const m = M.mean(a);
    return Math.sqrt(M.mean(a.map((x) => (x - m) * (x - m))));
  };
  // Seeded RNG (mulberry32) so demos are reproducible when needed.
  M.rng = function (seed) {
    let s = seed >>> 0;
    return function () {
      s = (s + 0x6d2b79f5) >>> 0;
      let t = s;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  };
  M.hash = function (str) {
    let h = 2166136261;
    for (let i = 0; i < str.length; i++) {
      h ^= str.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return h >>> 0;
  };
  M.sampleIndex = function (probs, r = Math.random()) {
    let acc = 0;
    for (let i = 0; i < probs.length; i++) {
      acc += probs[i];
      if (r < acc) return i;
    }
    return probs.length - 1;
  };

  // ---------- word splitting ----------
  L.words = function (text) {
    return (text.toLowerCase().match(/[a-z0-9']+|[.,!?]/g) || []);
  };

  // ---------- Byte-pair encoding (BPE) ----------
  // Trained on the corpus exactly the way GPT-style tokenizers are, at toy scale:
  // start from characters, repeatedly merge the most frequent adjacent pair.
  L.BPE = function (corpus, maxMerges = 400) {
    const wordFreq = new Map();
    for (const w of corpus.toLowerCase().split(/\s+/)) {
      if (!w) continue;
      const k = "▁" + w; // ▁ marks "this piece starts a new word", as in SentencePiece
      wordFreq.set(k, (wordFreq.get(k) || 0) + 1);
    }
    let splits = [...wordFreq.keys()].map((w) => ({ parts: [...w], f: wordFreq.get(w) }));
    const baseChars = new Set();
    splits.forEach((s) => s.parts.forEach((c) => baseChars.add(c)));
    const vocab = [...baseChars].sort();
    const merges = [];
    for (let m = 0; m < maxMerges; m++) {
      const pairs = new Map();
      for (const s of splits) {
        for (let i = 0; i < s.parts.length - 1; i++) {
          const k = s.parts[i] + "\u0000" + s.parts[i + 1];
          pairs.set(k, (pairs.get(k) || 0) + s.f);
        }
      }
      let best = null, bestN = 1;
      for (const [k, n] of pairs) if (n > bestN) { best = k; bestN = n; }
      if (!best) break;
      const [a, b] = best.split("\u0000");
      merges.push({ a, b, merged: a + b, count: bestN });
      vocab.push(a + b);
      for (const s of splits) {
        const out = [];
        for (let i = 0; i < s.parts.length; i++) {
          if (i < s.parts.length - 1 && s.parts[i] === a && s.parts[i + 1] === b) {
            out.push(a + b);
            i++;
          } else out.push(s.parts[i]);
        }
        s.parts = out;
      }
    }
    this.merges = merges;
    this.vocab = vocab;
    this.baseCount = baseChars.size;
  };
  L.BPE.prototype.encode = function (text, nMerges) {
    const n = Math.min(nMerges, this.merges.length);
    const ranks = new Map();
    for (let i = 0; i < n; i++) ranks.set(this.merges[i].a + "\u0000" + this.merges[i].b, i);
    const idOf = new Map(this.vocab.slice(0, this.baseCount + n).map((v, i) => [v, i]));
    const out = [];
    for (const w of text.toLowerCase().split(/\s+/)) {
      if (!w) continue;
      let parts = [..."▁" + w];
      // Apply the earliest-learned merge first, exactly like GPT-2's tokenizer.
      while (parts.length > 1) {
        let bestI = -1, bestR = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = ranks.get(parts[i] + "\u0000" + parts[i + 1]);
          if (r !== undefined && r < bestR) { bestR = r; bestI = i; }
        }
        if (bestI < 0) break;
        parts.splice(bestI, 2, parts[bestI] + parts[bestI + 1]);
      }
      for (const p of parts) {
        const id = idOf.has(p) ? idOf.get(p) : 1000 + p.charCodeAt(0); // unknown char: byte fallback
        out.push({ text: p, id, known: idOf.has(p) });
      }
    }
    return out;
  };

  // ---------- Trigram language model ----------
  // A counting model: P(next | previous two words). Real LLMs replace this lookup table with a
  // transformer, but the output is the same kind of object: one probability per vocabulary item.
  L.NGram = function (corpus) {
    const toks = L.words(corpus);
    this.tokens = toks;
    this.vocab = [...new Set(toks)].sort();
    this.index = new Map(this.vocab.map((w, i) => [w, i]));
    this.uni = new Map();
    this.bi = new Map();
    this.tri = new Map();
    const inc = (map, k, w) => {
      let m = map.get(k);
      if (!m) map.set(k, (m = new Map()));
      m.set(w, (m.get(w) || 0) + 1);
    };
    for (let i = 0; i < toks.length; i++) {
      this.uni.set(toks[i], (this.uni.get(toks[i]) || 0) + 1);
      if (i >= 1) inc(this.bi, toks[i - 1], toks[i]);
      if (i >= 2) inc(this.tri, toks[i - 2] + " " + toks[i - 1], toks[i]);
    }
    this.total = toks.length;
  };
  L.NGram.prototype.distribution = function (context) {
    const w = L.words(context);
    const a = w[w.length - 2], b = w[w.length - 1];
    const tri = a && b ? this.tri.get(a + " " + b) : null;
    const bi = b ? this.bi.get(b) : null;
    const sum = (m) => (m ? [...m.values()].reduce((s, x) => s + x, 0) : 0);
    const triN = sum(tri), biN = sum(bi);
    let l3 = tri ? 0.7 : 0, l2 = bi ? (tri ? 0.25 : 0.9) : 0;
    let l1 = 1 - l3 - l2;
    const probs = this.vocab.map((v) => {
      const p3 = tri ? (tri.get(v) || 0) / triN : 0;
      const p2 = bi ? (bi.get(v) || 0) / biN : 0;
      const p1 = (this.uni.get(v) || 0) / this.total;
      return l3 * p3 + l2 * p2 + l1 * p1;
    });
    return {
      probs,
      logits: probs.map((p) => Math.log(p)),
      used: tri ? `trigram "${a} ${b} → ?" (seen ${triN}×)` : bi ? `bigram "${b} → ?" (seen ${biN}×)` : "unigram (no context match)",
      counts: (v) => ({ tri: tri ? tri.get(v) || 0 : 0, bi: bi ? bi.get(v) || 0 : 0 }),
    };
  };

  // Turn logits into the final sampling distribution, the same way every LLM API does.
  L.applySampling = function (logits, { temperature = 1, topK = 0, topP = 1 }) {
    const n = logits.length;
    const base = M.softmax(logits, 1);
    const scaled = M.softmax(logits, temperature);
    const order = [...Array(n).keys()].sort((i, j) => scaled[j] - scaled[i]);
    const keep = new Array(n).fill(true);
    if (topK > 0) order.forEach((idx, rank) => { if (rank >= topK) keep[idx] = false; });
    if (topP < 1) {
      let acc = 0;
      for (const idx of order) {
        if (!keep[idx]) continue;
        if (acc >= topP) keep[idx] = false;
        acc += scaled[idx];
      }
    }
    const kept = scaled.map((p, i) => (keep[i] ? p : 0));
    const s = kept.reduce((a, b) => a + b, 0) || 1;
    return { base, scaled, final: kept.map((p) => p / s), keep, order };
  };

  // ---------- Embeddings ----------
  // Hand-built vectors with *named* dimensions so you can read them. Real models learn
  // thousands of unnamed dimensions from data; the arithmetic is the same.
  L.EMB_DIMS = ["royal", "female", "male", "animal", "food", "place", "nature", "machine", "young", "feeling"];
  const E = {
    king: [1, 0, 1, 0, 0, 0, 0, 0, 0, 0], queen: [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    prince: [0.85, 0, 1, 0, 0, 0, 0, 0, 0.8, 0], princess: [0.85, 1, 0, 0, 0, 0, 0, 0, 0.8, 0],
    man: [0, 0, 1, 0, 0, 0, 0, 0, 0, 0], woman: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    boy: [0, 0, 1, 0, 0, 0, 0, 0, 0.9, 0], girl: [0, 1, 0, 0, 0, 0, 0, 0, 0.9, 0],
    crown: [0.9, 0, 0, 0, 0, 0, 0, 0.2, 0, 0], castle: [0.8, 0, 0, 0, 0, 0.9, 0, 0, 0, 0],
    sword: [0.5, 0, 0.3, 0, 0, 0, 0, 0.4, 0, 0],
    cat: [0, 0, 0, 1, 0, 0, 0.3, 0, 0, 0], dog: [0, 0, 0, 1, 0, 0, 0.3, 0, 0, 0.1],
    kitten: [0, 0, 0, 1, 0, 0, 0.3, 0, 0.9, 0], puppy: [0, 0, 0, 1, 0, 0, 0.3, 0, 0.9, 0.1],
    horse: [0.2, 0, 0, 1, 0, 0, 0.4, 0, 0, 0], bird: [0, 0, 0, 1, 0, 0, 0.6, 0, 0, 0],
    milk: [0, 0, 0, 0.2, 1, 0, 0, 0, 0, 0], bread: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    apples: [0, 0, 0, 0, 1, 0, 0.4, 0, 0, 0], cheese: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    bone: [0, 0, 0, 0.4, 0.6, 0, 0, 0, 0, 0],
    park: [0, 0, 0, 0, 0, 0.8, 0.6, 0, 0, 0], town: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    market: [0, 0, 0, 0, 0.4, 0.9, 0, 0, 0, 0], field: [0, 0, 0, 0, 0, 0.6, 0.8, 0, 0, 0],
    river: [0, 0, 0, 0, 0, 0.4, 1, 0, 0, 0], sea: [0, 0, 0, 0, 0, 0.4, 1, 0, 0, 0],
    tree: [0, 0, 0, 0, 0, 0, 1, 0, 0, 0], rain: [0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0],
    sun: [0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0.2], moon: [0, 0, 0, 0, 0, 0, 0.9, 0, 0, 0],
    model: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0], computer: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    robot: [0, 0, 0, 0.1, 0, 0, 0, 1, 0, 0], program: [0, 0, 0, 0, 0, 0, 0, 0.9, 0, 0],
    token: [0, 0, 0, 0, 0, 0, 0, 0.8, 0, 0],
    happy: [0, 0, 0, 0, 0, 0, 0, 0, 0, 1], sad: [0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
    brave: [0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0.7], warm: [0, 0, 0, 0, 0.1, 0, 0.3, 0, 0, 0.5],
  };
  L.EMB = E;
  L.embed = function (word) {
    const w = word.toLowerCase();
    if (E[w]) return { vec: E[w].slice(), known: true };
    // Unknown words get a small deterministic pseudo-random vector (a stand-in for an untrained row).
    const r = M.rng(M.hash(w));
    return { vec: L.EMB_DIMS.map(() => (r() - 0.5) * 0.3), known: false };
  };
  L.nearest = function (vec, k = 5, exclude = []) {
    return Object.keys(E)
      .filter((w) => !exclude.includes(w))
      .map((w) => ({ w, s: M.cos(vec, E[w]) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, k);
  };
  // 2D projection with PCA (power iteration) so related words land near each other.
  L.pca2 = function (vectors) {
    const n = vectors.length, d = vectors[0].length;
    const mean = new Array(d).fill(0);
    vectors.forEach((v) => v.forEach((x, i) => (mean[i] += x / n)));
    const X = vectors.map((v) => v.map((x, i) => x - mean[i]));
    const cov = [...Array(d)].map((_, i) => [...Array(d)].map((_, j) => X.reduce((s, r) => s + r[i] * r[j], 0) / n));
    const comps = [];
    for (let c = 0; c < 2; c++) {
      let v = [...Array(d)].map((_, i) => Math.sin(i + 1 + c * 3));
      for (let it = 0; it < 200; it++) {
        let w = cov.map((row) => M.dot(row, v));
        comps.forEach((u) => { const p = M.dot(w, u); w = w.map((x, i) => x - p * u[i]); });
        const nn = M.norm(w) || 1;
        v = w.map((x) => x / nn);
      }
      comps.push(v);
    }
    return { project: (v) => comps.map((u) => M.dot(v.map((x, i) => x - mean[i]), u)) };
  };

  // ---------- Attention ----------
  // Three heads with hand-designed query/key maps. The computation itself — q·k/√d, causal mask,
  // softmax, weighted sum of values — is exactly what runs inside every transformer layer.
  const FREQS = [0.9, 0.45, 0.2, 0.1];
  const posVec = (p) => FREQS.flatMap((w) => [Math.cos(w * p), Math.sin(w * p)]);
  // A random unit vector per distinct word: identical words match (dot = 1), different words barely do.
  const idVec = (w) => {
    const r = M.rng(M.hash("id:" + w));
    const v = [...Array(16)].map(() => r() - 0.5);
    const n = M.norm(v);
    return v.map((x) => (x / n) * 5);
  };
  L.HEADS = {
    previous: {
      name: "Previous-word head",
      blurb: "Query = position of (me − 1), key = my position. The dot product of two position codes peaks when they match, so each word looks one step back. Real models learn heads like this in the first layers.",
      q: (toks, i) => posVec(i - 1).map((x) => x * 4),
      k: (toks, j) => posVec(j).map((x) => x * 4),
    },
    meaning: {
      name: "Related-meaning head",
      blurb: "Query and key are the word embeddings themselves, so a word attends to earlier words with similar meaning (queen → king, dog → cat). This head is set up to skip the word itself.",
      noSelf: true,
      q: (toks, i) => L.embed(toks[i]).vec.map((x) => x * 4),
      k: (toks, j) => L.embed(toks[j]).vec.map((x) => x * 4),
    },
    induction: {
      name: "Induction head",
      blurb: "The key of each word carries the identity of the word before it (copied there by a previous-word head in an earlier layer). So when a word repeats, it attends to whatever followed its earlier appearance. This is how models continue patterns from earlier in the prompt.",
      noSelf: true,
      q: (toks, i) => idVec(toks[i]),
      k: (toks, j) => (j === 0 ? idVec("") .map(() => 0) : idVec(toks[j - 1])),
    },
  };
  L.attention = function (toks, headKey) {
    const H = L.HEADS[headKey];
    const Q = toks.map((_, i) => H.q(toks, i));
    const K = toks.map((_, j) => H.k(toks, j));
    const d = Q[0].length;
    // Causal mask: position i may only look at j <= i (j < i for heads that skip themselves).
    const visible = (i, j) => (H.noSelf && i > 0 ? j < i : j <= i);
    const raw = Q.map((q, i) => K.map((k, j) => (visible(i, j) ? M.dot(q, k) : -Infinity)));
    const scaled = raw.map((row) => row.map((x) => (x === -Infinity ? x : x / Math.sqrt(d))));
    const weights = scaled.map((row) => M.softmax(row));
    return { Q, K, raw, scaled, weights, d };
  };

  // ---------- TF-IDF retrieval (the "R" in RAG) ----------
  const STOP = new Set("a an the of to and or in on for is are was were be by with as it its that this at from how what which who why when do does can into than then them they their there these those not but so if also such each per about".split(" "));
  const terms = (t) => (t.toLowerCase().match(/[a-z0-9]+/g) || []).filter((w) => !STOP.has(w) && w.length > 1).map((w) => w.replace(/(ing|es|s)$/, ""));
  L.Retriever = function (docs) {
    this.docs = docs;
    this.tf = docs.map((d) => {
      const m = new Map();
      terms(d.title + " " + d.text).forEach((w) => m.set(w, (m.get(w) || 0) + 1));
      return m;
    });
    const df = new Map();
    this.tf.forEach((m) => m.forEach((_, w) => df.set(w, (df.get(w) || 0) + 1)));
    this.idf = (w) => Math.log((docs.length + 1) / ((df.get(w) || 0) + 1)) + 1;
  };
  L.Retriever.prototype.search = function (query) {
    const q = new Map();
    terms(query).forEach((w) => q.set(w, (q.get(w) || 0) + 1));
    const vec = (m) => { const v = new Map(); m.forEach((c, w) => v.set(w, c * this.idf(w))); return v; };
    const qv = vec(q);
    const qn = Math.sqrt([...qv.values()].reduce((s, x) => s + x * x, 0)) || 1;
    return this.docs.map((d, i) => {
      const dv = vec(this.tf[i]);
      const dn = Math.sqrt([...dv.values()].reduce((s, x) => s + x * x, 0)) || 1;
      let s = 0;
      const hits = [];
      qv.forEach((x, w) => { if (dv.has(w)) { s += x * dv.get(w); hits.push(w); } });
      return { doc: d, score: s / (qn * dn), hits };
    }).sort((a, b) => b.score - a.score);
  };

  // Rough token estimate (≈ 4 characters per token for English).
  L.estimateTokens = (text) => Math.max(0, Math.round(text.length / 4));
})(window.LLM);
