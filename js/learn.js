// How models learn: the CharModel trained live in the Pretrain station.
// No DOM in this file; stations drive these models and draw what they report.
window.LLM = window.LLM || {};

(function (L) {
  const M = L.math;

  // A next-character model: one weight per (current char, next char) pair, trained by
  // full-batch gradient descent on cross-entropy loss over the corpus.
  class CharModel {
    // Build from raw text. Lowercases and keeps a–z, space and full stop: that is the vocabulary.
    static fromText(text, { seed = 7 } = {}) {
      return new CharModel(text.toLowerCase().replace(/\s+/g, " ").replace(/[^a-z .]/g, ""), seed);
    }

    constructor(clean, seed) {
      const chars = [...new Set(clean)].sort();
      const V = chars.length, ix = new Map(chars.map((c, i) => [c, i]));
      const C = [...Array(V)].map(() => new Float64Array(V));
      for (let i = 0; i < clean.length - 1; i++) C[ix.get(clean[i])][ix.get(clean[i + 1])]++;
      this.chars = chars;
      this._ix = ix;
      this._C = C;
      this._N = clean.length - 1;
      this._rowN = C.map((r) => r.reduce((a, b) => a + b, 0));
      // Lowest reachable loss: the entropy of the true next-char counts.
      let best = 0;
      C.forEach((r, a) => r.forEach((c, b) => { if (c) best -= c * Math.log(c / this._rowN[a]); }));
      this.bestLoss = best / this._N;
      this.uniformLoss = Math.log(V);
      const rng = M.rng(seed);
      this._W = [...Array(V)].map(() => new Float64Array(V).map(() => (rng() - 0.5) * 0.2));
      this.steps = 0;
      this._forward();
    }

    get size() { return this.chars.length; }
    // Raw loss; Infinity or NaN once training has blown up.
    get loss() { return this._loss; }
    get diverged() { return !Number.isFinite(this._loss) || this._loss > this.uniformLoss + 0.5; }
    // probs[a][b] = P(next char b | current char a)
    get probs() { return this._P; }

    step(lr, n = 1) {
      const V = this.size, W = this._W, C = this._C, rowN = this._rowN, N = this._N;
      for (let k = 0; k < n; k++) {
        const P = this._P; // fresh predictions each step; _forward() replaces them
        // Gradient of mean cross-entropy w.r.t. each weight: (predicted − actual) counts / N.
        for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) W[a][b] -= lr * V * (rowN[a] * P[a][b] - C[a][b]) / N;
        this.steps++;
        this._forward();
      }
      return this;
    }

    sample(rng, length = 150, start = "t") {
      let c = this._ix.has(start) ? this._ix.get(start) : 0, s = this.chars[c];
      for (let i = 0; i < length; i++) {
        c = M.sampleIndex(this._P[c], rng());
        s += this.chars[c];
      }
      return s;
    }

    _forward() {
      let loss = 0;
      this._P = this._W.map((row, a) => {
        const p = M.softmax(Array.from(row));
        for (let b = 0; b < p.length; b++) if (this._C[a][b]) loss -= this._C[a][b] * Math.log(Math.max(p[b], 1e-12));
        return p;
      });
      this._loss = loss / this._N;
    }
  }

  L.CharModel = CharModel;
})(window.LLM);
