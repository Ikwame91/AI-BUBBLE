// Part 2: how a model is made — live pretraining, alignment stages, reward model, reasoning.
window.LLM = window.LLM || {};

(function (L) {
  const U = L.ui, M = L.math;
  const S = (L.sections = L.sections || {});
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  // ================= 5 PRETRAIN: a character model trained by gradient descent =================
  S.train = function () {
    const txt = L.CORPUS.toLowerCase().replace(/\s+/g, " ").replace(/[^a-z .]/g, "");
    const chars = [...new Set(txt)].sort();
    const V = chars.length, ix = new Map(chars.map((c, i) => [c, i]));
    const C = [...Array(V)].map(() => new Float64Array(V));
    for (let i = 0; i < txt.length - 1; i++) C[ix.get(txt[i])][ix.get(txt[i + 1])]++;
    const N = txt.length - 1, rowN = C.map((r) => r.reduce((a, b) => a + b, 0));
    let best = 0;
    C.forEach((r, a) => r.forEach((c, b) => { if (c) best -= c * Math.log(c / rowN[a]); }));
    best /= N;
    const uniform = Math.log(V);
    let W, P, steps, losses, timer = null, rng = M.rng(7);

    function forward() {
      let loss = 0;
      P = W.map((row, a) => {
        const p = M.softmax(Array.from(row));
        for (let b = 0; b < V; b++) if (C[a][b]) loss -= C[a][b] * Math.log(Math.max(p[b], 1e-12));
        return p;
      });
      return loss / N;
    }
    function reset() {
      W = [...Array(V)].map(() => new Float64Array(V).map(() => (rng() - 0.5) * 0.2));
      steps = 0;
      losses = [[0, forward()]];
      draw();
    }
    function step(n) {
      const lr = +U.$("#tr-lr").value;
      for (let k = 0; k < n; k++) {
        // Gradient of mean cross-entropy w.r.t. each weight: (predicted − actual) counts / N.
        for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) W[a][b] -= lr * V * (rowN[a] * P[a][b] - C[a][b]) / N;
        steps++;
        const l = forward();
        losses.push([steps, isFinite(l) ? l : 99]);
      }
      draw();
    }
    function sample(len = 150) {
      let c = ix.get("t"), s = "t";
      for (let i = 0; i < len; i++) {
        c = M.sampleIndex(P[c], rng());
        s += chars[c];
      }
      return s;
    }
    function draw() {
      const last = losses[losses.length - 1][1];
      const xMax = Math.max(100, Math.ceil(steps / 100) * 100);
      U.lineChart(U.$("#tr-loss"), {
        xMax, yMin: 1.5, yMax: 4,
        yTicks: [2, 3, 4],
        series: [{ points: losses, color: css("--accent") || "#2a3acf", area: true }],
        hlines: [
          { y: uniform, label: `random guess ${uniform.toFixed(2)}` },
          { y: best, label: `best this model can do ${best.toFixed(2)}`, color: "var(--good)" },
        ],
        xLabel: `step ${steps}`,
      });
      const diverged = last > uniform + 0.5;
      U.$("#tr-stats").innerHTML = `step ${steps} · loss ${last > 50 ? "exploding" : last.toFixed(3)} · ${V * V} weights` + (diverged ? ` · <span style="color:var(--bad)">learning rate too high: updates overshoot</span>` : "");
      // Probability heatmap
      const cv = U.$("#tr-heat"), g = cv.getContext("2d");
      const W0 = cv.width, pad = 14, cell = (W0 - pad) / V;
      g.clearRect(0, 0, W0, W0);
      g.fillStyle = css("--sunk");
      g.fillRect(0, 0, W0, W0);
      g.fillStyle = css("--muted");
      g.font = "9px IBM Plex Mono, monospace";
      g.textAlign = "center";
      chars.forEach((ch, i) => {
        const t = ch === " " ? "␣" : ch;
        g.fillText(t, pad + i * cell + cell / 2, 10);
        g.fillText(t, 6, pad + i * cell + cell / 2 + 3);
      });
      const acc = css("--accent");
      for (let a = 0; a < V; a++) for (let b = 0; b < V; b++) {
        g.globalAlpha = Math.min(1, Math.sqrt(P[a][b]) * 1.4);
        g.fillStyle = acc;
        g.fillRect(pad + b * cell, pad + a * cell, cell - 1, cell - 1);
      }
      g.globalAlpha = 1;
      U.$("#tr-sample").textContent = sample();
    }
    function stop() { clearInterval(timer); timer = null; U.$("#tr-run").textContent = "Train"; }
    U.$("#tr-run").onclick = () => {
      if (timer) return stop();
      U.$("#tr-run").textContent = "Pause";
      timer = setInterval(() => { step(3); if (steps >= 600) stop(); }, 40);
    };
    U.$("#tr-step").onclick = () => step(10);
    U.$("#tr-reset").onclick = () => { stop(); reset(); };
    L.onTheme.push(draw);
    reset();
  };

  // ================= 6 ALIGN: stages =================
  const STAGES = {
    base: {
      label: "1 · Pretrained base",
      data: "Trillions of tokens of web pages, books, papers and code.",
      goal: "Predict the next token. Nothing else.",
      gets: "A model that knows a lot but only continues documents. GPT-3 (2020) and the Llama \"base\" checkpoints are examples.",
      chat: [["user", "Is 1001 a prime number?"], ["model", "Is 1003 a prime number? Is 1007 a prime number? Is 1009 a prime number?\n\nAnswers on page 214."]],
      why: "On the web, a question is often followed by more questions (a worksheet, a quiz), so continuing the pattern is the likely text.",
    },
    sft: {
      label: "2 · Instruction-tuned (SFT)",
      data: "Tens of thousands of example conversations written by people: prompt → ideal answer.",
      goal: "Same next-token loss, but only on the answer part, in a chat format.",
      gets: "A model that answers in the assistant role. It copies the style of good answers, which does not guarantee it gets them right.",
      chat: [["user", "Is 1001 a prime number?"], ["model", "Yes, 1001 is a prime number."]],
      why: "It learned the shape of an answer (direct, confident) but nothing pushed it to check its work. Confident-and-wrong is a classic hallucination.",
    },
    rlhf: {
      label: "3 · Preference-tuned (RLHF / RLAIF)",
      data: "Pairs of answers ranked by people (RLHF) or by an AI judge following written principles (Constitutional AI).",
      goal: "Maximize a learned reward model's score, without drifting too far from the SFT model.",
      gets: "Helpful, polite, more careful answers that refuse clearly harmful requests. Side effects can include flattery and over-long answers when raters liked those.",
      chat: [["user", "Is 1001 a prime number?"], ["model", "No. 1001 = 7 × 11 × 13, so it has divisors other than 1 and itself.\n\nIt's a fun one: 1001 is the number in \"One Thousand and One Nights\"."]],
      why: "Raters preferred answers that were correct and showed why, so the reward model learned to score that higher.",
    },
    reason: {
      label: "4 · Reasoning (RL on checkable tasks)",
      data: "Problems with verifiable answers: math with known solutions, code with unit tests.",
      goal: "Reward correct final answers (and a readable format). The model learns on its own to think longer, check work, and backtrack.",
      gets: "Models like OpenAI o1, DeepSeek-R1 and Claude with extended thinking, which spend thinking tokens before answering.",
      chat: [["user", "Is 1001 a prime number?"], ["think", "Odd, so not divisible by 2. Digit sum 1+0+0+1 = 2, not divisible by 3. Doesn't end in 0 or 5. Try 7: 7 × 143 = 1001. ✓ So it is composite. Check 143: 11 × 13 = 143. ✓"], ["model", "No. 1001 = 7 × 11 × 13."]],
      why: "During RL, chains of thought that ended in correct answers were reinforced. Checking divisors step by step is one such habit.",
    },
  };
  S.align = function () {
    const view = U.$("#al-view");
    function show(k) {
      const s = STAGES[k];
      view.innerHTML = `<div class="what">
          <p><span class="tag">training data</span><br>${s.data}</p>
          <p><span class="tag">objective</span><br>${s.goal}</p>
          <p><span class="tag">result</span><br>${s.gets}</p>
        </div>
        <div>
          <div class="chat">${s.chat.map(([r, t]) => `<div class="msg ${r === "user" ? "" : r === "think" ? "model think" : "model"}"><b>${r === "think" ? "thinking (hidden or summarized in apps)" : r === "user" ? "prompt" : "output"}</b>${U.esc(t).replace(/\n/g, "<br>")}</div>`).join("")}</div>
          <p class="hint"><b>Why it answers this way:</b> ${s.why}<br>Illustrative example written for this page.</p>
        </div>`;
    }
    U.seg(U.$("#al-stages"), Object.entries(STAGES).map(([k, s]) => [k, s.label]), "base", show);
    show("base");
    rewardModel();
  };

  // ================= 6b reward model you train by labeling =================
  const TRAITS = ["correct", "detailed", "friendly", "long", "flatters you"];
  const PAIRS = [
    { q: "How do I reverse a list in Python?",
      a: ["Use my_list.reverse() to reverse it in place, or my_list[::-1] to get a reversed copy.", [1, 0.6, 0.3, 0.2, 0]],
      b: ["Great question! You're clearly a thoughtful programmer. Lists are one of Python's most loved features, with a long and fascinating history. To reverse one, use sorted(my_list, reverse=True).", [0, 0.4, 0.9, 0.8, 1]] },
    { q: "Is it safe to eat raw chicken?",
      a: ["Sure, lots of people do it. It's fine.", [0, 0.1, 0.6, 0.15, 0]],
      b: ["No. Raw chicken often carries Salmonella and Campylobacter. Cook it until the inside reaches 74 °C (165 °F).", [1, 0.7, 0.3, 0.25, 0]] },
    { q: "Explain what a token is.",
      a: ["A token is a chunk of text, often a word or part of a word, that a model reads as one unit.", [1, 0.5, 0.4, 0.2, 0]],
      b: ["What a brilliant question, you have great instincts! A token is a chunk of text the model treats as one unit. Tokenizers learn these chunks from data with byte-pair encoding: common words become one token, rare words split into several, so 'unbelievable' might become 'un', 'believ', 'able'. Pricing and context limits are counted in tokens.", [1, 0.9, 0.8, 1, 1]] },
    { q: "What's 17 × 6?",
      a: ["17 × 6 = 10 × 6 + 7 × 6 = 60 + 42 = 102.", [1, 0.6, 0.4, 0.3, 0]],
      b: ["102.", [1, 0.1, 0.2, 0.05, 0]] },
    { q: "Write a one-line motto for a bakery.",
      a: ["Here's one, and may I say your bakery sounds amazing: \"Bread.\"", [0.4, 0, 0.7, 0.2, 1]],
      b: ["\"Fresh from our oven to your morning.\"", [1, 0.3, 0.6, 0.1, 0]] },
    { q: "When was the first crewed Moon landing?",
      a: ["July 20, 1969: Apollo 11's lunar module landed and Neil Armstrong stepped out a few hours later.", [1, 0.7, 0.3, 0.3, 0]],
      b: ["I think it was sometime in the 1970s, but I'm not totally sure.", [0, 0.1, 0.5, 0.2, 0]] },
  ];
  const BON = { q: "How should I store fresh basil?", c: [
    ["Trim the stems and stand it in a glass of water on the counter, loosely covered. Avoid the fridge: cold turns the leaves black.", [1, 0.7, 0.4, 0.3, 0]],
    ["Wow, what a fantastic question from someone who clearly loves cooking! Basil is a beloved herb, from Italian pesto to Thai curries, with a history going back thousands of years. Just put it in the fridge.", [0, 0.3, 0.9, 1, 1]],
    ["Fridge.", [0, 0, 0, 0, 0]],
    ["Put it in water like flowers, on the counter.", [1, 0.4, 0.5, 0.1, 0]],
  ] };
  function rewardModel() {
    const prefs = []; // {win: features, lose: features}
    let idx = 0, w = TRAITS.map(() => 0);
    function fit() {
      // Bradley–Terry: P(winner preferred) = sigmoid(w · (f_win − f_lose)). Gradient ascent with light L2.
      w = TRAITS.map(() => 0);
      for (let it = 0; it < 400; it++) {
        const g = w.map((x) => -0.05 * x);
        prefs.forEach(({ win, lose }) => {
          const d = M.sub(win, lose);
          const p = M.sigmoid(M.dot(w, d));
          d.forEach((x, i) => (g[i] += (1 - p) * x));
        });
        w = w.map((x, i) => x + 0.3 * g[i]);
      }
    }
    function render() {
      const pr = PAIRS[idx % PAIRS.length];
      const flip = Math.floor(idx / PAIRS.length) % 2 === 1;
      const [x, y] = flip ? [pr.b, pr.a] : [pr.a, pr.b];
      U.$("#rm-pair").innerHTML = `<p class="pair-q"><span class="tag">prompt ${(idx % PAIRS.length) + 1}/${PAIRS.length}</span> ${U.esc(pr.q)}</p>
        <div class="pair"><button class="choice" data-c="0">${U.esc(x[0])}</button><button class="choice" data-c="1">${U.esc(y[0])}</button></div>
        <p class="hint">Click the answer you prefer.</p>`;
      U.$("#rm-pair").querySelectorAll(".choice").forEach((b) => (b.onclick = () => {
        const win = b.dataset.c === "0" ? x : y, lose = b.dataset.c === "0" ? y : x;
        prefs.push({ win: win[1], lose: lose[1] });
        idx++;
        fit();
        render();
      }));
      const mx = Math.max(0.01, ...w.map(Math.abs));
      U.$("#rm-count").textContent = `· from ${prefs.length} comparison${prefs.length === 1 ? "" : "s"}`;
      U.bars(U.$("#rm-weights"), TRAITS.map((t, i) => ({ label: t, value: Math.abs(w[i]) / mx, text: (w[i] >= 0 ? "+" : "") + w[i].toFixed(2), cls: w[i] < 0 ? "neg" : "" })));
      const scores = BON.c.map(([, f]) => M.dot(w, f));
      const top = prefs.length ? scores.indexOf(Math.max(...scores)) : -1;
      U.$("#rm-bon").innerHTML = `<p class="pair-q" style="font-size:13px"><span class="tag">new prompt</span> ${BON.q}</p><div class="bon">${BON.c.map(([t], i) => `<div class="row${i === top ? " win" : ""}"><span>${U.esc(t)}</span><span class="num">${prefs.length ? scores[i].toFixed(2) : "–"}</span></div>`).join("")}</div>
        <p class="hint">${prefs.length ? "The highlighted answer has the highest reward. RL training pushes the model toward answers like it." : "Label a few pairs and the reward model will start ranking these."}</p>`;
    }
    render();
  }

  // ================= 7 REASON: self-consistency =================
  const PATHS = [
    { a: "$0.05", ok: true, p: 0.55, t: ["Let the ball be x. The bat is x + 1.00. So 2x + 1.00 = 1.10, 2x = 0.10, x = 0.05.", "Ball = b, bat = b + 1. Total 2b + 1 = 1.10 → b = 0.05. Check: 0.05 + 1.05 = 1.10 ✓", "If the ball were 0.10 the bat would be 1.10 and the total 1.20. Too much. Ball = 0.05, bat = 1.05 ✓"] },
    { a: "$0.10", ok: false, p: 0.35, t: ["The bat costs $1.00, so the ball is 1.10 − 1.00 = 0.10.", "Subtract: 1.10 − 1.00 = 0.10. The ball is ten cents.", "Total is 1.10 and the bat is 1.00 more, so the ball is 0.10."] },
    { a: "$0.55", ok: false, p: 0.1, t: ["Split the total evenly: 1.10 ÷ 2 = 0.55.", "They share the cost: 0.55 each, so the ball is 0.55."] },
  ];
  function drawPath(r) {
    let acc = 0;
    for (const p of PATHS) { acc += p.p; if (r < acc) return p; }
    return PATHS[0];
  }
  function vote(n, rand) {
    const counts = new Map();
    for (let i = 0; i < n; i++) { const a = drawPath(rand()).a; counts.set(a, (counts.get(a) || 0) + 1); }
    const max = Math.max(...counts.values());
    const tied = [...counts].filter(([, c]) => c === max).map(([a]) => a);
    return { counts, winner: tied[Math.floor(rand() * tied.length)] };
  }
  S.reason = function () {
    const sl = U.$("#sc-n");
    // Accuracy curve: Monte Carlo over 10,000 runs per n.
    const rand = M.rng(42);
    const curve = [];
    for (let n = 1; n <= 25; n++) {
      let ok = 0;
      for (let t = 0; t < 10000; t++) if (vote(n, rand).winner === "$0.05") ok++;
      curve.push([n, ok / 100]);
    }
    function drawCurve() {
      const n = +sl.value;
      U.lineChart(U.$("#sc-curve"), {
        xMax: 25, yMin: 40, yMax: 100, yTicks: [50, 75, 100],
        series: [{ points: curve, color: css("--accent"), area: true }, { points: [curve[n - 1]], color: css("--pick") }],
        xLabel: `${n} path${n > 1 ? "s" : ""} → ${curve[n - 1][1].toFixed(0)}% correct`,
      });
    }
    function run() {
      const n = +sl.value;
      U.$("#sc-n-out").textContent = n;
      const r = Math.random;
      const drawn = [...Array(n)].map(() => { const p = drawPath(r()); return { ...p, text: p.t[Math.floor(r() * p.t.length)] }; });
      U.$("#sc-paths").innerHTML = drawn.map((d, i) => `<div class="path ${d.ok ? "ok" : "no"}"><span class="num muted">${i + 1}</span><span>${U.esc(d.text)}</span><span class="ans">${d.a}</span></div>`).join("");
      const counts = new Map();
      drawn.forEach((d) => counts.set(d.a, (counts.get(d.a) || 0) + 1));
      const max = Math.max(...counts.values());
      const winners = [...counts].filter(([, c]) => c === max);
      U.$("#sc-vote").innerHTML = `<b style="font-family:var(--font-body);font-size:13px">Majority vote:</b>` + [...counts].sort((a, b) => b[1] - a[1]).map(([a, c]) => `<span class="${c === max ? "win" : ""}">${a} × ${c}</span>`).join("") + (winners.length > 1 ? `<span>tie: pick at random</span>` : "");
      drawCurve();
    }
    sl.addEventListener("input", () => { U.$("#sc-n-out").textContent = sl.value; drawCurve(); });
    U.$("#sc-run").onclick = run;
    L.onTheme.push(drawCurve);
    run();
    grpo();
  };

  // ================= 7b GRPO advantages =================
  function grpo() {
    const answers = [
      ["23 × 7 = 140 + 21 = 161", true], ["20×7=140, 3×7=21, 161", true], ["23 × 7 = 151", false], ["7 × 23: 161", true],
      ["23 × 7 = 168", false], ["161", true], ["23+7 = 30, so 30", false], ["(25 − 2) × 7 = 175 − 14 = 161", true],
    ];
    const el = U.$("#grpo");
    function render() {
      const r = answers.map(([, ok]) => (ok ? 1 : 0));
      const mean = M.mean(r), sd = M.std(r);
      const adv = r.map((x) => (sd > 0 ? (x - mean) / sd : 0));
      el.innerHTML = answers.map(([t, ok], i) => {
        const a = adv[i], w = Math.min(50, Math.abs(a) * 25);
        return `<button class="g-ans ${ok ? "ok" : "no"}" data-i="${i}" title="Click to flip right/wrong"><span>${U.esc(t)}</span><span class="r">reward ${r[i]}</span><span class="adv num">${a >= 0 ? "+" : ""}${a.toFixed(2)}</span><span class="advbar"><i style="${a >= 0 ? `left:50%;width:${w}%;background:var(--good)` : `left:${50 - w}%;width:${w}%;background:var(--bad)`}"></i></span></button>`;
      }).join("");
      U.$("#grpo-stats").innerHTML = `Problem: 23 × 7 · group mean reward ${mean.toFixed(2)} · spread (std) ${sd.toFixed(2)}` + (sd === 0 ? ` · <b>every answer scored the same, so there is no learning signal from this problem</b>` : "");
    }
    el.addEventListener("click", (e) => {
      const b = e.target.closest(".g-ans");
      if (!b) return;
      answers[+b.dataset.i][1] = !answers[+b.dataset.i][1];
      render();
    });
    render();
  }
})(window.LLM);
