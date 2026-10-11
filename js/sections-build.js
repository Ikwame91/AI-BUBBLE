// Part 3: building with LLMs — retrieval, the assembled prompt, and the tool-use loop.
window.LLM = window.LLM || {};

(function (L) {
  const U = L.ui;
  const S = (L.sections = L.sections || {});

  S.build = function () {
    const retriever = new L.Retriever(L.DOCS);
    const q = U.$("#rag-q");
    const SYSTEM = "You are a helpful assistant. Answer using only the sources below and cite them like [1]. If the sources do not contain the answer, say so.";
    const HISTORY = [["user", "What is a token?"], ["assistant", "A token is a chunk of text, often a word or part of one, that the model reads as a single unit [1]."]];
    const WINDOW = 2048;

    function render() {
      const res = retriever.search(q.value);
      const picked = res.filter((r) => r.score > 0).slice(0, 3);
      U.$("#rag-results").innerHTML = res.slice(0, 5).map((r, i) => {
        const inPrompt = picked.includes(r);
        let text = U.esc(r.doc.text);
        r.hits.forEach((h) => (text = text.replace(new RegExp(`\\b(${h}[a-z]*)`, "gi"), "<mark>$1</mark>")));
        return `<div class="rag-doc${inPrompt ? " in" : ""}"><header><span>${inPrompt ? `[${picked.indexOf(r) + 1}] ` : ""}${U.esc(r.doc.title)}</span><span class="num">similarity ${r.score.toFixed(3)}${inPrompt ? " · sent" : ""}</span></header><p>${text}</p></div>`;
      }).join("") + `<p class="hint">Ranking uses TF-IDF word overlap across ${L.DOCS.length} documents. Production systems use embedding vectors (Step 2) and a reranker, but the job is the same: pick the few passages worth pasting into the prompt.</p>`;

      const sources = picked.map((r, i) => `<source id="${i + 1}" title="${r.doc.title}">\n${r.doc.text}\n</source>`).join("\n");
      const hist = HISTORY.map(([r, t]) => `${r}: ${t}`).join("\n");
      const parts = [
        ["system", SYSTEM, "var(--accent)"],
        ["sources", sources || "(no matching sources)", "var(--pick)"],
        ["history", hist, "var(--muted)"],
        ["question", q.value, "var(--good)"],
      ];
      const toks = parts.map(([, t]) => L.estimateTokens(t));
      const used = toks.reduce((a, b) => a + b, 0);
      U.$("#ctx-bar").innerHTML = parts.map(([, , c], i) => `<i style="width:${(toks[i] / WINDOW) * 100}%;background:${c}"></i>`).join("");
      U.$("#ctx-bar").nextElementSibling?.classList.contains("ctx-legend") && U.$("#ctx-bar").nextElementSibling.remove();
      U.$("#ctx-bar").insertAdjacentHTML("afterend", `<div class="ctx-legend">${parts.map(([n, , c], i) => `<span><i style="background:${c}"></i>${n} ${toks[i]}</span>`).join("")}<span>free for the answer: ${WINDOW - used} of a ${WINDOW.toLocaleString()}-token window</span></div>`);
      U.$("#rag-prompt").innerHTML =
        `<span class="k">system:</span> ${U.esc(SYSTEM)}\n\n` +
        (sources ? `<span class="p">${U.esc(sources)}</span>\n\n` : `<span class="c">(no sources matched)</span>\n\n`) +
        `<span class="c"># earlier turns are re-sent every time</span>\n${HISTORY.map(([r, t]) => `<span class="k">${r}:</span> ${U.esc(t)}`).join("\n")}\n` +
        `<span class="k">user:</span> ${U.esc(q.value)}\n<span class="k">assistant:</span> <span class="c">▌ the model starts generating here, one token at a time</span>`;
    }
    q.addEventListener("input", render);
    render();
    agent();
    U.copy(U.$("#copy-chat"), () => U.$("#chat-code").textContent);
  };

  // ---------- agent loop walkthrough ----------
  const STEPS = [
    ["The app sends the question and a tool list", "the model sees tool names, descriptions, and JSON schemas",
`POST /v1/messages
{
  "model": "claude-opus-5-5",
  "tools": [{
    "name": "get_forecast",
    "description": "Weather forecast for a city",
    "input_schema": {"type": "object",
      "properties": {"city": {"type": "string"},
                     "day": {"enum": ["today", "tomorrow"]}}}
  }],
  "messages": [
    {"role": "user",
     "content": "Do I need an umbrella in Lisbon tomorrow?"}
  ]
}`],
    ["The model asks for a tool instead of answering", "it is still next-token prediction: it was trained to write this JSON shape",
`{
  "role": "assistant",
  "stop_reason": "tool_use",
  "content": [
    {"type": "text", "text": "Let me check the forecast."},
    {"type": "tool_use", "id": "toolu_01A",
     "name": "get_forecast",
     "input": {"city": "Lisbon", "day": "tomorrow"}}
  ]
}`],
    ["Your code runs the tool", "the model never runs anything itself; your program does",
`# in your app, not in the model
result = get_forecast(city="Lisbon", day="tomorrow")
# → {"rain_chance": 0.7, "high_c": 17}`],
    ["The app sends everything back, plus the result", "the full history again, since the model has no memory",
`"messages": [
  {"role": "user", "content": "Do I need an umbrella..."},
  {"role": "assistant", "content": [ ...tool_use toolu_01A... ]},
  {"role": "user", "content": [
    {"type": "tool_result", "tool_use_id": "toolu_01A",
     "content": "{\\"rain_chance\\": 0.7, \\"high_c\\": 17}"}
  ]}
]`],
    ["The model writes the final answer", "stop_reason is end_turn, so the loop stops",
`{
  "role": "assistant",
  "stop_reason": "end_turn",
  "content": [{"type": "text", "text":
    "Yes, take one. There's a 70% chance of rain in Lisbon tomorrow, with a high of 17 °C."}]
}`],
    ["An agent is this loop, repeated", "coding agents, research agents and computer-use agents all work this way",
`while response.stop_reason == "tool_use":
    run every tool the model asked for
    append the results to messages
    response = call the model again

# Claude Code: tools = read_file, edit_file, run_tests, ...
# Perplexity-style search: tools = web_search, fetch_page
# MCP (Model Context Protocol) is a standard way to plug
# tools and data sources into any model app.`],
  ];
  function agent() {
    let i = 0;
    const list = U.$("#agent-steps");
    list.innerHTML = STEPS.map(([t, s], k) => `<li data-k="${k}"><div>${U.esc(t)}<small>${U.esc(s)}</small></div></li>`).join("");
    function show() {
      list.querySelectorAll("li").forEach((li, k) => li.classList.toggle("on", k === i));
      U.$("#agent-msg").textContent = STEPS[i][2];
      U.$("#agent-prev").disabled = i === 0;
      U.$("#agent-next").textContent = i === STEPS.length - 1 ? "Start over" : "Next step";
    }
    list.onclick = (e) => { const li = e.target.closest("li"); if (li) { i = +li.dataset.k; show(); } };
    U.$("#agent-next").onclick = () => { i = (i + 1) % STEPS.length; show(); };
    U.$("#agent-prev").onclick = () => { i = Math.max(0, i - 1); show(); };
    show();
  }
})(window.LLM);
