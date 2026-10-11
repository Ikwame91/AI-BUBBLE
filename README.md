# Token by Token

An interactive field guide to how large language models work. Everything runs in the browser in plain JavaScript, with no AI service, so you can see each number being computed.

Open `index.html` in a browser (double-click it; no server or build step needed).

## What's inside

**Hero: the generation loop.** Press *Next token* and watch tokenize → read context → logits → softmax → sample → append, the loop every LLM runs.

| Station | What you do | What is real vs. illustrative |
|---|---|---|
| 1 Tokenize | Replay byte-pair-encoding merges with a slider | Real BPE, trained on the page's corpus |
| 2 Embed | Explore a word map, do `king − man + woman` | Real vector math; the 10 named dimensions are hand-built |
| 3 Attend | Inspect query·key scores, causal mask, softmax for 3 heads | Real attention math; head weights are hand-designed |
| 4 Predict & sample | Change temperature, top-k, top-p and sample | Real trigram model trained on the corpus |
| 5 Pretrain | Train a character model live and watch the loss fall | Real gradient descent on cross-entropy |
| 6 Align | Compare base → SFT → RLHF → reasoning; label preference pairs | Stage outputs are scripted; the reward model is real logistic regression on your labels |
| 7 Reason | Self-consistency voting; GRPO advantages | Real Monte Carlo simulation and GRPO arithmetic on scripted answers |
| 8 Build | RAG retrieval, the assembled prompt, the tool-use loop | Real TF-IDF retrieval; API messages shown are examples |

Plus a timeline (1950 → 2025), a research-frontier map with primary sources, and a reading list.

## Talk to a real model

`examples/chat.py` is a complete terminal chatbot and `examples/agent.py` is the tool-use loop written out by hand. Both use the Anthropic Python SDK:

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
python examples/chat.py
python examples/agent.py "Do I need an umbrella in Lisbon tomorrow?"
```

Both opt in to server-side refusal fallbacks (`fallbacks="default"`), so a request a safety classifier declines is retried on a fallback model, and both check `stop_reason == "refusal"`.

## Files

```
index.html            page structure and explanations
css/style.css         design tokens (light + dark) and layout
js/corpus.js          training text and the RAG knowledge base
js/core.js            math, BPE, trigram model, embeddings, attention, retrieval
js/ui.js              DOM helpers, bar and line charts
js/sections-*.js      each station's interactive logic
js/content.js         timeline, frontier and reading-list data
js/app.js             startup and navigation
examples/             runnable Python chatbot and agent
```
