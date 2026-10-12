const test = require("node:test");
const assert = require("node:assert/strict");
const load = require("./load");

const L = load("corpus.js", "core.js");
const ngram = new L.NGram(L.CORPUS);
const dec = new L.Decoder(ngram);
const ctx = ["the", "cat", "sat", "on", "the"];
const finals = (r) => r.candidates.map((c) => c.final);
const sum = (xs) => xs.reduce((a, b) => a + b, 0);

// A model with hand-set scores, so the expected numbers are known exactly.
const fixed = (logits) => new L.Decoder({ vocab: ["a", "b", "c", "d"], distribution: () => ({ logits, used: "fixed" }) });

test("candidates cover the whole vocabulary, most likely first, and sum to 1", () => {
  const r = dec.next(ctx);
  assert.equal(r.candidates.length, ngram.vocab.length);
  assert.ok(Math.abs(sum(finals(r)) - 1) < 1e-9);
  for (let i = 1; i < r.candidates.length; i++) assert.ok(r.candidates[i - 1].base >= r.candidates[i].base);
});

test("no rng means a preview: candidates but no pick", () => {
  assert.equal(dec.next(ctx).pick, null);
});

test("the pick follows the dice roll", () => {
  const d = fixed([Math.log(0.5), Math.log(0.3), Math.log(0.15), Math.log(0.05)]);
  assert.equal(d.next([], {}, () => 0.1).pick.token, "a");
  assert.equal(d.next([], {}, () => 0.6).pick.token, "b");
  assert.equal(d.next([], {}, () => 0.9).pick.token, "c");
  const p = d.next([], {}, () => 0.97).pick;
  assert.equal(p.token, "d");
  assert.equal(p.roll, 0.97);
  assert.ok(Math.abs(p.p - 0.05) < 1e-9);
});

test("temperature 0 is greedy: all probability on the top token, whatever the roll", () => {
  const r = dec.next(ctx, { temperature: 0 }, () => 0.999);
  assert.equal(r.candidates[0].final, 1);
  assert.equal(r.candidates.filter((c) => c.kept).length, 1);
  assert.equal(r.pick.token, r.candidates[0].token);
});

test("greedy breaks ties by vocabulary order", () => {
  assert.equal(fixed([0, 1, 1, 0]).next([], { temperature: 0 }, () => 0.5).pick.token, "b");
});

test("temperature below 1 sharpens and above 1 flattens", () => {
  const cold = dec.next(ctx, { temperature: 0.5 }).candidates[0].final;
  const warm = dec.next(ctx, { temperature: 1 }).candidates[0].final;
  const hot = dec.next(ctx, { temperature: 2 }).candidates[0].final;
  assert.ok(cold > warm && warm > hot);
});

test("top-k keeps exactly k tokens and renormalizes", () => {
  const r = dec.next(ctx, { topK: 3 });
  assert.equal(r.candidates.filter((c) => c.kept).length, 3);
  assert.ok(Math.abs(sum(finals(r)) - 1) < 1e-9);
});

test("top-p keeps the smallest set reaching p", () => {
  const d = fixed([Math.log(0.5), Math.log(0.3), Math.log(0.15), Math.log(0.05)]);
  const kept = (p) => d.next([], { topP: p }).candidates.filter((c) => c.kept).map((c) => c.token);
  assert.deepEqual([...kept(0.5)], ["a"]);
  assert.deepEqual([...kept(0.7)], ["a", "b"]);
  assert.deepEqual([...kept(0.9)], ["a", "b", "c"]);
});

test("base probabilities ignore the settings", () => {
  const plain = dec.next(ctx).candidates.map((c) => c.base);
  const tuned = dec.next(ctx, { temperature: 0.3, topK: 2 }).candidates.map((c) => c.base);
  assert.deepEqual(tuned, plain);
});

test("detokenize attaches punctuation to the previous word", () => {
  assert.equal(dec.detokenize(["the", "cat", "sat", ".", "the", "end", "!"]), "the cat sat. the end!");
});

test("reports which context the model matched", () => {
  assert.match(dec.next(["sat", "on"]).used, /trigram "sat on/);
});
