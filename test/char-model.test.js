const test = require("node:test");
const assert = require("node:assert/strict");
const load = require("./load");

const L = load("corpus.js", "core.js", "learn.js");
const fresh = () => L.CharModel.fromText(L.CORPUS);

test("vocabulary is lowercase letters, space and full stop", () => {
  const m = L.CharModel.fromText("The Cat, sat! On 2 mats.");
  assert.deepEqual([...m.chars], [" ", ".", "a", "c", "e", "h", "m", "n", "o", "s", "t"]);
});

test("an untrained model scores close to random guessing", () => {
  const m = fresh();
  assert.ok(Math.abs(m.loss - m.uniformLoss) < 0.05, `loss ${m.loss} vs uniform ${m.uniformLoss}`);
  assert.ok(m.bestLoss < m.uniformLoss);
  assert.equal(m.diverged, false);
});

test("training at a sensible learning rate approaches the best possible loss", () => {
  const m = fresh().step(0.5, 300);
  assert.equal(m.steps, 300);
  assert.ok(m.loss - m.bestLoss < 0.1, `loss ${m.loss} vs best ${m.bestLoss}`);
});

test("loss never goes below the best possible loss", () => {
  const m = fresh().step(0.5, 300);
  assert.ok(m.loss >= m.bestLoss - 1e-9);
});

test("a learning rate that is too high diverges", () => {
  const m = fresh().step(12, 50);
  assert.equal(m.diverged, true);
});

test("each row of predictions is a probability distribution", () => {
  const m = fresh().step(0.5, 20);
  for (const row of m.probs) {
    const sum = row.reduce((a, b) => a + b, 0);
    assert.ok(Math.abs(sum - 1) < 1e-9);
  }
});

test("the same seed gives the same starting model, however much it was sampled", () => {
  const a = fresh();
  a.sample(L.math.rng(1), 500);
  assert.equal(fresh().loss, a.loss);
});

test("sampling is repeatable with a seeded generator and uses only the vocabulary", () => {
  const m = fresh().step(0.5, 100);
  const s1 = m.sample(L.math.rng(3), 80), s2 = m.sample(L.math.rng(3), 80);
  assert.equal(s1, s2);
  assert.equal(s1.length, 81);
  assert.ok([...s1].every((c) => m.chars.includes(c)));
});
