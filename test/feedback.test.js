const test = require("node:test");
const assert = require("node:assert/strict");
const load = require("./load");

const L = load("corpus.js", "core.js", "learn.js");
const F = L.Feedback;
const rng = (seed) => L.math.rng(seed);

// Traits: [correct, detailed, friendly, long, flatters you]
const honest = [1, 0.6, 0.3, 0.2, 0];
const flattering = [0, 0.4, 0.9, 0.8, 1];

test("with no comparisons the reward model scores everything 0", () => {
  const rm = F.fitRewardModel([]);
  assert.equal(rm.score(honest), 0);
});

test("preferring correct answers teaches the model to reward correctness", () => {
  const rm = F.fitRewardModel([{ win: honest, lose: flattering }, { win: [1, 0, 0, 0, 0], lose: [0, 0, 0, 0, 0] }]);
  assert.ok(rm.weights[0] > 0);
  assert.ok(rm.score(honest) > rm.score(flattering));
});

test("always preferring flattery is learned too: reward hacking", () => {
  const rm = F.fitRewardModel([{ win: flattering, lose: honest }, { win: flattering, lose: honest }]);
  const biggest = rm.weights.indexOf(Math.max(...rm.weights));
  assert.equal(biggest, 4, `weights ${rm.weights}`);
  assert.ok(rm.weights[0] < 0, "correctness is now punished");
});

test("weights stay finite even for a perfectly separable comparison", () => {
  const rm = F.fitRewardModel([{ win: [1], lose: [0] }]);
  assert.ok(Number.isFinite(rm.weights[0]) && rm.weights[0] < 20);
});

test("majority vote picks the most common answer", () => {
  const v = F.majorityVote(["a", "b", "a", "c", "a"], rng(1));
  assert.equal(v.winner, "a");
  assert.deepEqual([...v.tied], ["a"]);
  assert.equal(v.counts.get("a"), 3);
});

test("ties are broken by the generator, among the tied answers only", () => {
  assert.equal(F.majorityVote(["x", "y", "z", "y", "x"], () => 0).winner, "x");
  assert.equal(F.majorityVote(["x", "y", "z", "y", "x"], () => 0.99).winner, "y");
});

const odds = [{ answer: "$0.05", p: 0.55, correct: true }, { answer: "$0.10", p: 0.35, correct: false }, { answer: "$0.55", p: 0.1, correct: false }];

test("drawAnswer follows the roll", () => {
  assert.equal(F.drawAnswer(odds, 0.1).answer, "$0.05");
  assert.equal(F.drawAnswer(odds, 0.7).answer, "$0.10");
  assert.equal(F.drawAnswer(odds, 0.95).answer, "$0.55");
});

test("one reasoning path is right as often as a single answer is", () => {
  const acc = F.votingAccuracy(odds, 1, 20000, rng(5));
  assert.ok(Math.abs(acc - 0.55) < 0.02, `accuracy ${acc}`);
});

test("more reasoning paths raise accuracy (self-consistency)", () => {
  const one = F.votingAccuracy(odds, 1, 5000, rng(6));
  const nine = F.votingAccuracy(odds, 9, 5000, rng(6));
  const twentyFive = F.votingAccuracy(odds, 25, 5000, rng(6));
  assert.ok(one < nine && nine < twentyFive, `${one} ${nine} ${twentyFive}`);
  assert.ok(twentyFive > 0.8);
});

test("voting cannot help when the right answer is not the most likely one", () => {
  const bad = [{ answer: "right", p: 0.3, correct: true }, { answer: "wrong", p: 0.7, correct: false }];
  assert.ok(F.votingAccuracy(bad, 25, 3000, rng(7)) < 0.1);
});

test("GRPO: correct answers get positive advantage, wrong ones negative", () => {
  const { advantages, mean } = F.groupAdvantages([1, 1, 0, 1, 0, 1, 0, 1]);
  assert.equal(mean, 0.625);
  advantages.forEach((a, i) => assert.ok([1, 1, 0, 1, 0, 1, 0, 1][i] ? a > 0 : a < 0));
  assert.ok(Math.abs(advantages.reduce((x, y) => x + y, 0)) < 1e-9, "advantages sum to zero");
});

test("GRPO: identical rewards give no learning signal", () => {
  for (const r of [[1, 1, 1, 1], [0, 0, 0, 0]]) {
    const g = F.groupAdvantages(r);
    assert.equal(g.spread, 0);
    assert.deepEqual([...g.advantages], [0, 0, 0, 0]);
  }
});
