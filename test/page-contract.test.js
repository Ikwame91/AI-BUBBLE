// The stations find their elements by id, and app.js starts them by name. Nothing else
// ties index.html to the scripts, so these tests do: rename an id in one place and they fail.
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");

const root = path.join(__dirname, "..");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const jsFiles = fs.readdirSync(path.join(root, "js")).filter((f) => f.endsWith(".js"));
const js = Object.fromEntries(jsFiles.map((f) => [f, fs.readFileSync(path.join(root, "js", f), "utf8")]));
const pageIds = new Set([...html.matchAll(/\sid="([\w-]+)"/g)].map((m) => m[1]));

test("every element id a script looks up exists in index.html", () => {
  const missing = [];
  for (const [file, src] of Object.entries(js)) {
    for (const m of src.matchAll(/(?:\$|querySelector)\(\s*["']#([\w-]+)["']/g)) {
      if (!pageIds.has(m[1])) missing.push(`${file}: #${m[1]}`);
    }
  }
  assert.deepEqual(missing, []);
});

test("every id in index.html is unique", () => {
  const all = [...html.matchAll(/\sid="([\w-]+)"/g)].map((m) => m[1]);
  assert.deepEqual(all.filter((id, i) => all.indexOf(id) !== i), []);
});

test("every station app.js starts is defined by a sections file", () => {
  const started = JSON.parse(js["app.js"].match(/\[("[a-z]+"(?:,\s*"[a-z]+")*)\]\.forEach/)[0].replace(/\.forEach$/, ""));
  const defined = new Set(Object.values(js).flatMap((src) => [...src.matchAll(/\bS\.(\w+) = function/g)].map((m) => m[1])));
  assert.deepEqual(started.filter((k) => !defined.has(k)), []);
});

test("every script index.html loads exists, and the table of contents points at real sections", () => {
  for (const m of html.matchAll(/<script src="(js\/[\w-]+\.js)">/g)) assert.ok(fs.existsSync(path.join(root, m[1])), m[1]);
  for (const m of html.matchAll(/<a href="#([\w-]+)"><b>/g)) assert.ok(pageIds.has(m[1]), `#${m[1]}`);
});
