// Loads the page's plain browser scripts into a sandbox and returns window.LLM.
// The scripts are written for <script> tags (see docs/adr/0001), so we fake `window`.
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

module.exports = function load(...files) {
  const sandbox = { console, Math };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  for (const f of files) {
    const file = path.join(__dirname, "..", "js", f);
    vm.runInContext(fs.readFileSync(file, "utf8"), sandbox, { filename: file });
  }
  return sandbox.LLM;
};
