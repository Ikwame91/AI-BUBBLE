# Plain scripts, no build step

The page ships as classic `<script>` tags sharing one `window.LLM` namespace, with no bundler and no ES modules, so `index.html` works when opened straight from disk (`file://`), where browsers block module imports. The cost is that script order in `index.html` is part of the contract between files; revisit only if the page gains a dev server or a build step for another reason.
