// Boot: render reference content, start every section, keep the table of contents in sync.
window.LLM = window.LLM || {};

(function (L) {
  const U = L.ui;

  function timeline() {
    let thread = "all";
    const list = U.$("#timeline");
    function render() {
      list.innerHTML = L.TIMELINE.filter((e) => thread === "all" || e.t === thread).map((e) => `<li class="tl"><span class="yr">${e.y}</span><span class="who">${U.esc(e.who)}<br><span class="tag">${L.THREADS[e.t]}</span></span><div><h4>${U.esc(e.h)}</h4><p>${U.esc(e.p)}${e.a ? ` <a href="${e.a}" target="_blank" rel="noopener">Source ↗</a>` : ""}</p></div></li>`).join("");
    }
    U.seg(U.$("#tl-filter"), Object.entries(L.THREADS), thread, (k) => { thread = k; render(); });
    render();
  }
  function frontier() {
    U.$("#frontier-cards").innerHTML = L.FRONTIER.map((c) => `<article class="card"><h4>${U.esc(c.h)}</h4><p>${U.esc(c.p)}</p><p class="open"><b>Open question</b>${U.esc(c.q)}</p><div class="links">${c.l.map(([t, u]) => `<a href="${u}" target="_blank" rel="noopener">${U.esc(t)} ↗</a>`).join("")}</div></article>`).join("");
    U.$("#learn-list").innerHTML = L.LEARN.map(([k, t, d, u]) => `<a href="${u}" target="_blank" rel="noopener"><span>${k}</span><div><strong>${U.esc(t)}</strong><small>${U.esc(d)}</small></div></a>`).join("");
  }
  function scrollspy() {
    const links = [...document.querySelectorAll(".toc a")];
    const map = new Map(links.map((a) => [a.getAttribute("href").slice(1), a]));
    if (!("IntersectionObserver" in window)) return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((a) => a.classList.remove("active"));
        const a = map.get(e.target.id);
        if (a) {
          a.classList.add("active");
          if (getComputedStyle(U.$("#side")).position === "sticky" && window.innerWidth <= 980) a.scrollIntoView({ block: "nearest", inline: "center" });
        }
      });
    }, { rootMargin: "-30% 0px -60% 0px" });
    document.querySelectorAll("section.sec").forEach((s) => io.observe(s));
  }
  function themeWatch() {
    const fire = () => L.onTheme.forEach((f) => { try { f(); } catch (e) { /* keep other redraws going */ } });
    try { matchMedia("(prefers-color-scheme: dark)").addEventListener("change", fire); } catch (e) {}
    new MutationObserver(fire).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  }

  function start() {
    const S = L.sections;
    ["loop", "tokenize", "embed", "attend", "predict", "train", "align", "reason", "build"].forEach((k) => {
      try { S[k](); } catch (e) { console.error("section " + k + " failed", e); }
    });
    timeline();
    frontier();
    scrollspy();
    themeWatch();
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})(window.LLM);
