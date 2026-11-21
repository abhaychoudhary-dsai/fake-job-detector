const form = document.getElementById("form");
const result = document.getElementById("result");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = x => (x * 100).toFixed(x < 0.01 ? 1 : 0) + "%";
let examples = [];

fetch("/api/examples").then(r => r.json()).then(d => { examples = d; });

document.querySelectorAll("[data-ex]").forEach(btn => btn.addEventListener("click", () => {
  const wantFake = btn.dataset.ex === "fake";
  const pool = examples.filter(e => Boolean(e.fraudulent) === wantFake);
  if (!pool.length) return;
  const ex = pool[Math.floor(Math.random() * pool.length)];
  for (const el of form.elements) {
    if (!el.name) continue;
    if (el.type === "checkbox") el.checked = Boolean(Number(ex[el.name]));
    else el.value = ex[el.name] ?? "";
  }
  form.scrollIntoView({ behavior: "smooth", block: "start" });
}));

form.addEventListener("submit", async e => {
  e.preventDefault();
  const body = {};
  for (const el of form.elements) {
    if (!el.name) continue;
    body[el.name] = el.type === "checkbox" ? el.checked : el.value;
  }
  const btn = document.getElementById("go");
  btn.disabled = true; btn.textContent = "Checking…";
  try {
    const res = await fetch("/api/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    const data = await res.json();
    if (!res.ok) throw new Error(Array.isArray(data.detail) ? data.detail.map(d => d.msg).join(", ") : data.detail);
    show(data);
  } catch (err) {
    result.innerHTML = `<p class="error">${esc(err.message)}</p>`;
  }
  btn.disabled = false; btn.textContent = "Check this posting";
});

function show(r) {
  const cls = r.verdict === "Likely fake" ? "v-fake" : r.verdict === "Be careful" ? "v-warn" : "v-real";
  const chips = (list, c) => list.length ? list.map(w => `<span class="chip ${c}">${esc(w.term)}</span>`).join("") : `<span class="none">Nothing stands out</span>`;
  const signs = r.warning_signs.length
    ? r.warning_signs.map(s => `<div class="sign"><b>${esc(s.signal)}</b><span>${pct(s.fake_rate_with)} of such ads were fake vs ${pct(s.fake_rate_without)} otherwise</span></div>`).join("")
    : `<span class="none">No common warning signs</span>`;
  result.innerHTML = `
    <span class="verdict ${cls}">${esc(r.verdict)}</span>
    <div class="big">${pct(r.fake_probability)} <small>chance of being fake</small></div>
    <div class="meter"><div class="cut" style="left:${r.threshold * 100}%"></div><div class="pin" style="left:calc(${r.fake_probability * 100}% - 2px)"></div></div>
    <div class="scale"><span>Real</span><span>cut-off ${pct(r.threshold)}</span><span>Fake</span></div>
    <h3>Words that look suspicious</h3><div class="chips">${chips(r.words_pointing_fake, "f")}</div>
    <h3>Words that look normal</h3><div class="chips">${chips(r.words_pointing_real, "r")}</div>
    <h3>Warning signs</h3>${signs}`;
}

fetch("/api/model").then(r => r.json()).then(m => {
  const rows = Object.entries(m.test_results);
  document.getElementById("scores").innerHTML =
    `<tr><th>Model (test set, ${Math.round(m.rows * 0.2).toLocaleString()} postings)</th><th class="n">Precision</th><th class="n">Recall</th><th class="n">F1</th><th class="n">PR-AUC</th></tr>` +
    rows.map(([name, s]) => `<tr class="${name === m.chosen_model ? "best" : ""}"><td>${esc(name)}${name === m.chosen_model ? " · used in this app" : ""}</td>
      <td class="n">${s.precision.toFixed(3)}</td><td class="n">${s.recall.toFixed(3)}</td><td class="n">${s.f1.toFixed(3)}</td><td class="n">${s.pr_auc.toFixed(3)}</td></tr>`).join("");
});
