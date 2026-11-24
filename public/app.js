const $ = (s, el = document) => el.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const ICON = {
  ok: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="var(--good)" stroke-width="2.4"><path d="M5 12l5 5L20 7"/></svg>',
  bad: '<svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="var(--bad)" stroke-width="2.4"><path d="M6 6l12 12M18 6L6 18"/></svg>',
  shield: '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z"/></svg>',
  stop: '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4"><circle cx="12" cy="12" r="9"/><path d="M8 12h8"/></svg>',
  warn: '<svg viewBox="0 0 24 24" width="12" height="12" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M12 3L2 21h20zM12 10v5M12 18v.5"/></svg>',
};
const SUGG = [
  ['Error code', 'My desktop app shows KB-412. What do I do?'],
  ['Table lookup', 'How much does the Business plan cost per user on monthly billing?'],
  ['False premise', 'How do I turn on SCIM provisioning on our Business plan?'],
  ['Follow-up', ['How much is the Team plan per user on annual billing?', 'And Business?']],
  ['Updated fact', 'What’s the API v2 rate limit on the Business plan?'],
  ['Not in the docs', 'Who is the CEO of Kitebox?'],
  ['Unknown code', 'What does error KB-777 mean?'],
  ['Policy', 'Can I get a refund if I cancel my monthly plan?'],
];
const NOTES = {
  fixed: 'Hybrid retrieval, relevance gate, cite-or-refuse prompt, grounding checks.',
  naive: 'A typical first version: 1,000-char chunks, dense top-4, permissive prompt, no checks.',
  both: 'Each question goes through both pipelines. Naive on the left, fixed on the right.',
};

let pipe = 'fixed', chat = [], busy = false, results = [], meta = null;

/* ---------- tabs & theme ---------- */
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => showTab(b.dataset.tab)));
function showTab(name) {
  document.querySelectorAll('.tabs button').forEach((b) => b.setAttribute('aria-selected', b.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('on', t.id === 'tab-' + name));
  if (name === 'eval') loadEval();
  if (name === 'kb') loadKb();
  try { window.history.replaceState(null, '', name === 'chat' ? location.pathname : '#' + name); } catch (e) {}
}
$('#theme').addEventListener('click', () => {
  const dark = document.documentElement.dataset.theme
    ? document.documentElement.dataset.theme === 'dark'
    : matchMedia('(prefers-color-scheme: dark)').matches;
  document.documentElement.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('theme', document.documentElement.dataset.theme); } catch (e) {}
});

/* ---------- pipeline switch ---------- */
document.querySelectorAll('.seg button').forEach((b) => b.addEventListener('click', () => {
  pipe = b.dataset.pipe;
  document.querySelectorAll('.seg button').forEach((x) => { x.classList.toggle('on', x === b); x.setAttribute('aria-checked', x === b); });
  $('#pipe-note').textContent = NOTES[pipe];
}));
$('#pipe-note').textContent = NOTES.fixed;

/* ---------- suggestions ---------- */
$('#sugg').innerHTML = SUGG.map(([k, q], i) => `<button data-i="${i}"><small>${esc(k)}</small><span>${esc(Array.isArray(q) ? q.join(' → ') : q)}</span></button>`).join('');
$('#sugg').addEventListener('click', async (e) => {
  const b = e.target.closest('button'); if (!b || busy) return;
  const q = SUGG[+b.dataset.i][1];
  for (const one of Array.isArray(q) ? q : [q]) await ask(one);
});
$('#clear').addEventListener('click', () => location.reload());

/* ---------- chat ---------- */
$('#ask').addEventListener('submit', (e) => { e.preventDefault(); const v = $('#q').value.trim(); if (v && !busy) { $('#q').value = ''; ask(v); } });

async function call(message, pipeline, hist) {
  const r = await fetch('/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message, history: hist, pipeline }) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || j.detail || `HTTP ${r.status}`);
  if (j.error) throw new Error(j.error);
  return j;
}

async function ask(message) {
  busy = true; $('#send').disabled = true;
  $('.welcome')?.remove();
  const turn = document.createElement('div'); turn.className = 'turn';
  const pipes = pipe === 'both' ? ['naive', 'fixed'] : [pipe];
  turn.innerHTML = `<div class="msg-user">${esc(message)}</div><div class="answers ${pipes.length > 1 ? 'two' : ''}">${pipes.map((p) =>
    `<div class="bubble ${p}" data-p="${p}"><div class="who"><i></i>${p === 'fixed' ? 'Fixed pipeline' : 'Naive pipeline'}</div><div class="txt"><span class="typing"><i></i><i></i><i></i></span></div></div>`).join('')}</div>`;
  $('#log').appendChild(turn); turn.scrollIntoView({ block: 'end', behavior: 'smooth' });
  const hist = chat.slice(-6);
  const outs = await Promise.all(pipes.map((p) => call(message, p, hist).catch((e) => ({ error: e.message, pipeline: p }))));
  outs.forEach((r, i) => {
    const el = turn.querySelectorAll('.bubble')[i];
    if (r.error) { el.querySelector('.txt').innerHTML = `<span class="err">${esc(r.error)}</span>`; return; }
    const id = results.push(r) - 1; el.dataset.id = id;
    el.querySelector('.txt').innerHTML = renderAnswer(r.answer);
    el.insertAdjacentHTML('beforeend', badges(r, message));
    el.addEventListener('click', (ev) => {
      const c = ev.target.closest('.cite');
      select(id, el, c ? +c.dataset.n : null);
    });
  });
  const main = outs.find((r) => r.pipeline === 'fixed' && !r.error) || outs.find((r) => !r.error);
  chat.push({ role: 'user', content: message });
  if (main) chat.push({ role: 'assistant', content: main.answer.slice(0, 1500) });
  const lastEl = turn.querySelector(`.bubble[data-p="${main?.pipeline || pipes[0]}"]`);
  if (main && lastEl) select(+lastEl.dataset.id, lastEl);
  turn.scrollIntoView({ block: 'end', behavior: 'smooth' });
  busy = false; $('#send').disabled = false; $('#q').focus();
}

function renderAnswer(text) {
  let h = esc(text.trim())
    .replace(/\*\*(.+?)\*\*/g, '<b>$1</b>')
    .replace(/\[(\d+(?:\s*,\s*\d+)*)\]/g, (_, ns) => ns.split(/\s*,\s*/).map((n) => `<button class="cite" data-n="${n}" title="Show excerpt ${n}">${n}</button>`).join(''));
  h = h.split(/\n{2,}/).map((para) => {
    const lines = para.split('\n');
    if (lines.every((l) => /^\s*([-*•]|\d+\.)\s+/.test(l))) return '<ul style="margin:0 0 6px;padding-left:18px">' + lines.map((l) => '<li>' + l.replace(/^\s*([-*•]|\d+\.)\s+/, '') + '</li>').join('') + '</ul>';
    return '<p>' + lines.join('<br>') + '</p>';
  }).join('');
  return h;
}

function badges(r, message) {
  const t = [];
  if (r.pipeline === 'fixed') {
    if (r.gate && r.gate.passed === false) t.push(`<span class="tag warn">${ICON.stop} Gate refused, no LLM call</span>`);
    else if (r.refused) t.push(`<span class="tag warn">${ICON.stop} Refused: not in sources</span>`);
    else if (r.checks.every((c) => c.ok)) t.push(`<span class="tag good">${ICON.shield} Grounded: checks passed</span>`);
    else t.push(`<span class="tag bad">${ICON.warn} Check failed</span>`);
    if (r.attempts === 2) t.push('<span class="tag blue">Retried once</span>');
    if (r.standalone_question && r.standalone_question.trim() !== message.trim()) t.push('<span class="tag blue">Question rewritten</span>');
  } else {
    const bad = r.checks.filter((c) => !c.ok);
    t.push(bad.length ? `<span class="tag bad">${ICON.warn} ${bad.length === 1 ? '1 check would fail' : bad.length + ' checks would fail'}</span>` : '<span class="tag">No checks run</span>');
  }
  t.push(`<span class="ms">${(r.timings.total_ms / 1000).toFixed(1)} s · ${r.mode === 'llm' ? 'LLM' : 'extractive'}</span>`);
  return `<div class="meta-row">${t.join('')}</div>`;
}

/* ---------- inspector ---------- */
function select(id, el, n) {
  document.querySelectorAll('.bubble.sel').forEach((b) => b.classList.remove('sel'));
  el.classList.add('sel');
  renderInspector(results[id]);
  if (n) {
    const s = document.getElementById('src-' + n);
    if (s) { s.scrollIntoView({ block: 'nearest', behavior: 'smooth' }); s.classList.remove('flash'); void s.offsetWidth; s.classList.add('flash'); s.querySelector('details')?.setAttribute('open', ''); }
  }
}

function renderInspector(r) {
  const parts = [`<div class="insp"><h3>${r.pipeline === 'fixed' ? 'Fixed pipeline' : 'Naive pipeline'} · ${esc(r.model)}</h3>`];
  if (r.pipeline === 'fixed' && r.standalone_question) {
    parts.push(`<h3>Question used for retrieval</h3><div class="rewrite">${esc(r.standalone_question)}</div>`);
  } else if (r.pipeline === 'naive') {
    parts.push(`<h3>Question used for retrieval</h3><div class="rewrite">${esc(r.standalone_question)} <span class="muted" style="color:var(--muted)">(latest message only)</span></div>`);
  }
  if (r.gate) {
    const g = r.gate, pct = Math.max(0, Math.min(1, g.best_similarity)) * 100, fl = g.floor * 100;
    parts.push(`<h3>Relevance gate</h3><div class="gate">
      <div style="display:flex;justify-content:space-between"><span>Best excerpt similarity</span><b>${g.best_similarity.toFixed(2)}</b></div>
      <div class="bar"><span style="width:${pct}%"></span><em style="left:${fl}%" title="floor ${g.floor}"></em></div>
      <div style="font-size:12px;color:var(--muted)">Floor ${g.floor.toFixed(2)}${g.exact_code_match ? ' · exact error-code match' : ''}${g.names_not_in_kb?.length ? ' · not in any article: ' + esc(g.names_not_in_kb.join(', ')) : ''}</div>
      <div>${g.passed ? `<span class="tag good">Passed: ${r.mode === 'llm' ? 'model called' : 'answer extracted'}</span>` : `<span class="tag warn">Refused before ${r.mode === 'llm' ? 'the model' : 'answering'}</span>`}</div></div>`);
  }
  parts.push('<h3>Checks</h3><ul class="checks">' + r.checks.map((c) => `<li>${c.ok ? ICON.ok : ICON.bad}<span>${esc(c.name)}<small>${esc(c.detail)}</small></span></li>`).join('') + '</ul>');
  parts.push(`<h3>Retrieved excerpts (${r.sources.length})</h3>`);
  parts.push(r.sources.map((s) => {
    const tags = [];
    if (s.cited) tags.push('<span class="tag blue">cited</span>');
    if (s.exact_code) tags.push('<span class="tag good">exact code match</span>');
    if (s.dense_rank) tags.push(`<span class="tag">dense #${s.dense_rank} · ${Number(s.dense).toFixed(2)}</span>`);
    if (s.bm25_rank) tags.push(`<span class="tag">BM25 #${s.bm25_rank}</span>`);
    if (r.pipeline === 'fixed' && !s.bm25_rank && s.found_by === 'dense') tags.push('<span class="tag">dense only</span>');
    return `<div class="src ${s.cited ? 'cited' : ''}" id="src-${s.n}"><div class="src-h"><span class="src-n">${s.n}</span><div class="src-t">${esc(s.title)}<small>${esc(s.section)} · updated ${esc(s.updated)}</small></div></div>
      <div class="tags">${tags.join('')}</div><details><summary>Show excerpt text</summary><pre>${esc(s.text)}</pre></details></div>`;
  }).join(''));
  parts.push(`<div class="note">Retrieval ${r.timings.retrieve_ms} ms · generation ${r.timings.generate_ms} ms</div></div>`);
  $('#inspector').innerHTML = parts.join('');
}

/* ---------- eval ---------- */
let evalDone = false;
async function loadEval() {
  if (evalDone) return; evalDone = true;
  const el = $('#eval');
  el.innerHTML = '<p class="note">Loading…</p>';
  let d; try { d = await (await fetch('/api/eval')).json(); } catch (e) { el.innerHTML = '<p class="err">Couldn’t load eval results.</p>'; return; }
  const S = d.summary, n = S.naive, f = S.fixed;
  const frac = (a) => `${a[0]}<small>/${a[1]}</small>`;
  const kpi = (label, key, sub, lowerBetter) => `<div class="kpi"><div class="l">${label}</div><div class="pair"><div class="v n" title="Naive">${frac(n[key])}</div><div class="v f" title="Fixed">${frac(f[key])}</div></div><div class="sub">${sub}</div></div>`;
  const TYPES = { fact: 'Facts', table: 'Table lookups', code: 'Error codes', false_premise: 'False premises', updated_fact: 'Updated facts', multi_part: 'Multi-part', follow_up: 'Follow-ups', out_of_scope: 'Not in the docs' };
  const typeRows = Object.keys(TYPES).filter((k) => f.by_type[k]).map((k) => {
    const [an, tn] = n.by_type[k], [af, tf] = f.by_type[k];
    return `<div class="trow"><span>${TYPES[k]} <span style="color:var(--muted)">(${tn})</span></span><div class="tbars">
      <div class="tbar"><div class="track"><div class="fill n" style="width:${(an / tn) * 100}%"></div></div><b>${an}/${tn}</b></div>
      <div class="tbar"><div class="track"><div class="fill f" style="width:${(af / tf) * 100}%"></div></div><b>${af}/${tf}</b></div></div></div>`;
  }).join('');
  const rows = d.rows.map((r, i) => `<tr class="q" data-i="${i}"><td>${esc(r.q)}${r.history.length ? '<br><span class="tag blue" style="margin-top:4px">follow-up</span>' : ''}</td><td>${TYPES[r.type] || r.type}</td>
    <td>${pf(r.naive.ok)}</td><td>${pf(r.fixed.ok)}</td></tr>
    <tr class="detail" data-for="${i}" hidden><td colspan="4">${r.history.length ? `<div class="note" style="margin:0 0 8px">Earlier in the chat: “${esc(r.history[0].content)}”</div>` : ''}<div class="ans2">
      <div><b style="color:var(--naive)">Naive</b>${esc(r.naive.answer)}</div><div><b style="color:var(--accent-text)">Fixed</b>${esc(r.fixed.answer)}${r.fixed.standalone !== r.q ? `<div class="note">Retrieved with: ${esc(r.fixed.standalone)}</div>` : ''}</div></div></td></tr>`).join('');
  el.innerHTML = `<h2>33 questions through both pipelines</h2>
    <p class="lede">Same documents, same ${d.model.startsWith('none') ? 'extractive answerer' : 'model (' + esc(d.model) + ')'}, same embeddings. A question passes when the answer has every required fact, or refuses when the answer isn’t in the help center. Run it yourself with <code>python -m eval.run_eval</code>.</p>
    <div class="legend"><span><i class="dot n"></i>Naive</span><span><i class="dot f"></i>Fixed</span><span style="color:var(--muted)">Ran ${esc(d.ran_at)}</span></div>
    <div class="kpis">
      ${kpi('Answers correct', 'answers_correct', 'Required facts present, or a refusal when it should refuse')}
      ${kpi('Out-of-scope refused', 'out_of_scope_refused', 'CEO, Salesforce, phone number, Dropbox, KB-777')}
      ${kpi('Right excerpt retrieved', 'retrieval_hit', 'In-scope questions where the evidence was in the top results')}
      ${kpi('In-scope wrongly refused', 'in_scope_wrongly_refused', 'Lower is better: the cost of being careful')}
    </div>
    <div class="card"><div class="card-h"><b>Correct answers by question type</b><span>Top bar naive, bottom bar fixed</span></div><div class="types">${typeRows}</div></div>
    <div class="card"><div class="card-h"><b>Every question</b><span>Click a row to compare the two answers</span></div><div class="scroll"><table>
      <thead><tr><th>Question</th><th>Type</th><th>Naive</th><th>Fixed</th></tr></thead><tbody>${rows}</tbody></table></div></div>`;
  el.querySelectorAll('tr.q').forEach((tr) => tr.addEventListener('click', () => { const d = el.querySelector(`tr[data-for="${tr.dataset.i}"]`); d.hidden = !d.hidden; }));
}
const pf = (ok) => ok ? `<span class="pf p">${ICON.ok} Pass</span>` : `<span class="pf f">${ICON.bad} Fail</span>`;

/* ---------- knowledge base ---------- */
let kbDone = false;
async function loadKb() {
  if (kbDone || !meta) return; kbDone = true;
  $('#kb-list').innerHTML = meta.docs.map((d, i) => `<button data-s="${esc(d.source)}" class="${i ? '' : 'on'}">${esc(d.title)}<small>updated ${esc(d.updated)}</small></button>`).join('');
  $('#kb-list').addEventListener('click', (e) => { const b = e.target.closest('button'); if (b) openDoc(b.dataset.s, b); });
  openDoc(meta.docs[0].source, $('#kb-list button'));
}
async function openDoc(src, btn) {
  document.querySelectorAll('#kb-list button').forEach((b) => b.classList.toggle('on', b === btn));
  const raw = await (await fetch('/api/kb/' + encodeURIComponent(src))).text();
  $('#kb-doc').innerHTML = md(raw);
}
function md(raw) {
  const fm = raw.match(/^---\n([\s\S]*?)\n---\n/); let body = fm ? raw.slice(fm[0].length) : raw;
  const out = []; const lines = body.split('\n'); let i = 0;
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
  while (i < lines.length) {
    const l = lines[i];
    if (/^#{1,3} /.test(l)) { const h = l.match(/^#+/)[0].length; out.push(`<h${h}>${inline(l.slice(h + 1))}</h${h}>`); i++; continue; }
    if (l.startsWith('|')) { const rows = []; while (i < lines.length && lines[i].startsWith('|')) rows.push(lines[i++]); const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      out.push('<table><thead><tr>' + cells(rows[0]).map((c) => `<th>${inline(c)}</th>`).join('') + '</tr></thead><tbody>' + rows.slice(2).map((r) => '<tr>' + cells(r).map((c) => `<td>${inline(c)}</td>`).join('') + '</tr>').join('') + '</tbody></table>'); continue; }
    if (/^(- |\d+\. )/.test(l)) { const ord = /^\d/.test(l); const items = []; while (i < lines.length && /^(- |\d+\. )/.test(lines[i])) items.push(lines[i++].replace(/^(- |\d+\. )/, ''));
      out.push(`<${ord ? 'ol' : 'ul'}>` + items.map((x) => `<li>${inline(x)}</li>`).join('') + `</${ord ? 'ol' : 'ul'}>`); continue; }
    if (l.trim()) out.push(`<p>${inline(l)}</p>`);
    i++;
  }
  const upd = fm && fm[1].match(/updated:\s*(.+)/);
  return (upd ? `<div class="fm">Updated ${esc(upd[1])} · this text is what the bot retrieves from</div>` : '') + out.join('');
}

/* ---------- boot ---------- */
fetch('/api/meta').then((r) => r.json()).then((m) => {
  meta = m;
  const p = $('#model-pill');
  p.innerHTML = m.llm ? `<i></i>${esc(m.model)}` : '<i></i>No LLM key: extractive answers';
  p.classList.toggle('warn', !m.llm);
  p.title = `Embeddings: ${m.embeddings} · ${m.chunks.fixed} chunks (fixed) / ${m.chunks.naive} (naive)`;
  const h = location.hash.slice(1); if (['eval', 'kb', 'how'].includes(h)) showTab(h);
}).catch(() => { $('#model-pill').textContent = 'API offline'; });
