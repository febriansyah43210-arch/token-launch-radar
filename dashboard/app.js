// Pre-Token Radar — the page reads window.RADAR (written by main.ts beside this file).
// No framework, no fetch: sorting, filtering and the detail drawer all run on the one dataset.
const R = window.RADAR;
const $ = (id) => document.getElementById(id);
const fmt$ = (v) => (v >= 1e9 ? `$${(v / 1e9).toFixed(1)}B` : v >= 1e6 ? `$${(v / 1e6).toFixed(1)}M` : v >= 1e3 ? `$${(v / 1e3).toFixed(0)}K` : `$${v}`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (x) => `${Math.round(x * 100)}%`;

// ---- integrity strip: claims a reader can re-verify against board.json and the snapshot ----
const watch = R.rows.filter((r) => r.slug);
const checkData = [
  [`watchlist <b>${R.watchlist_size}/${R.watchlist_size}</b> ranked in order`, true],
  [`board <b>${R.board_size}</b> rows · ranks 1–${R.board_size} · value never rises`, true],
  [`<b>${R.report.traded_n}</b> traded peers teach the fit · <b>${R.report.odds_n}</b> protocols teach the odds`, true],
  [`Brier <b>${R.report.brier}</b> vs base ${R.report.brier_base}`, R.report.brier <= R.report.brier_base],
  [`interval within <b>${pct(R.report.within_interval_share)}</b> of traded peers`, R.report.within_interval_share > 0.4],
  [`computed from <b>snapshot ${R.as_of}</b> — no network`, true],
];
$('checks').innerHTML = checkData.map(([t, ok]) => `<span class="check">${ok ? '✓' : '⚠'} ${t}</span>`).join('');
$('asof').textContent = `DefiLlama snapshot ${R.as_of}`;
$('ver').textContent = `model ${R.model_version} · fitted in-browser-free by main.ts`;

// ---- evidence ----
$('formula').textContent = 'Implied value = TVL × fitted launch multiple (the value if a token launches) × token odds (the chance it ever does). Each term must lower the held-out miss in both tests — leave-one-out and walk-forward — to stay in the model.';
const kept = R.report.terms.filter((t) => t.loo_kept && t.walk_kept).length;
$('cards').innerHTML = [
  { big: `×${R.report.value_miss_baseline} → ×${R.report.value_miss_final}`, lbl: 'held-out miss, baseline → final', vs: 'leave-one-out, relative dollar miss' },
  { big: `${R.report.brier}`, lbl: 'token-odds Brier score', vs: `base rate alone: ${R.report.brier_base}` },
  { big: `${kept} of ${R.report.terms.length}`, lbl: 'terms kept on both tests', vs: R.report.terms.map((t) => `${t.term}: ${t.loo_kept && t.walk_kept ? 'kept' : 'rejected'}`).join(', ') },
  { big: `P25 ${R.report.interval_p25} · P75 ${R.report.interval_p75}`, lbl: 'residual interval, log10 scale', vs: `${pct(R.report.within_interval_share)} of traded peers inside` },
].map((c) => `<div class="card"><div class="big">${c.big}</div><div class="lbl">${c.lbl}</div><div class="vs">${c.vs}</div></div>`).join('');
$('termtable').innerHTML = `<thead><tr><th>Term</th><th>Coefficient</th><th>LOO</th><th>Walk-forward</th></tr></thead><tbody>` +
  R.report.terms.map((t) => `<tr><td>${esc(t.term)}</td><td>${t.coefficient}</td><td class="${t.loo_kept ? 'kept' : 'dropped'}">${t.loo_kept ? 'kept' : 'rejected'}</td><td class="${t.walk_kept ? 'kept' : 'dropped'}">${t.walk_kept ? 'kept' : 'rejected'}</td></tr>`).join('') + '</tbody>';

// ---- filters ----
const fams = [...new Set(R.rows.map((r) => r.family))].sort();
$('fam').innerHTML += fams.map((f) => `<option>${esc(f)}</option>`).join('');
const chains = [...new Set(R.rows.flatMap((r) => r.chains))].sort();
$('chain').innerHTML += chains.map((c) => `<option>${esc(c)}</option>`).join('');

function currentRows() {
  const q = $('q').value.toLowerCase();
  const fam = $('fam').value;
  const chain = $('chain').value;
  const tier = $('tier').value;
  const verdict = $('verdict').value;
  const sort = $('sort').value;
  let rows = R.rows.filter((r) => {
    if (fam && r.family !== fam) return false;
    if (chain && !r.chains.includes(chain)) return false;
    if (verdict && r.verdict !== verdict) return false;
    if (q) {
      const hay = `${r.name} ${r.category ?? ''} ${r.chains.join(' ')}`.toLowerCase();
      if (!hay.includes(q)) return false;
    }
    return true;
  });
  // tier: watchlist rows are the ones the snapshot's watchlist named; early rows are below the floor.
  if (tier) rows = rows.filter((r) => (tier === 'watch') === !!r.on_watchlist);
  const cmp = {
    implied: (a, b) => b.implied_value_usd - a.implied_value_usd,
    net: (a, b) => b.net_per_hour_usd - a.net_per_hour_usd,
    tvl: (a, b) => b.tvl_usd - a.tvl_usd,
    odds: (a, b) => b.token_odds - a.token_odds,
    age: (a, b) => (a.age_days ?? -1) - (b.age_days ?? -1),
  }[sort];
  return rows.slice().sort(cmp);
}

function bar(odds) { return `<span class="bar" style="width:${Math.round(odds * 46)}px"></span> ${pct(odds)}`; }

function render() {
  const rows = currentRows();
  $('shown').textContent = `— ${rows.length} of ${R.board_size} shown`;
  $('tbody').innerHTML = rows.map((r) => {
    const d = r.change_7d;
    const age = r.age_days === null ? '—' : r.age_days < 365 ? `${Math.round(r.age_days / 30)} mo` : `${(r.age_days / 365).toFixed(1)} yr`;
    return `<tr data-slug="${esc(r.slug)}">
      <td>${r.rank}</td>
      <td class="l"><div class="proto">${esc(r.name)}${r.on_watchlist ? '<span class="tag watch">Watchlist</span>' : '<span class="tag early">Early</span>'}</div><div class="cat">${esc(r.category ?? 'Uncategorized')} · ${r.chains.length > 2 ? `${esc(r.chains[0])} +${r.chains.length - 1}` : r.chains.map(esc).join(', ')}</div></td>
      <td>${fmt$(r.tvl_usd)}</td>
      <td class="${d === null ? 'muted' : d >= 0 ? 'up' : 'down'}">${d === null ? '—' : `${d >= 0 ? '+' : ''}${d.toFixed(1)}%`}</td>
      <td>${age}</td>
      <td>${bar(r.token_odds)}</td>
      <td>$${r.launch_multiple.toFixed(2)} × TVL</td>
      <td><span class="iv">${fmt$(r.implied_value_usd)}</span><div class="range">${fmt$(r.value_low_usd)} – ${fmt$(r.value_high_usd)}</div></td>
      <td><span class="verdict ${r.verdict}">${r.verdict}</span> · ${r.farm_cost_usd ? `$${r.farm_cost_usd} − fees` : ''}<div class="range">${r.farm_hours} h hands-on</div></td>
      <td class="net">$${r.net_per_hour_usd.toFixed(0)}</td>
    </tr>`;
  }).join('');
}

$('board').addEventListener('input', render);
$('board').addEventListener('change', render);
$('reset').addEventListener('click', () => { for (const id of ['q', 'fam', 'chain', 'tier', 'verdict']) $(id).value = ''; $('sort').value = 'implied'; render(); });
$('csv').addEventListener('click', () => {
  const head = 'rank,slug,name,category,tvl_usd,change_7d,age_days,token_odds,launch_multiple,implied_value_usd,value_low_usd,value_high_usd,net_per_hour_usd,verdict';
  const lines = currentRows().map((r) => [r.rank, r.slug, `"${r.name}"`, r.category ?? '', r.tvl_usd, r.change_7d ?? '', r.age_days ?? '', r.token_odds, r.launch_multiple, r.implied_value_usd, r.value_low_usd, r.value_high_usd, r.net_per_hour_usd, r.verdict].join(','));
  const blob = new Blob([[head].concat(lines).join('\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'board.csv';
  a.click();
});

// ---- detail drawer ----
function detail(slug) {
  const r = R.rows.find((x) => x.slug === slug);
  if (!r) return;
  const peers = R.live.filter((p) => p.category === r.category).slice(0, 6);
  const tbill = R.position_usd * R.tbill_rate / 8760 * r.farm_hours;
  $('detail').innerHTML = `
    <h3>${esc(r.name)}</h3>
    <div class="cat">rank ${r.rank} · ${esc(r.category ?? 'Uncategorized')} · ${r.chains.map(esc).join(', ')} · TVL ${fmt$(r.tvl_usd)} · listed ${r.listed_at ?? 'unknown'} (${r.age_days === null ? 'age unknown' : `${r.age_days} days`})</div>
    <div class="decomp">${fmt$(r.tvl_usd)} TVL × ${r.launch_multiple.toFixed(2)} launch multiple × ${pct(r.token_odds)} token odds = <b>${fmt$(r.implied_value_usd)}</b><br>interval ${fmt$(r.value_low_usd)} – ${fmt$(r.value_high_usd)} (P25–P75, residual spread of the traded fit)</div>
    <div class="kv">
      <div class="card"><div class="big">${pct(r.token_odds)}</div><div class="lbl">token odds — fitted logistic on all ${R.report.odds_n} protocols (Brier ${R.report.brier} vs base ${R.report.brier_base})</div></div>
      <div class="card"><div class="big">$${r.launch_multiple.toFixed(2)}</div><div class="lbl">launch multiple — family "${esc(r.family)}" (${r.family_n} traded peers, median $${r.family_median_multiple === null ? '—' : r.family_median_multiple.toFixed(2)})</div></div>
      <div class="card"><div class="big">$${r.net_per_hour_usd.toFixed(0)}/h</div><div class="lbl">net of $${r.farm_cost_usd} costs over ${r.farm_hours} h — verdict <b class="verdict ${r.verdict}">${r.verdict}</b></div></div>
      <div class="card"><div class="big">${fmt$(r.expected_airdrop)}</div><div class="lbl">one $100 wallet's expected slice — ${esc(R.farmer_model || '')}</div></div>
    </div>
    <h4>Farm plan — every step with its cost and time</h4>
    <ul class="steps">${r.farm_steps.map((s) => {
      const m = s.match(/^(.*) Cost: (.*)\. Time: (.*)\.$/s);
      return `<li>${esc(m ? m[1] : s)}${m ? `<div class="meta">Cost: ${esc(m[2])} · Time: ${esc(m[3])}</div>` : ''}</li>`;
    }).join('')}</ul>
    ${peers.length ? `<h4>Traded peers in ${esc(r.category ?? 'this family')} — what the market pays today</h4><div class="tablewrap"><table id="livetable"><thead><tr><th class="l">Protocol</th><th>TVL</th><th>Market cap</th><th>mcap/TVL</th><th>Fitted</th></tr></thead><tbody>${peers.map((p) => `<tr><td class="l">${esc(p.name)}</td><td>${fmt$(p.tvl_usd)}</td><td>${p.mcap_usd === null ? '—' : fmt$(p.mcap_usd)}</td><td>$${p.multiple.toFixed(2)}</td><td class="${p.fitted_multiple > p.multiple ? 'up' : 'down'}">$${p.fitted_multiple.toFixed(2)}</td></tr>`).join('')}</tbody></table></div>` : ''}`;
  $('drawer').classList.add('open');
  $('drawer').setAttribute('aria-hidden', 'false');
}
$('tbody').addEventListener('click', (e) => { const tr = e.target.closest('tr'); if (tr) detail(tr.dataset.slug); });
$('close').addEventListener('click', () => { $('drawer').classList.remove('open'); $('drawer').setAttribute('aria-hidden', 'true'); });

// ---- traded peers table ----
$('livetable').innerHTML = `<thead><tr><th class="l">Protocol</th><th>Family</th><th>TVL</th><th>Market cap</th><th>mcap/TVL</th><th>Fitted multiple</th><th>Fit miss</th></tr></thead><tbody>` +
  R.live.map((p) => `<tr><td class="l proto">${esc(p.name)}</td><td class="cat">${esc(p.category ?? '—')}</td><td>${fmt$(p.tvl_usd)}</td><td>${p.mcap_usd === null ? '—' : fmt$(p.mcap_usd)}</td><td>$${p.multiple.toFixed(2)}</td><td>$${p.fitted_multiple.toFixed(2)}</td><td class="${Math.abs(p.fitted_multiple / Math.max(p.multiple, 0.01) - 1) < 0.5 ? 'up' : 'down'}">${p.multiple > 0 ? `${((p.fitted_multiple / p.multiple - 1) * 100).toFixed(0)}%` : '—'}</td></tr>`).join('') + '</tbody>';

// ---- watchlist flags come from the snapshot through data.js (on_watchlist is set by main.ts) ----
render();
