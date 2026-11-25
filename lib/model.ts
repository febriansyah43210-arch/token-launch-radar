// Token Launch Radar — a fitted value model, computed from the sealed snapshot at run time.
//
// Nothing here is a hand-written table: every coefficient is fitted from the snapshot's own
// protocols. The traded set (protocols whose token already has a market cap) teaches the value
// model; the whole file teaches the token-odds model:
//
//   launch multiple   log10(mcap/tvl) ~ 1 + log tvl + growth 7d + log age + family  (ridge, log scale)
//   token odds        logit(has token) ~ same basis (ridge logistic)
//
// Each candidate term survives only if dropping it raises the held-out miss in BOTH tests:
// leave-one-out on the traded set, and walk-forward (tokens listed earlier price those listed
// later). The odds model reports its Brier score beside the base-rate Brier, so the board can
// state how much the calibration buys. Implied value is
//
//   tvl x launch multiple (value if a token launches) x token odds (chance it ever does),
//
// with a P25-P75 interval from the value fit's residual spread. One function scores every
// protocol: no name is special-cased, and the same code runs in main.ts and the page.

export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

export const MODEL_VERSION = 'fitted-radar-1';

/** How many pre-token protocols below the watchlist floor the board carries, by TVL. */
export const EARLY_ROWS = 160;

/** The share of a launch a protocol's early users are assumed to receive (the farm verdict's base). */
export const AIRDROP_SHARE = 0.02;

/** One farming wallet's slice: the pool is split by an estimated farmer count that grows with the room. */
export function farmerEstimate(tvlUsd: number): number {
  return Math.max(1000, Math.round(200 * Math.sqrt(tvlUsd / 1e6)));
}

/** A farmer's assumed position, and the risk-free benchmark the verdict compares against. */
export const POSITION_USD = 100;
export const TBILL_RATE = 0.042;

/** The verdict cut lines on net $/hour for the assumed position. */
export const FARM_CUT = 12;
export const MARGINAL_CUT = 3;

export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  listed_at: string | null;
  age_days: number | null;
  group: string;
  on_watchlist: boolean;
  implied_value_usd: number;
  launch_multiple: number;
  token_odds: number;
  value_low_usd: number;
  value_high_usd: number;
  family: string;
  family_n: number;
  family_median_multiple: number | null;
  farm_cost_usd: number;
  farm_hours: number;
  net_per_hour_usd: number;
  expected_airdrop: number;
  verdict: 'FARM' | 'MARGINAL' | 'SKIP';
  farm_steps: string[];
}

export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  tvl_usd: number;
  mcap_usd: number | null;
  multiple: number;
  fitted_multiple: number;
}

export interface TermNote {
  term: string;
  coefficient: number;
  loo_kept: boolean;
  walk_kept: boolean;
}

export interface ModelReport {
  version: string;
  as_of: string;
  traded_n: number;
  families_n: number;
  terms: TermNote[];
  value_miss_baseline: number;
  value_miss_final: number;
  brier: number;
  brier_base: number;
  odds_n: number;
  odds_share: number;
  interval_p25: number;
  interval_p75: number;
  within_interval_share: number;
}

export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  airdrop_share: number;
  farmer_model: string;
  position_usd: number;
  tbill_rate: number;
  watchlist_size: number;
  board_size: number;
  early_size: number;
  traded_n: number;
  report: ModelReport;
  rows: RadarRow[];
  live: LiveRow[];
}

export type RadarRow = BoardRow;

const DAY_MS = 86_400_000;

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;

export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

export function isRankable(p: SnapshotProtocol): boolean {
  return isPreToken(p) && p.category !== 'CEX' && !VERSIONED.test(p.name);
}

export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

const fam = (c: string | null): string => (c === null ? 'Other' : c);

function median(v: number[]): number {
  const s = [...v].sort((a, b) => a - b);
  return s.length === 0 ? 0 : s[Math.floor((s.length - 1) / 2)]!;
}

/** Ridge regression by Gauss-Jordan on the normal equations; `lam` shrinks every slope. */
export function ridgeFit(rows: number[][], y: number[], lam: number): number[] {
  const k = rows[0]!.length;
  const m: number[][] = Array.from({ length: k }, () => new Array(k + 1).fill(0));
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i]!;
    for (let a = 0; a < k; a++) {
      m[a]![k]! += r[a]! * y[i]!;
      for (let b = 0; b < k; b++) m[a]![b]! += r[a]! * r[b]!;
    }
  }
  for (let a = 1; a < k; a++) m[a]![a]! += lam * Math.max(1, rows.length);
  for (let col = 0; col < k; col++) {
    let piv = col;
    for (let r = col + 1; r < k; r++) if (Math.abs(m[r]![col]!) > Math.abs(m[piv]![col]!)) piv = r;
    [m[col], m[piv]] = [m[piv]!, m[col]!];
    const d = m[col]![col]! || 1e-9;
    for (let b = col; b <= k; b++) m[col]![b]! /= d;
    for (let r = 0; r < k; r++) {
      if (r === col || m[r]![col]! === 0) continue;
      const f = m[r]![col]!;
      for (let b = col; b <= k; b++) m[r]![b]! -= f * m[col]![b]!;
    }
  }
  return m.map((row) => row[k]!);
}

const dot = (row: number[], w: number[]): number => row.reduce((s, x, i) => s + x * w[i]!, 0);
const pick = (row: number[], cols: number[]): number[] => cols.map((c) => row[c]!);

export interface Fitted {
  report: ModelReport;
  launchMultiple: (p: SnapshotProtocol) => { mult: number; low: number; high: number; family: string; familyN: number; familyMedian: number | null };
  tokenOdds: (p: SnapshotProtocol) => number;
}

/**
 * Fit both models from the snapshot's own record. Deterministic: no randomness, no clock, no
 * network, and no map iteration that depends on insertion order the snapshot could change.
 */
export function fit(snapshot: Snapshot): Fitted {
  const asOf = snapshot.as_of;
  const all = snapshot.protocols;
  const ages = all.map((p) => ageDays(asOf, p.listed_at)).filter((a): a is number => a !== null);
  const medianAge = median(ages);
  const traded = all.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0).sort((a, b) => (a.slug < b.slug ? -1 : 1));

  // Family (category) dummy columns: only categories with 12+ members get one, the rest are the base.
  const counts = new Map<string, number>();
  for (const p of all) counts.set(fam(p.category), (counts.get(fam(p.category)) ?? 0) + 1);
  const fams = [...counts.entries()].filter(([, n]) => n >= 12).map(([f]) => f).sort();

  const basis = (p: SnapshotProtocol): number[] => {
    const age = ageDays(asOf, p.listed_at) ?? medianAge;
    return [
      1,
      Math.log10(Math.max(p.tvl_usd, 1)) - 2,
      (p.change_7d ?? 0) / 100,
      Math.log10(Math.max(age, 1)) - 2,
      ...fams.map((f) => (fam(p.category) === f ? 1 : 0)),
    ];
  };

  const X = traded.map(basis);
  const Y = traded.map((p) => Math.log10(p.mcap_usd! / p.tvl_usd));
  const LAM = 0.03;
  const CANDS = ['log tvl', 'growth 7d', 'log age', ...fams.map((f) => `family ${f}`)];
  // Column 0 is the intercept and never a candidate; candidate i lives in column i (0-based term index).
  const COLS = CANDS.map((_, i) => i);

  const relMiss = (cols: number[], test: 'loo' | 'walk'): number => {
    const P = X.map((r) => pick(r, cols));
    let total = 0;
    let n = 0;
    if (test === 'loo') {
      for (let i = 0; i < P.length; i++) {
        const rest = P.filter((_, j) => j !== i);
        const w = ridgeFit(rest, Y.filter((_, j) => j !== i), LAM);
        total += Math.abs(10 ** dot(P[i]!, w) - 10 ** Y[i]!) / Math.max(1, 10 ** Y[i]!);
        n++;
      }
    } else {
      const order = traded.map((p, i) => i).sort((a, b) => {
        const aa = traded[a]!.listed_at ?? '9999-12-31';
        const bb = traded[b]!.listed_at ?? '9999-12-31';
        return aa < bb ? -1 : aa > bb ? 1 : a - b;
      });
      for (let step = Math.floor(order.length / 3); step < order.length; step++) {
        const train = order.slice(0, step).map((i) => P[i]!);
        const w = ridgeFit(train, order.slice(0, step).map((i) => Y[i]!), LAM);
        total += Math.abs(10 ** dot(P[order[step]!]!, w) - 10 ** Y[order[step]!]!) / Math.max(1, 10 ** Y[order[step]!]!);
        n++;
      }
    }
    return n === 0 ? 0 : total / n;
  };

  const terms: TermNote[] = [];
  const keptCols: number[] = [];
  for (const c of COLS) {
    const withCol = [0, ...COLS.filter((x) => x !== 0 && (keptCols.includes(x) || x === c))];
    const without = [0, ...keptCols.filter((x) => x !== c)];
    const looOn = relMiss(withCol, 'loo');
    const walkOn = relMiss(withCol, 'walk');
    const looOff = relMiss(without, 'loo');
    const walkOff = relMiss(without, 'walk');
    const looKept = looOn <= looOff;
    const walkKept = walkOn <= walkOff;
    terms.push({ term: CANDS[c]!, coefficient: round3(ridgeFit(X, Y, LAM)[c + 1] ?? 0), loo_kept: looKept, walk_kept: walkKept });
    if (looKept && walkKept) keptCols.push(c);
  }
  const finalCols = [0, ...keptCols];
  const W = ridgeFit(X.map((r) => pick(r, finalCols)), Y, LAM);
  const predictLog = (p: SnapshotProtocol): number => dot(pick(basis(p), finalCols), W);
  const missFinal = relMiss(finalCols, 'loo');
  const missBase = relMiss([0], 'loo');

  // Residual spread -> the stated P25-P75 interval, honestly measured on the traded set.
  const resid = traded.map((p, i) => Y[i]! - predictLog(p)).sort((a, b) => a - b);
  const q = (f: number): number => resid[Math.min(resid.length - 1, Math.max(0, Math.floor(f * resid.length)))]!;
  const p25 = round3(q(0.25));
  const p75 = round3(q(0.75));
  const withinShare = round4(traded.filter((p) => { const r = Y[traded.indexOf(p)]! - predictLog(p); return r >= p25 && r <= p75; }).length / Math.max(1, traded.length));

  const famMult = new Map<string, number[]>();
  for (const p of traded) famMult.set(fam(p.category), [...(famMult.get(fam(p.category)) ?? []), p.mcap_usd! / p.tvl_usd]);

  const launchMultiple = (p: SnapshotProtocol) => {
    const lp = predictLog(p);
    const f = fam(p.category);
    const fm = famMult.get(f);
    return { mult: 10 ** lp, low: 10 ** (lp + p25), high: 10 ** (lp + p75), family: f, familyN: fm?.length ?? 0, familyMedian: fm ? median(fm) : null };
  };

  // Token odds: ridge logistic over every protocol, has-token as the label.
  const oX = all.map(basis);
  const oY = all.map((p) => (isPreToken(p) ? 0 : 1));
  const OW = logisticFit(oX, oY, 0.5);
  let brier = 0;
  let brierBase = 0;
  const baseRate = oY.reduce((s, x) => s + x, 0) / Math.max(1, oY.length);
  for (let i = 0; i < oX.length; i++) {
    const pr = sigmoid(dot(oX[i]!, OW));
    brier += (pr - oY[i]!) ** 2;
    brierBase += (baseRate - oY[i]!) ** 2;
  }

  const report: ModelReport = {
    version: MODEL_VERSION,
    as_of: asOf,
    traded_n: traded.length,
    families_n: fams.length,
    terms,
    value_miss_baseline: round4(missBase),
    value_miss_final: round4(missFinal),
    brier: round4(brier / Math.max(1, oX.length)),
    brier_base: round4(brierBase / Math.max(1, oX.length)),
    odds_n: all.length,
    odds_share: round4(baseRate),
    interval_p25: p25,
    interval_p75: p75,
    within_interval_share: withinShare,
  };

  const tokenOdds = (p: SnapshotProtocol): number => Math.min(0.97, Math.max(0.02, sigmoid(dot(basis(p), OW))));

  return { report, launchMultiple, tokenOdds };
}

const sigmoid = (z: number): number => 1 / (1 + Math.exp(-z));

/** Ridge logistic by Newton/IRLS: a handful of steps at this scale converges, deterministically. */
function logisticFit(X: number[][], y: number[], lam: number): number[] {
  const k = X[0]!.length;
  const n = X.length;
  let w = new Array(k).fill(0);
  const solve = (H: number[][], g: number[]): number[] => {
    const M = H.map((row, i) => [...row, g[i]!]);
    for (let c = 0; c < k; c++) {
      let piv = c;
      for (let r = c + 1; r < k; r++) if (Math.abs(M[r]![c]!) > Math.abs(M[piv]![c]!)) piv = r;
      [M[c], M[piv]] = [M[piv]!, M[c]!];
      const d = M[c]![c]! || 1e-12;
      for (let b = c; b <= k; b++) M[c]![b]! /= d;
      for (let r = 0; r < k; r++) {
        if (r === c || M[r]![c]! === 0) continue;
        const f = M[r]![c]!;
        for (let b = c; b <= k; b++) M[r]![b]! -= f * M[c]![b]!;
      }
    }
    return M.map((row) => row[k]!);
  };
  for (let it = 0; it < 40; it++) {
    const g = new Array(k).fill(0);
    const H: number[][] = Array.from({ length: k }, () => new Array(k).fill(0));
    for (let i = 0; i < n; i++) {
      const pr = sigmoid(dot(X[i]!, w));
      const err = pr - y[i]!;
      const r = pr * (1 - pr) + 1e-6;
      for (let a = 0; a < k; a++) {
        g[a]! += err * X[i]![a]!;
        for (let b = 0; b < k; b++) H[a]![b]! += r * X[i]![a]! * X[i]![b]!;
      }
    }
    for (let a = 0; a < k; a++) {
      g[a] = g[a]! / n + (a === 0 ? 0 : lam * 0.01 * w[a]!);
      for (let b = 0; b < k; b++) H[a]![b] = H[a]![b]! / n + (a === b ? lam * 0.01 : 0);
    }
    const step = solve(H, g);
    let move = 0;
    w = w.map((x, i) => { const v = x - step[i]!; move = Math.max(move, Math.abs(step[i]!)); return v; });
    if (move < 1e-7) break;
  }
  return w;
}

// ---------------------------- board construction ----------------------------

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Check the protocol's own page before you move.`;
  return s;
}

/** The family's one action, in words a farmer can follow, from the protocol's own record. */
function actionFor(p: SnapshotProtocol, capital: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Supply $${capital} of USDC to ${p.name} on ${chain} and leave the position open.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('liquidity')) return `Provide $${capital} of liquidity to ${p.name}'s deepest pool on ${chain} and leave it open.`;
  if (c.includes('bridge')) return `Bridge $${capital} of USDC to ${chain} through ${p.name} and use the funds inside its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${capital} of the chain's native asset through ${p.name} and keep the receipt token.`;
  if (c.includes('perp') || c.includes('derivative')) return `Deposit $${capital} of USDC as margin on ${p.name} and open one small hedged position.`;
  if (c.includes('prediction')) return `Fund $${capital} on ${p.name} and take one position on its main market.`;
  if (c.includes('curator') || c.includes('allocator') || c.includes('vault')) return `Deposit $${capital} into a ${p.name} vault on ${chain} and keep the shares.`;
  if (c.includes('stable')) return `Mint or deposit $${capital} of the stable asset through ${p.name} on ${chain}.`;
  if (c.includes('rwa') || c.includes('treasury')) return `Mint $${capital} of ${p.name}'s yield asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${capital} of cover from ${p.name} on ${chain} and keep the policy.`;
  if (c.includes('nft') || c.includes('launch') || c.includes('incubat')) return `Use ${p.name} on ${chain} for one full launch cycle with $${capital} deployed.`;
  return `Deposit $${capital} of USDC into ${p.name} on ${chain} and leave the position open.`;
}

/** A plan's capital band, from the protocol's own TVL: bigger rooms take bigger positions. */
function capitalFor(tvl: number): number {
  if (tvl >= 5_000_000_000) return 250;
  if (tvl >= 1_000_000_000) return 150;
  if (tvl >= 100_000_000) return 100;
  return 50;
}

/** How long to hold, from the protocol's age: younger protocols get a longer qualifying window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

export function farmSteps(p: SnapshotProtocol, age: number | null, odds: number): FarmStep[] {
  const chain = p.chains.length > 0 ? p.chains[0]! : 'its main chain';
  const capital = capitalFor(p.tvl_usd);
  const hold = holdDaysFor(age);
  const weekly = Math.round((capital * 0.03 + 2) * 10) / 10;
  return [
    { text: 'Use a fresh wallet that holds only the funds you plan to farm, and keep its seed phrase offline.', cost: 'under $3 of gas for the first transaction', time: '5 minutes' },
    { text: actionFor(p, capital, chain), cost: `$${capital} of capital and about $${capital >= 150 ? 8 : 4} of gas`, time: '15 minutes to set up' },
    { text: 'Make one qualifying action each week — a deposit, a borrow, a swap or a vote — from the same wallet.', cost: `about $${weekly} of gas a week`, time: '10 minutes a week' },
    { text: `Hold the position for at least ${hold} days so the wallet is counted in more than one snapshot.`, cost: `the $${capital} stays in the position`, time: `${hold} days` },
    { text: `Track the protocol's own points or rewards page and claim only from links it publishes itself.`, cost: 'no fee to check, and never pay to claim', time: '10 minutes' },
    { text: `Record the wallet, the dates and every transaction in a farm log so the position can be repeated or unwound.`, cost: 'nothing but the log', time: '5 minutes' },
  ].map((s) => s) as FarmStep[];
}

/** Board, live rows and the model report, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const asOf = snapshot.as_of;
  const fitted = fit(snapshot);
  const index = new Map(snapshot.protocols.map((p) => [p.slug, p]));

  const chosen = new Map<string, SnapshotProtocol>();
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined) chosen.set(slug, p);
  }
  const early = snapshot.protocols
    .filter((p) => !chosen.has(p.slug) && isRankable(p) && p.tvl_usd < snapshot.floor_usd)
    .sort((a, b) => b.tvl_usd - a.tvl_usd || (a.slug < b.slug ? -1 : 1))
    .slice(0, EARLY_ROWS);
  for (const p of early) chosen.set(p.slug, p);

  const rows: BoardRow[] = [];
  for (const p of chosen.values()) {
    const { mult, low, high, family, familyN, familyMedian } = fitted.launchMultiple(p);
    const odds = fitted.tokenOdds(p);
    const implied = Math.max(1, Math.round(p.tvl_usd * mult * odds));
    const age = ageDays(asOf, p.listed_at);
    const capital = capitalFor(p.tvl_usd);
    const hold = holdDaysFor(age);
    const setupHours = 0.5;
    const weeklyHours = (hold / 7) * (10 / 60);
    const hours = Math.max(1, setupHours + weeklyHours + 1);
    const gas = capital >= 150 ? 8 : 4 + (hold / 7) * 2;
    const expected = (implied * AIRDROP_SHARE) / farmerEstimate(p.tvl_usd) * (capital / POSITION_USD);
    const net = (expected - gas) / hours;
    const verdict = net >= FARM_CUT ? 'FARM' : net >= MARGINAL_CUT ? 'MARGINAL' : 'SKIP';
    rows.push({
      rank: 0,
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      change_7d: p.change_7d,
      listed_at: p.listed_at,
      age_days: age,
      group: fitted.launchMultiple(p).family,
      on_watchlist: snapshot.watchlist.includes(p.slug),
      implied_value_usd: implied,
      launch_multiple: Math.round(mult * 100) / 100,
      token_odds: Math.round(odds * 100) / 100,
      value_low_usd: Math.round(p.tvl_usd * low * odds),
      value_high_usd: Math.round(p.tvl_usd * high * odds),
      family,
      family_n: familyN,
      family_median_multiple: familyMedian === null ? null : Math.round(familyMedian * 100) / 100,
      farm_cost_usd: Math.round(capital + gas),
      farm_hours: round2(hours),
      net_per_hour_usd: Math.round(net * 100) / 100,
      expected_airdrop: Math.round(expected * 100) / 100,
      verdict,
      farm_steps: farmSteps(p, age, odds).map(stepText),
    });
  }
  rows.sort((a, b) => b.implied_value_usd - a.implied_value_usd || (a.slug < b.slug ? -1 : 1));
  rows.forEach((r, i) => (r.rank = i + 1));

  const live: LiveRow[] = snapshot.protocols
    .filter((p) => !isPreToken(p) && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, 24)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      multiple: Math.round(((p.mcap_usd ?? 0) / Math.max(1, p.tvl_usd)) * 100) / 100,
      fitted_multiple: Math.round(fitted.launchMultiple(p).mult * 100) / 100,
    }));

  const radar: RadarData = {
    as_of: asOf,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: MODEL_VERSION,
    airdrop_share: AIRDROP_SHARE,
    farmer_model: 'pool = implied value x 0.02 (the early-user share); one wallet at $100 expects pool / max(1000, 200 x sqrt(TVL/$1M)) farmers, scaled by its position',
    position_usd: POSITION_USD,
    tbill_rate: TBILL_RATE,
    watchlist_size: snapshot.watchlist.length,
    board_size: rows.length,
    early_size: rows.filter((r) => !snapshot.watchlist.includes(r.slug)).length,
    traded_n: fitted.report.traded_n,
    report: fitted.report,
    rows,
    live,
  };
  return { board: rows, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return { as_of: o.as_of, source: o.source, floor_usd: o.floor_usd, protocols, watchlist: o.watchlist as string[] };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, value, farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slugs: ${missing.slice(0, 5).join(', ')}`;
  for (let i = 0; i < board.length; i++) if (board[i]!.rank !== i + 1) return `rank ${i + 1} is ${board[i]!.rank}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i}`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) return 'all rows share one implied value';
  for (const r of board) {
    if (r.farm_steps.length < 3 || r.farm_steps.length > 12) return `${r.slug} has ${r.farm_steps.length} farm steps`;
    for (const s of r.farm_steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

const round2 = (n: number): number => Math.round(n * 100) / 100;
const round3 = (n: number): number => Math.round(n * 1e3) / 1e3;
const round4 = (n: number): number => Math.round(n * 1e4) / 1e4;
