// Token Launch Radar — the entrypoint the grader runs (Node 22, standard library only).
//
// Reads the sealed snapshot from /input/snapshot.json (or --input <file|dir>), fits both models
// from it, writes the ranked board to /output/board.json, writes the dashboard's data beside the
// page, copies dashboard/ and screenshots/ to /output, and writes /output/dossier.json with the
// vendored make.fun library — byte for byte what the scorer's own gate reads.
//
// Outside the grader's container, MAKEFUN_OUTPUT_DIR stands in for /output:
//   mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input ../input/snapshot.json

import { copyFileSync, existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { boardIssue, buildRadar, dataScript, parseSnapshot } from './lib/model.ts';
import { buildDossier } from './vendor/makefun-board.ts';

const here = fileURLToPath(new URL('.', import.meta.url));

const TEXT_MAX_BYTES = 65_536;
const SHOT_CAP = 512 * 1024;
const SHOT_MAX = 4;
const IMAGE_NAME = /^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/;

function arg(name: string, fallback: string): string {
  const at = process.argv.indexOf(`--${name}`);
  if (at >= 0 && at + 1 < process.argv.length) return process.argv[at + 1]!;
  const eq = process.argv.find((a) => a.startsWith(`--${name}=`));
  return eq === undefined ? fallback : eq.slice(name.length + 3);
}

const inputArg = arg('input', '/input');
const inputFile = inputArg.endsWith('.json') ? inputArg : join(inputArg, 'snapshot.json');
if (!existsSync(inputFile)) {
  process.stderr.write(`token radar: no snapshot at ${inputFile}: pass --input <snapshot.json|dir>\n`);
  process.exit(2);
}
const out = arg('output', process.env.MAKEFUN_OUTPUT_DIR ?? '/output');

const snapshot = parseSnapshot(readFileSync(inputFile, 'utf8'));
const t0 = Date.now();
const { board, radar } = buildRadar(snapshot);
const issue = boardIssue(board, snapshot);
if (issue !== null) {
  process.stderr.write(`token radar: the model wrote a board the gate refuses: ${issue}\n`);
  process.exit(1);
}

mkdirSync(out, { recursive: true });
writeFileSync(join(out, 'board.json'), `${JSON.stringify({ rows: board })}\n`);
const data = dataScript(radar);
writeFileSync(join(out, 'data.js'), data);
writeFileSync(join(out, 'dashboard', 'data.js'), data);
copyDir(join(here, 'dashboard'), join(out, 'dashboard'));
copyDir(join(here, 'screenshots'), join(out, 'screenshots'));
writeFileSync(join(out, 'dossier.json'), `${JSON.stringify(buildDossier(out, { text_max_bytes: TEXT_MAX_BYTES }))}\n`);

const fitMs = Date.now() - t0;
process.stdout.write(`token radar: ${board.length} rows (${radar.early_size} early, watchlist ${radar.watchlist_size}/${snapshot.watchlist.length}), fitted in ${fitMs} ms\n`);
process.stdout.write(`token radar: miss ${radar.report.value_miss_baseline} -> ${radar.report.value_miss_final}, Brier ${radar.report.brier_base} -> ${radar.report.brier}\n`);

/** Copy one directory's regular files into `to`, skipping dot-names and subdirectories. */
function copyDir(from: string, to: string): void {
  if (!existsSync(from)) return;
  mkdirSync(to, { recursive: true });
  for (const name of readdirSync(from)) {
    if (name.startsWith('.')) continue;
    const at = join(from, name);
    const s = statSync(at);
    if (s.isFile()) copyFileSync(at, join(to, name));
  }
}
