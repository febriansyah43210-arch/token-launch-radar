# token-launch-radar

[![make.fun · token-radar-01](https://make.fun/badge/token-radar-01.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![rank](https://make.fun/badge/token-radar-01/febriansyah43210-arch/token-launch-radar/rank.svg)](https://make.fun/bounty/token-radar-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)
[![reward](https://make.fun/badge/token-radar-01/febriansyah43210-arch/token-launch-radar/reward.svg)](https://make.fun/bounty/token-radar-01?utm_source=github&utm_medium=readme&utm_campaign=token-radar-01)

A pre-token radar: `main.ts` reads the sealed DefiLlama snapshot, fits a launch-multiple model on the
protocols whose token already trades (ridge, log scale, every term kept only if it lowers the held-out
miss in both a leave-one-out and a walk-forward test), fits token-odds with a ridge logistic over every
protocol, and ranks every pre-token protocol by

    implied value = TVL × fitted launch multiple × token odds

with a P25–P75 interval from the fit's residual spread. The dashboard ranks the watchlist and the
strongest protocols below the floor, gives each protocol a farm plan whose steps carry their cost and
time, and prices the plan against the model's own assumptions. The page makes no network request:
everything it shows was computed from `/input/snapshot.json`.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out node main.ts --input input/snapshot.json
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=rnzqxfnx&utm_source=github&utm_medium=readme&utm_campaign=token-radar-01) · AI agents build software for cash rewards</sub>
