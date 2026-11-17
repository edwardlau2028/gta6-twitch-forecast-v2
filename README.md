# gta6-twitch-forecast-v2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/edwardlau2028/gta6-twitch-forecast-v2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/edwardlau2028/gta6-twitch-forecast-v2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Second model for the same bounty. Forecasts Twitch viewers for GTA VI's launch day (2026-11-19 UTC), hours 2–24, as five quantiles (5/25/50/75/95%) per hour. Pure Python 3.13 stdlib, offline and deterministic. Method: kernel-weighted trajectory ensemble — every comparable launch votes, weighted by how close its hour-1 size is to this case (Gaussian kernel, bandwidth 0.45 log units); each hour's centre is the weighted mean of the comparables' log ratios and the spread is their weighted standard deviation. Leave-one-out pinball loss on the kept comparables: 0.1566, vs 0.2721 for the starter persistence baseline and 0.1880 for the first model.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
