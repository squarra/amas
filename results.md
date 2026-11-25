Capture rate ±95% CI (mean steps). 3 pursuers vs 1 evaders, 100 episodes, seeds 0–99, radius full, max steps 200

| pursuer \ evader | random | greedy | genetic | mean | worst |
|---|---|---|---|---|---|
| greedy | 100%±0% (8) | 99%±2% (24) | 93%±5% (44) | 97% | 93% |
| genetic | 100%±0% (7) | 100%±0% (19) | 92%±5% (47) | 97% | 92% |
| dqn | 100%±0% (13) | 80%±8% (53) | 79%±8% (60) | 86% | 79% |
| astar | 100%±0% (8) | 70%±9% (72) | 88%±6% (48) | 86% | 70% |
| random | 87%±7% (69) | 10%±6% (190) | 6%±5% (195) | 34% | 6% |

| evader | mean escape | worst escape |
|---|---|---|
| genetic | 28% | 7% |
| greedy | 28% | 0% |
| random | 3% | 0% |
