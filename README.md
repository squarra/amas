# Adversarial Multi-Agent Systems

Pursuit-evasion on a 14×14 grid: pursuers try to tag evaders (capture = Manhattan distance ≤ 1).

## Setup

```bash
uv sync
```

## Strategies

| Role    | Name     | Description                                        |
| ------- | -------- | -------------------------------------------------- |
| evader  | `random` | random legal move                                  |
| evader  | `greedy` | maximizes distance to nearest visible pursuer      |
| evader  | `genetic`| genetic algorithm evolves a 5-move escape plan     |
| pursuer | `random` | random legal move                                  |
| pursuer | `greedy` | minimizes distance to nearest visible evader       |
| pursuer | `astar`  | A* to the nearest visible evader                   |
| pursuer | `genetic`| genetic algorithm evolves a 5-move chase plan      |
| pursuer | `dqn`    | Deep Q-Network, needs `dqn_pursuer.pt` (see below) |

## Run

```bash
uv run tag                         # watch random vs random
uv run tag -p astar -e greedy      # pick strategies
uv run tag -p dqn -n 200           # headless: capture rate + mean steps over 200 episodes
```

Options: `-ne/-np` number of evaders/pursuers, `-r` view radius (default: full map),
`-s` seed, `-f` fps, `-m` max steps. Esc quits.

## Training

```bash
uv run tag-train                   # DQN pursuer vs greedy evader, 2000 episodes
```

Weights are not in git. Download `dqn_pursuer.pt` from "universe link" into the repo root.
A model only works with the radius (`-r`) it was trained with.
