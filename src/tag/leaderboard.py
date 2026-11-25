import argparse
from pathlib import Path

import numpy as np
import torch

from tag.arena import EVADER, PURSUER
from tag.dqn import DQN_PATH, DQNPursuer
from tag.episode import evaluate
from tag.strategies import EVADER_STRATEGIES, PURSUER_STRATEGIES


def ci95(p: float, n: int) -> float:
    return 1.96 * np.sqrt(p * (1 - p) / n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross table: every pursuer vs every evader on the same seeds")
    parser.add_argument("-ne", "--evaders", type=int, default=1)
    parser.add_argument("-np", "--pursuers", type=int, default=3)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-r", "--radius", type=int)
    parser.add_argument("-m", "--max-steps", type=int, default=200)
    parser.add_argument("-n", "--episodes", type=int, default=100)
    args = parser.parse_args()

    pursuers = dict(PURSUER_STRATEGIES)
    # DQN weights only work with the radius they were trained with; skip instead of crashing mid-run
    if Path(DQN_PATH).exists() and torch.load(DQN_PATH)["radius"] == args.radius:
        pursuers["dqn"] = DQNPursuer(DQN_PATH)

    # same seed range for every cell (common random numbers), so cells are directly comparable
    rates = np.zeros((len(pursuers), len(EVADER_STRATEGIES)))
    steps = rates.copy()
    for i, pursuer in enumerate(pursuers.values()):
        for j, evader in enumerate(EVADER_STRATEGIES.values()):
            strategies = {PURSUER: pursuer, EVADER: evader}
            rates[i, j], steps[i, j] = evaluate(strategies, args.episodes, args.evaders, args.pursuers, args.seed, args.max_steps, args.radius)

    p_names, e_names = list(pursuers), list(EVADER_STRATEGIES)
    out = [
        (
            f"Capture rate ±95% CI (mean steps). {args.pursuers} pursuers vs {args.evaders} evaders, {args.episodes} episodes, "
            f"seeds {args.seed}–{args.seed + args.episodes - 1}, radius {args.radius or 'full'}, max steps {args.max_steps}\n"
        ),
        "| pursuer \\ evader | " + " | ".join(e_names) + " | mean | worst |",
        "|---" * (len(e_names) + 3) + "|",
    ]
    # rank pursuers by mean capture rate; "worst" = maximin security level
    for i in np.argsort(-rates.mean(axis=1)):
        cells = [f"{rates[i, j]:.0%}±{ci95(rates[i, j], args.episodes):.0%} ({steps[i, j]:.0f})" for j in range(len(e_names))]
        out.append(f"| {p_names[i]} | " + " | ".join(cells) + f" | {rates[i].mean():.0%} | {rates[i].min():.0%} |")

    escape = 1 - rates
    out.append("\n| evader | mean escape | worst escape |\n|---|---|---|")
    for j in np.argsort(-escape.mean(axis=0)):
        out.append(f"| {e_names[j]} | {escape[:, j].mean():.0%} | {escape[:, j].min():.0%} |")

    text = "\n".join(out)
    print(text)
    Path("results.md").write_text(text + "\n")
