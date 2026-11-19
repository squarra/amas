import heapq
from collections.abc import Callable

import numpy as np

from tag.arena import MOVES, OPPONENTS, STAY, WALLS, is_free

type Strategy = Callable[[np.ndarray, np.random.Generator], int]


def center(obs: np.ndarray) -> np.ndarray:
    return np.full(2, obs.shape[1] // 2)


def legal_actions(obs: np.ndarray) -> list[int]:
    return [a for a, move in enumerate(MOVES) if is_free(obs[WALLS], *(center(obs) + move))]


def nearest_distance(cell: np.ndarray, others: np.ndarray) -> int:
    return int(np.abs(others - cell).sum(axis=1).min())


def random_strategy(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Choose one legal action at random, including staying in place."""
    return int(rng.choice(legal_actions(obs)))


def greedy(obs: np.ndarray, rng: np.random.Generator, sign: int) -> int:
    opponents = np.argwhere(obs[OPPONENTS])
    if len(opponents) == 0:
        return random_strategy(obs, rng)
    actions = legal_actions(obs)
    scores = [sign * nearest_distance(center(obs) + MOVES[a], opponents) for a in actions]
    best = [a for a, s in zip(actions, scores) if s == max(scores)]
    return int(rng.choice(best))


def greedy_pursuer(obs: np.ndarray, rng: np.random.Generator) -> int:
    return greedy(obs, rng, -1)


def greedy_evader(obs: np.ndarray, rng: np.random.Generator) -> int:
    return greedy(obs, rng, 1)


def astar_pursuer(obs: np.ndarray, rng: np.random.Generator) -> int:
    """Find a shortest route within the view to a cell next to a visible evader and return its first action."""
    evaders = np.argwhere(obs[OPPONENTS])
    if len(evaders) == 0:
        return random_strategy(obs, rng)
    start = tuple(center(obs))

    def heuristic(cell):
        # Capture happens at distance <= 1, so stop one cell before the evader.
        return max(0, nearest_distance(np.array(cell), evaders) - 1)

    # Entries contain: estimated total cost, travelled cost, cell, first action.
    frontier = [(heuristic(start), 0, start, STAY)]
    costs = {start: 0}
    while frontier:
        _, cost, cell, first_action = heapq.heappop(frontier)
        if cost != costs[cell]:
            continue
        if heuristic(cell) == 0:
            return first_action
        for action, move in enumerate(MOVES):
            if action == STAY:
                continue
            next_cell = (cell[0] + move[0], cell[1] + move[1])
            if not is_free(obs[WALLS], *next_cell):
                continue
            new_cost = cost + 1
            if new_cost < costs.get(next_cell, float("inf")):
                costs[next_cell] = new_cost
                first = action if cell == start else first_action
                priority = new_cost + heuristic(next_cell)
                heapq.heappush(frontier, (priority, new_cost, next_cell, first))
    return STAY  # No route to any evader within the view.


def genetic(obs: np.ndarray, rng: np.random.Generator, sign: int) -> int:
    p_size = 15  # this is the size of population
    generations = 10  # total how many geneations will be created
    c_length = 5  # this is the length of off-sprimg / chromosome
    mutation = 0.1  # this is the percentage of mutation i.e. 10%

    opponents = np.argwhere(obs[OPPONENTS])

    if len(opponents) == 0:
        return STAY

    # 1: Generating random population of length 5 [from 0 to 4]. it will form 15 X 5 array. there will be 15 off-springs
    population = rng.integers(0, len(MOVES), size=(p_size, c_length))

    # 2: Evaluate the fitness of each cell(r,c) and calculate the fitness value of each entire row
    # one row is a one off-spring
    def fitness(off_spring):

        position = center(obs).copy()
        total_score = 0

        for action in off_spring:
            new_position = position + MOVES[action]

            if is_free(obs[WALLS], *new_position):
                position = new_position

            distance = nearest_distance(position, opponents)

            total_score += sign * distance  # here sign we are taking -1 or 1 to identify evader and purusers
            # for eavders distrance should be less and for pursure value should be greater
        return total_score

    # Genetic algorithm generations
    for generation in range(generations):
        scores = [fitness(off_spring) for off_spring in population]

        new_population = []

        # Keep the best off_spring
        best_index = int(np.argmax(scores))
        new_population.append(population[best_index].copy())

        # Create the remaining off_spring
        while len(new_population) < p_size:
            # Selection of parents using tournament
            def select_parent(scores, population):

                candidates = rng.choice(p_size, size=3, replace=False)

                winner = max(candidates, key=lambda candidate_index: scores[candidate_index])

                return population[winner]

            parent1 = select_parent(scores, population)
            parent2 = select_parent(scores, population)

            # combining parent one and parent 2 gene to form a child
            if c_length > 1:
                point = int(rng.integers(1, c_length))
                child = np.concatenate((parent1[:point], parent2[point:]))
            else:
                child = parent1.copy()

            # Mutation
            for i in range(c_length):
                if rng.random() < mutation:
                    child[i] = rng.integers(0, len(MOVES))

            new_population.append(child)

        population = np.array(new_population)

    # Return first action of the best off_spring
    final_scores = [fitness(off_spring) for off_spring in population]

    best_index = int(np.argmax(final_scores))

    best_off_spring = population[best_index]

    return int(best_off_spring[0])


def genetic_evader(obs: np.ndarray, rng: np.random.Generator) -> int:
    return genetic(obs, rng, 1)


def genetic_pursuer(obs: np.ndarray, rng: np.random.Generator) -> int:
    return genetic(obs, rng, -1)


EVADER_STRATEGIES: dict[str, Strategy] = {"random": random_strategy, "greedy": greedy_evader, "genetic": genetic_evader}
PURSUER_STRATEGIES: dict[str, Strategy] = {"random": random_strategy, "greedy": greedy_pursuer, "astar": astar_pursuer, "genetic": genetic_pursuer}
