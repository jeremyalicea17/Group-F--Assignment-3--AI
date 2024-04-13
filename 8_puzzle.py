import random
import search
import math
import matplotlib.pyplot as plt

class EightPuzzle(search.Problem):
    def __init__(self, initial, goal=(1, 2, 3, 4, 5, 6, 7, 8, 0)):
        super().__init__(initial, goal)

    def actions(self, state):
        zero_pos = state.index(0)
        possible_actions = []
        if zero_pos % 3 != 0: possible_actions.append('left')
        if zero_pos % 3 != 2: possible_actions.append('right')
        if zero_pos // 3 != 0: possible_actions.append('up')
        if zero_pos // 3 != 2: possible_actions.append('down')
        return possible_actions

    def result(self, state, action):
        zero_pos = state.index(0)
        new_pos = zero_pos
        if action == 'left': new_pos -= 1
        if action == 'right': new_pos += 1
        if action == 'up': new_pos -= 3
        if action == 'down': new_pos += 3
        new_state = list(state)
        new_state[zero_pos], new_state[new_pos] = new_state[new_pos], new_state[zero_pos]
        return tuple(new_state)

    def value(self, state):
        # A simple heuristic: the number of misplaced tiles
        return -sum(p != g for p, g in zip(state, self.goal) if p != 0)


class EightQueens(search.Problem):
    def __init__(self, initial=None, goal=None):
        if initial is None:
            initial = tuple(random.sample(range(8), 8))  # A random arrangement of queens
        super().__init__(initial, goal)

    def actions(self, state):
        actions = []
        for i in range(8):
            for j in range(8):
                if state[i] != j:
                    actions.append((i, j))
        return actions

    def result(self, state, action):
        col, newRow = action
        newState = list(state)
        newState[col] = newRow
        return tuple(newState)

    def value(self, state):
        # Counts the number of pairs of queens that are attacking each other
        num_attacking_pairs = 0
        for i in range(len(state)):
            for j in range(i + 1, len(state)):
                if state[i] == state[j] or abs(state[i] - state[j]) == j - i:
                    num_attacking_pairs += 1
        return -num_attacking_pairs


# The definitions for hill_climbing, hill_climbing_steepest_ascent, hill_climbing_first_choice,hill_climbing random restart
def hill_climbing(problem, max_steps=100):
    """Simple hill climbing algorithm with max steps limit."""
    print("Running hill climbing")
    current = problem.initial
    step = 0
    while step < max_steps:
        step += 1
        neighbors = problem.actions(current)
        if not neighbors:
            break
        neighbor = max(neighbors, key=lambda action: problem.value(problem.result(current, action)))
        if problem.value(neighbor) <= problem.value(current):
            break
        current = problem.result(current, neighbor)
    return current, step


def hill_climbing_steepest_ascent(problem, max_steps=100):
    """Steepest ascent version of hill climbing with step count."""
    current = problem.initial
    step = 0
    while step < max_steps:
        step += 1
        next_move = max([(problem.result(current, action), action) for action in problem.actions(current)],
                        key=lambda x: problem.value(x[0]), default=(current, None))
        if problem.value(next_move[0]) <= problem.value(current):
            return current, step  # return also the step count
        current = next_move[0]
    return current, step


def hill_climbing_first_choice(problem, max_steps=100):
    """First-choice version of hill climbing that also counts steps."""
    current = problem.initial
    for step in range(1, max_steps + 1):
        neighbors = problem.actions(current)
        random.shuffle(neighbors)
        for action in neighbors:
            next_state = problem.result(current, action)
            if problem.value(next_state) > problem.value(current):
                current = next_state
                break
        else:  # No better neighbor found
            return current, step
    return current, max_steps

def hill_climbing_random_restart(problem, num_restarts=10):
    """Hill climbing with random restarts."""
    best = problem.initial
    best_value = problem.value(best)
    step = 0
    for _ in range(num_restarts):
        current = problem.initial
        while True:
            step += 1
            next_moves = [problem.result(current, action) for action in problem.actions(current)]
            next_move = max(next_moves, key=problem.value, default=current)
            if problem.value(next_move) <= problem.value(current):
                break
            current = next_move
        if problem.value(current) > best_value:
            best, best_value = current, problem.value(current)
    return best, step  # Ensure to return the total steps taken


def exp_schedule(k=20, lam=0.005, limit=1000):
    """Return a function that implements an exponential decay schedule."""
    return lambda t: (k * math.exp(-lam * t) if t < limit else 0)

def simulated_annealing(problem, max_steps=100, schedule=exp_schedule()):
    """Perform the simulated annealing algorithm with maximum steps."""
    current = problem.initial
    for step in range(max_steps):
        T = schedule(step)
        if T == 0:
            return current, step
        neighbors = problem.actions(current)
        if not neighbors:
            return current, step
        next_choice = random.choice(neighbors)
        next_state = problem.result(current, next_choice)
        delta_e = problem.value(next_state) - problem.value(current)
        if delta_e > 0 or random.uniform(0, 1) < math.exp(delta_e / T):
            current = next_state
    return current, max_steps

def run_problem_with_step_count(problem, algorithm, max_steps=100):
    print(f"Running {algorithm.__name__} with max steps {max_steps}...")
    result = algorithm(problem, max_steps)
    print("Algorithm returned:", result)  # This will show what is being returned
    final_state, steps = result  # This is where it fails if the tuple is not correct
    print(f"Final state: {final_state} with value: {problem.value(final_state)}")
    if problem.goal_test(final_state):
        print(f"Goal reached: {final_state} with value: {problem.value(final_state)}")
    else:
        print(f"Goal not reached after {steps} steps.")
    return final_state, steps


def compare_algorithms_plot(algorithms, problem_instances, max_steps=100):
    for algorithm in algorithms:
        step_counts = []
        for problem in problem_instances:
            final_state, steps = run_problem_with_step_count(problem, algorithm, max_steps)  # Expecting two values here
            step_counts.append(steps)
        if len(problem_instances) == 1:
            plt.scatter([1], step_counts, label=algorithm.__name__)  # Plot a single point
        else:
            plt.plot(step_counts, label=algorithm.__name__, marker='o')  # Plot a line with markers
    plt.xlabel('Problem Instance')
    plt.ylabel('Steps to Solution')
    plt.title('Algorithm Comparison')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    algorithms = [hill_climbing,hill_climbing_steepest_ascent, hill_climbing_first_choice, hill_climbing_random_restart, simulated_annealing]
    eight_puzzle_instances = [EightPuzzle((2, 8, 3, 1, 6, 4, 7, 0, 5))]  # Simplified for testing
    eight_queens_instances = [EightQueens()]  # Simplified for testing

    # Run tests on both problem types
    print("Comparing on 8-Puzzle Problems...")
    compare_algorithms_plot(algorithms, eight_puzzle_instances, 100)  # Ensure max_steps is adequate

    print("\nComparing on 8-Queens Problems...")
    compare_algorithms_plot(algorithms, eight_queens_instances, 100)  # Ensure max_steps is adequate