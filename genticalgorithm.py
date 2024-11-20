import numpy as np
import csv
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from plots import plot_combined_fitness  # Ensure `plot_combined_fitness` is implemented correctly
from problem_config import get_problem  # Import the problem configuration

# Initialize configurable parameters
pop_size = 100  # Population size
n_gen = 100  # Number of generations


class LogCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data = []
        self.best = []

    def notify(self, algorithm):
        generation_number = algorithm.n_gen
        solutions = np.round(algorithm.pop.get("X"), 4)  # Round off solutions (parameters)
        fitness_values = algorithm.pop.get("F")  # Use raw fitness values without rounding
        constraints = np.round(algorithm.pop.get("G"), 4)  # Round off constraints
        self.best.append(np.min(algorithm.pop.get("F")))
        for i in range(len(solutions)):  # Loop over solutions by index
            self.data.append({
                "Generation": generation_number,
                "Parameters": solutions[i].tolist(),  # Convert rounded array to list
                "Fitness": fitness_values[i].tolist(),  # Use raw fitness values as is
                "Constraints": constraints[i].tolist(),  # Convert rounded array to list
                "Satisfied Constraints": int(np.sum(constraints[i] <= 0)),  # Count satisfied constraints
            })


# Function to perform optimization with a specified algorithm
def run_optimization(algorithm_name, algorithm, problem):
    callback = LogCallback()
    # Run the optimization
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', n_gen),
                   seed=1,
                   verbose=True,
                   callback=callback)

    return res, callback.data, callback.best


# Function to write optimization results to a CSV file
def save_to_csv(data, algorithm_name):
    fieldnames = ["Generation", "Parameters", "Fitness", "Constraints", "Satisfied Constraints"]
    file_name = f"C:/Users/vyshn/OneDrive/Desktop/optimization_log_{algorithm_name}.csv"
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data has been saved to {file_name}")


# Main function
def main():
    problem_name = "Ackley"  # Change this to "G1", "G2", etc., as needed
    problem = get_problem(problem_name)

    # Define the algorithms to test
    algorithms = {
        "GA": GA(pop_size=pop_size),
        "DE": DE(
            pop_size=pop_size,
            sampling=LHS(),
            variant="DE/rand/1/bin",
            CR=0.3,
            dither="vector",
            jitter=False
        ),
        "PSO": PSO(),
        "ES": ES(n_offsprings=200, pop_size=pop_size, rule=1.0 / 7.0)
    }

    # Store fitness data for all algorithms
    all_data = {}
    overall_best_fitness = {}
    best_fitness = {}

    # Run each algorithm and save results
    for algorithm_name, algorithm in algorithms.items():
        print(f"Running optimization with {algorithm_name} on {problem_name}...")
        res, data, best = run_optimization(algorithm_name, algorithm, problem)
        # Save results to CSV
        save_to_csv(data, algorithm_name)
        # Store data for plotting
        all_data[algorithm_name] = data
        best_fitness[algorithm_name] = best
        overall_best_fitness[algorithm_name] = res.F
        # Print best solution for verification
        print(f"Best solution found with {algorithm_name}: ", res.X)
        print(f"Function value: {res.F}")

    # Plot combined fitness results
    plot_combined_fitness(all_data, best_fitness, overall_best_fitness)


# Entry point
if __name__ == "__main__":
    main()
