import numpy as np
import csv
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.operators.sampling.lhs import LHS
from pymoo.optimize import minimize
from pymoo.core.callback import Callback
from plots import plot_combined_fitness, plot_fitness  # Ensure `plot_combined_fitness` is implemented correctly
from problem_config import get_problem  # Import the problem configuration
from XAI import processData

# Initialize configurable parameters
pop_size = 30  # Population size
n_gen = 30  # Number of generations


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
                "Solution": solutions[i].tolist(),  # Convert rounded array to list
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
def save(data, algorithm_name):
    fieldnames = ["Generation", "Solution", "Fitness", "Constraints", "Satisfied Constraints"]
    file_name = f"C:/Users/vyshn/OneDrive/Desktop/optimization_log_{algorithm_name}.csv"
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data has been saved to {file_name}")


def save_to_csv(data, algorithm_name, iteration):
    fieldnames = ["Generation", "Solution", "Fitness", "Constraints", "Satisfied Constraints"]
    file_name_append = f"C:/Users/vyshn/OneDrive/Desktop/append_log_{algorithm_name}_.csv"

    # Check if the file already exists and is non-empty
    try:
        with open(file_name_append, mode="r") as file:
            file_is_empty = file.readline() == ''
    except FileNotFoundError:
        file_is_empty = True  # File doesn't exist yet

    # Open the file in append mode and write data
    with open(file_name_append, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        if file_is_empty:  # Write header only if the file is empty
            writer.writeheader()
        writer.writerows(data)

    print(f"Data has been appended to {file_name_append}")

    file_name = f"C:/Users/vyshn/OneDrive/Desktop/optimization_log_{algorithm_name}_{iteration}.csv"
    with open(file_name, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames)
        writer.writeheader()
        writer.writerows(data)
    print(f"Data has been saved to {file_name}")



def main():
    problem_name = "Ackley"  # Change this to "G1", "G2", etc., as needed
    # Initial problem setup
    lb = mainlb = np.array([0]*10, dtype=float)
    ub = mainub =  np.array([1]*10, dtype=float)  # Start with default bounds for the problem
    max_iterations = 5  # Number of times to run the process
    csv_path =  "C:\\Users\\vyshn\\OneDrive\\Desktop\\append_log_GA_.csv"

    combined_best_fitness = {}
    for iteration in range(max_iterations):
        print(f"Starting iteration {iteration + 1}/{max_iterations}...")
        # Get problem instance with updated bounds
        problem = get_problem(problem_name, lb=lb, ub=ub)
        # Define the algorithms to test
        algorithms = {
            "GA": GA(pop_size=pop_size),
            # Uncomment other algorithms as needed
            # "DE": DE(
            #     pop_size=pop_size,
            #     sampling=LHS(),
            #     variant="DE/rand/1/bin",
            #     CR=0.3,
            #     dither="vector",
            #     jitter=False
            # ),
            # "PSO": PSO(),
            # "ES": ES(n_offsprings=200, pop_size=pop_size, rule=1.0 / 7.0)
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
            save_to_csv(data, algorithm_name, iteration)

            # Store data for plotting
            all_data[algorithm_name] = data
            best_fitness[algorithm_name] = best
            overall_best_fitness[algorithm_name] = res.F
            combined_best_fitness[iteration] = best
            # Print best solution for verification
            print(f"Best solution found with {algorithm_name}: ", res.X)
            print(f"Function value: {res.F}")

        # Plot combined fitness results (optional after each iteration)
        plot_fitness(best_fitness, overall_best_fitness, lb, ub, iteration, n_gen)

        # Use the generated CSV to get recommendations for updating bounds
        print(f"Processing CSV data from {csv_path} for iteration {iteration + 1}...")
        topKFeatureDictionary = processData(csv_path)

        # Update bounds based on the dictionary
        lb, ub = update_bounds(topKFeatureDictionary, lb, ub, mainlb, mainub )

        print(f"Updated bounds for next iteration: lb = {lb}, ub = {ub}")
    plot_combined_fitness(combined_best_fitness, n_gen)

def update_bounds(feature_dict, lb, ub, main_lb, main_ub):
    for feature, value in feature_dict.items():
        if value < 0:  # Negative value: Increase lower bound by |value|%
            increment = (abs(value)/10) * lb[feature]
            new_lb = lb[feature] + increment
            # Check if the new lb exceeds the main upper bound
            if new_lb <= main_ub[feature]:
                lb[feature] = new_lb
            # Otherwise, keep the old value
        elif value > 0:  # Positive value: Decrease upper bound by |value|%
            decrement = (abs(value)/10) * ub[feature]
            new_ub = ub[feature] - decrement
            # Check if the new ub goes below the main lower bound
            if new_ub >= main_lb[feature] and new_ub <= main_ub[feature]:
                ub[feature] = new_ub
            # Otherwise, keep the old value

    return lb, ub


# Entry point
if __name__ == "__main__":
    main()
