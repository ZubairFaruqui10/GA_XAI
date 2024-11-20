import matplotlib.pyplot as plt


def plot_combined_fitness(all_data,min_fitness, best):
    plt.figure(figsize=(10, 6))

    for algorithm_name in min_fitness:
        # Plot the best fitness vs generation
        plt.plot(list(range(1,101)), min_fitness[algorithm_name], label=f"{algorithm_name} - Best Fitness")

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Value")

    # Format best fitness values string for title
    best_str = " | ".join([f"{alg}: {float(val):.4f}" for alg, val in best.items()])
    plt.title(f"Fitness Comparison Across Algorithms\n{best_str}")

    plt.legend()  # Set appropriate y-axis range
    plt.grid()
    plt.tight_layout()
    plt.show()