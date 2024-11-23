import os
import matplotlib.pyplot as plt

def plot_fitness(
    min_fitness,best, lb, ub, iteration, save_dir="C:/Users/vyshn/PycharmProjects/EA-XAI/GA_XAI_plots",
):
    """
    Plots fitness graphs for each run separately and also combines sequential fitness across all runs.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Save individual plots for each run
    plt.figure(figsize=(10, 6))
    for algorithm_name, fitness_values in min_fitness.items():
        plt.plot(
            list(range(1, 101)),
            fitness_values,
            label=f"Run {iteration} - Best Fitness",
            color="blue"
        )
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Value")
    plt.title(f"Run {iteration} Fitness Plot\nBounds:best={best}| LB={lb} | UB={ub}")
    plt.legend()
    plt.grid()
    plt.tight_layout()

    # Save the individual plot
    file_name = f"Run_{iteration}_fitness_plot.png"
    save_path = os.path.join(save_dir, file_name)
    try:
        plt.savefig(save_path, format="png")
        print(f"Saved plot for Run {iteration}: {save_path}")
    except Exception as e:
        print(f"Error saving plot for Run {iteration}: {e}")
    plt.close()

    # Update combined data with sequential fitness values



def plot_combined_fitness(combined_data):
    """
    Combines sequential fitness data across all runs into a single plot for comparison,
    with each run's fitness plotted in a separate color but the same generation range (1 to 100).
    """
    plt.figure(figsize=(12, 8))
    colors = ['blue', 'green', 'red', 'orange', 'purple']  # Add more colors if more runs

    # Loop over each run
    for iteration, fitness_values in combined_data.items():
        x_values = list(range(1, 101))  # x-values for generations (1 to 100 for each run)

        # Plot the fitness values for each run, but all on the same x-axis (1 to 100)
        plt.plot(x_values, fitness_values, label=f"Run {iteration}", color=colors[iteration % len(colors)])

    plt.xlabel("Generation")
    plt.ylabel("Best Fitness Value")
    plt.title("Comparison of Fitness Across Runs (1 to 100 Generations)")
    plt.legend()  # Show the legend with run labels
    plt.grid()  # Show grid for better readability
    plt.tight_layout()  # Adjust layout to avoid clipping

    # Show the combined plot in the run window
    plt.show()
