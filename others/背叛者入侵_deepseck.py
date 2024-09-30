import numpy as np
import random
import re
import os
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from openai import OpenAI
import matplotlib.colors as mcolors

# Scenario for DeepSeek agent
scenario = """
You are playing a spatial Prisoner's Dilemma game on a 49x49 grid with fixed boundary conditions.
Each cell in the grid is connected to its eight neighbors, where edge cells have 5 neighbors and corner cells have 3 neighbors.
In addition, each agent also interacts with itself. If you cooperate with yourself, you receive a payoff of 1. If you defect, you receive no payoff from self-interaction.
The payoff rules with your neighbors are as follows:
- If both you and your neighbor cooperate, both receive a payoff of 1.
- If you defect and your neighbor cooperates, you receive a payoff of 1.85, and your neighbor gets 0.
- If both defect, both receive a payoff of 0.
Based on the decisions of your neighbors and your self-interaction, decide whether to cooperate or defect in each round.
Reply with 'I choose to cooperate' or 'I choose to defect.'
"""


def deepseek_agent(scenario, neighbor_decisions, neighbor_payoff, deepseek_payoff, neighbor_positions):
    """Call DeepSeek API to decide whether to cooperate or defect"""
    client = OpenAI(api_key="sk-ef13f2416c6d4c65b2ab0e1dac517a51", base_url="https://api.deepseek.com")

    # Combine the position, decision, and payoff for each neighbor into a single string
    decisions_text = ', '.join([f"Position {pos}: {'合作' if decision == 0 else '背叛'}, Payoff: {payoff}"
                                for pos, decision, payoff in
                                zip(neighbor_positions, neighbor_decisions, neighbor_payoff)])

    # Create the request to the API including all the necessary information
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": scenario},
            {"role": "user",
             "content": f"Your neighbors' choices in the last round are: {decisions_text}. "
                        f"Your neighbors' payoffs in the last round are {neighbor_payoff}. Your payoff in the last round is {deepseek_payoff}. "
                        "Please choose one of the following options: 'I choose to cooperate' or 'I choose to defect' "
                        "and explain the reasoning behind your decision."}
        ],
        stream=False
    )

    decision_text = response.choices[0].message.content
    print(f"API decision and reasoning: {decision_text}")  # Output API decision and reasoning
    return decision_text


def extract_decision(text):
    """Extract decision from the DeepSeek response."""
    pattern = r"(I choose to cooperate|I choose to defect)(.*)"
    match = re.search(pattern, text, re.IGNORECASE)  # Case insensitive matching
    if match:
        decision_text = match.group(1).lower()  # Extract choice part

        # Map choices to 0 or 1
        if "cooperate" in decision_text:
            decision = 0  # 0 means cooperation
        elif "defect" in decision_text:
            decision = 1  # 1 means defection
        else:
            decision = -1  # If no choice recognized, return -1

    return decision


# Grid setup
grid_size = 49  # 49x49 grid
iterations = 201  # Number of iterations
b = 1.85  # Temptation parameter b

# Initialize strategy: all cells are cooperators (C)
grid = np.zeros((grid_size, grid_size), dtype=int)

# Set the center position to be determined by GPT (the model will decide whether to cooperate or defect)
center = grid_size // 2


# GPT controls the center; it will decide its initial strategy
def initialize_gpt_strategy():
    # Call GPT to decide the initial strategy for the center
    initial_neighbor_decisions = [0] * 8  # Assume all neighbors initially cooperate
    initial_neighbor_payoffs = [1] * 8  # Assume initial payoffs from cooperation
    initial_payoff = 8  # Assume initial total payoff
    initial_positions = [(center - 1, center), (center + 1, center), (center, center - 1), (center, center + 1),
                         (center - 1, center - 1), (center - 1, center + 1), (center + 1, center - 1),
                         (center + 1, center + 1)]

    initial_decision_text = deepseek_agent(scenario, initial_neighbor_decisions, initial_neighbor_payoffs,
                                           initial_payoff, initial_positions)
    return extract_decision(initial_decision_text)


# Get the initial strategy for the center from GPT
grid[center, center] = initialize_gpt_strategy()

# Define neighbor directions (8-neighborhood)
neighborhood = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Define color mapping
cmap = mcolors.ListedColormap(['blue', 'yellow', 'green', 'red'])


def calculate_payoff(cell, neighbor, b):
    """Calculate the game payoff between two cells"""
    if cell == 0 and neighbor == 0:  # C vs C
        return 1, 1
    elif cell == 0 and neighbor == 1:  # C vs D
        return 0, b
    elif cell == 1 and neighbor == 0:  # D vs C
        return b, 0
    else:  # D vs D
        return 0, 0


def get_neighbors(i, j):
    """Get the neighbors of a cell"""
    neighbors = [(i, j)]  # Include self

    if i > 0:
        neighbors.append((i - 1, j))  # Up
    if i < grid_size - 1:
        neighbors.append((i + 1, j))  # Down
    if j > 0:
        neighbors.append((i, j - 1))  # Left
    if j < grid_size - 1:
        neighbors.append((i, j + 1))  # Right

    if i > 0 and j > 0:
        neighbors.append((i - 1, j - 1))  # Upper-left
    if i > 0 and j < grid_size - 1:
        neighbors.append((i - 1, j + 1))  # Upper-right
    if i < grid_size - 1 and j > 0:
        neighbors.append((i + 1, j - 1))  # Lower-left
    if i < grid_size - 1 and j < grid_size - 1:
        neighbors.append((i + 1, j + 1))  # Lower-right

    return neighbors


def update_strategy(grid, b):
    """Update the strategy of each cell, with GPT controlling the center"""
    new_grid = grid.copy()

    for i in range(grid_size):
        for j in range(grid_size):
            total_payoff = 0
            neighbors = get_neighbors(i, j)

            for ni, nj in neighbors:
                payoff_cell, _ = calculate_payoff(grid[i, j], grid[ni, nj], b)
                total_payoff += payoff_cell  # Calculate the total payoff of the cell

            # GPT controls the center, it makes its own decision
            if i == center and j == center:
                neighbor_decisions = [grid[ni, nj] for ni, nj in neighbors]  # Current neighbors' choices
                neighbor_positions = [(ni, nj) for ni, nj in neighbors]  # Current neighbors' positions
                deepseek_payoff = total_payoff

                # Calculate the payoffs for the neighbors of the center
                neighbor_payoff = []
                for ni, nj in neighbors:
                    payoff_sum = 0
                    for nni, nnj in get_neighbors(ni, nj):
                        payoff_neighbor, _ = calculate_payoff(grid[ni, nj], grid[nni, nnj], b)
                        payoff_sum += payoff_neighbor
                    neighbor_payoff.append(payoff_sum)

                # Call GPT to make a decision based on the neighbor's choices and payoffs
                decision_text = deepseek_agent(scenario, neighbor_decisions, neighbor_payoff, deepseek_payoff,
                                               neighbor_positions)
                deepseek_decision = extract_decision(decision_text)
                new_grid[center, center] = deepseek_decision  # Update the strategy of the center

            else:
                best_neighbor_payoff = total_payoff
                best_neighbor_strategy = grid[i, j]

                # Check neighbors' payoffs to imitate the best strategy
                for ni, nj in neighbors:
                    neighbor_payoff = 0
                    for nni, nnj in get_neighbors(ni, nj):
                        payoff_neighbor, _ = calculate_payoff(grid[ni, nj], grid[nni, nnj], b)
                        neighbor_payoff += payoff_neighbor

                    # If the neighbor's payoff is higher, imitate the neighbor's strategy
                    if neighbor_payoff > best_neighbor_payoff:
                        best_neighbor_payoff = neighbor_payoff
                        best_neighbor_strategy = grid[ni, nj]

                new_grid[i, j] = best_neighbor_strategy  # Update strategy for regular cells

    return new_grid


def visualize_grid(grid, previous_grid, iteration, ax):
    """Visualize the grid on a subplot"""
    color_grid = np.zeros((grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            if grid[i, j] == 0:  # Currently C
                color_grid[i, j] = 0 if previous_grid[i, j] == 0 else 2  # Blue (C->C) or Green (D->C)
            else:  # Currently D
                color_grid[i, j] = 1 if previous_grid[i, j] == 0 else 3  # Yellow (C->D) or Red (D->D)

    ax.imshow(color_grid, cmap=cmap, vmin=0, vmax=3)
    ax.set_title(f"Iteration {iteration}")
    ax.axis('off')


# Set new grid equal to original grid
previous_grid = grid.copy()

# Define the output directory
output_dir = os.path.join(os.getcwd(), "grid_iterations_output")

# Check and create the directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate and visualize every 10 iterations
for fig_num in range(1, 21):
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    subplot_idx = 0

    for t in range((fig_num - 1) * 10 + 1, fig_num * 10 + 1):
        if t >= iterations:
            break

        grid = update_strategy(previous_grid, b)

        # Visualize each iteration
        row, col = subplot_idx // 5, subplot_idx % 5
        visualize_grid(grid, previous_grid, t, axes[row, col])
        subplot_idx += 1
        previous_grid = grid.copy()

    plt.ioff()
    plt.tight_layout()

    # Save the figure to the output directory
    save_path = os.path.join(output_dir, f"grid_iterations_{(fig_num - 1) * 10 + 1}_to_{fig_num * 10}.png")
    plt.savefig(save_path)
    plt.close(fig)

    print(f"Saved figure to {save_path}")