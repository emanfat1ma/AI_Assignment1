import heapq
import time
import math
import random
import sys

MAX_COST = sys.maxsize # Used for unreachable paths
CITY_GRID = [
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
    ['#', 'S', '1', '1', '#', '1', '1', '1', 'A', '#'],
    ['#', '1', '#', '2', '#', '2', '#', '#', '1', '#'],
    ['#', '1', '#', '2', '2', '3', '3', '#', '1', '#'],
    ['#', '1', '1', '2', '#', '#', '1', '#', '1', '#'],
    ['#', '#', '#', '3', '3', '3', '#', '2', '1', '#'],
    ['#', '1', '#', '3', '#', '#', '3', '3', '1', '#'],
    ['#', '1', '1', '2', '1', '#', '#', '#', '1', '#'],
    ['#', 'C', '#', '1', '1', '2', '2', '2', 'B', '#'],
    ['#', '#', '#', '#', '#', '#', '#', '#', '#', '#'],
]
START_NODE = (1, 1)
GOAL_A, GOAL_B, GOAL_C = (1, 8), (8, 8), (8, 1)
ALL_GOALS = [GOAL_A, GOAL_B, GOAL_C]

# --- Core Utilities ---

def get_cost(node, grid):
    """Returns the travel cost (g) for entering a cell."""
    r, c = node
    char = grid[r][c]
    if char == '#':
        return MAX_COST
    try:
        # Handles costs '1', '2', '3'
        return int(char)
    except ValueError:
        # Handles 'S', 'A', 'B', 'C' as cost 1 upon entry
        return 1

def neighbors(node, grid):
    """Returns all valid (r, c) neighbors."""
    r, c = node
    rows, cols = len(grid), len(grid[0])
    valid_neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]: 
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
            valid_neighbors.append((nr, nc))
    return valid_neighbors

def reconstruct_path(came_from, start, goal):
    """Reconstructs the path from start to goal."""
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from.get(current)
        if current is None:
            return None 
    path.append(start)
    return path[::-1]

def calculate_path_cost(path, grid):
    """Calculates the total path cost on the weighted grid."""
    if not path or len(path) < 2: return 0
    # Cost is sum of costs of entering each node after the start
    total_cost = sum(get_cost(node, grid) for node in path[1:])
    return total_cost

# --- Dynamic Grid Feature ---

def create_dynamic_grid(original_grid, dynamic_prob=0.10, cost_factor=10):
    """
    Called REPEATEDLY to generate a new, unique snapshot of costs (traffic, closures)
    in the environment before each planning cycle. (No code change needed here,
    but its *usage* changes.)
    """
    dynamic_grid = [list(row) for row in original_grid] # Deep copy
    rows, cols = len(dynamic_grid), len(dynamic_grid[0])
    
    # Randomly inflate costs for non-obstacle, non-goal cells
    for r in range(rows):
        for c in range(cols):
            cell = dynamic_grid[r][c]
            if cell not in ['#', 'S', 'A', 'B', 'C'] and random.random() < dynamic_prob:
                # Inflate the cost of the path by a factor
                try:
                    original_cost = int(cell)
                    dynamic_grid[r][c] = str(original_cost * cost_factor)
                except ValueError:
                    # Should only happen if cost is '1' but still good practice
                    dynamic_grid[r][c] = str(cost_factor)
    return dynamic_grid

# --- Heuristics (Task 3) ---
def manhattan(a, b):
    """Admissible Heuristic 1: Manhattan Distance (L1 norm)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def zero_heuristic(a, b):
    """Admissible Heuristic 2: Zero Heuristic (Dijkstra's baseline)."""
    return 0

def inflated_heuristic(a, b):
    """Non-Admissible Heuristic: Manhattan Distance multiplied by 2."""
    return 2 * manhattan(a, b)

# --- Complex Multi-Goal Heuristic (For Task 1) ---
def prim_mst(num_goals, goal_to_goal_cost_matrix):
    """Calculates MST cost using pre-calculated true shortest path costs."""
    if num_goals <= 1: return 0
    mst_cost = 0
    visited = {0} 
    min_heap = [] 

    for j in range(1, num_goals):
        cost = goal_to_goal_cost_matrix[0][j]
        heapq.heappush(min_heap, (cost, j))

    while min_heap and len(visited) < num_goals:
        cost, node_index = heapq.heappop(min_heap)
        if node_index in visited: continue
        visited.add(node_index)
        mst_cost += cost

        for neighbor_index in range(num_goals):
            if neighbor_index not in visited:
                weight = goal_to_goal_cost_matrix[node_index][neighbor_index]
                heapq.heappush(min_heap, (weight, neighbor_index))
    
    return mst_cost

def a_star_single_target(start, goal, grid, heuristic=manhattan):
    """Helper A* to find true shortest path cost on the weighted grid."""
    if start == goal: return 0, 0, 1 # Cost, Time, Nodes

    frontier = [(0, 0, start)] # (f_cost, g_cost, node)
    cost_so_far = {start: 0}
    nodes_expanded = 0    
    start_time = time.time()

    while frontier:
        f_cost, g_cost, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            end_time = time.time()
            return g_cost, end_time - start_time, nodes_expanded

        for next_node in neighbors(current, grid):
            move_cost = get_cost(next_node, grid)
            new_cost = g_cost + move_cost
            
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                h_cost = heuristic(next_node, goal) 
                priority = new_cost + h_cost
                heapq.heappush(frontier, (priority, new_cost, next_node))
                
    return MAX_COST, time.time() - start_time, nodes_expanded

# Pre-calculate the true costs between all goals for an ADMISSIBLE MST heuristic
def pre_calculate_goal_matrix(goals, grid):
    N = len(goals)
    matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(i + 1, N):
            cost, _, _ = a_star_single_target(goals[i], goals[j], grid)
            matrix[i][j] = cost
            matrix[j][i] = cost
    return matrix

# Run pre-calculation for the static grid
GOAL_COST_MATRIX = pre_calculate_goal_matrix(ALL_GOALS, CITY_GRID)

def multi_goal_heuristic(current, all_goals, goal_matrix):
    """
    Admissible Multi-Goal Heuristic: TrueCost(n, closest goal) + MST(all goals).
    """
    if not all_goals: return 0

    # 1. True Cost from current node to the closest goal (using A* helper)
    # NOTE: This makes the heuristic EXPENSIVE but correct (admissible)
    dist_to_closest = min(a_star_single_target(current, goal, CITY_GRID)[0] for goal in all_goals)
    
    # 2. MST Cost for all goals (using the pre-calculated matrix)
    mst_cost = prim_mst(len(all_goals), goal_matrix)
    
    return dist_to_closest + mst_cost

# --- 4. Search Algorithm Implementations ---

# --- Task 1: Multi-Goal Greedy Best-First Search ---
def multi_goal_gbfs(start, all_goals, grid, goal_matrix):
    """
    Implements GBFS prioritizing nodes by the multi_goal_heuristic (f(n) = h(n)).
    Finds the first leg of the multi-goal tour.
    """
    frontier = []
    initial_priority = multi_goal_heuristic(start, all_goals, goal_matrix)
    heapq.heappush(frontier, (initial_priority, start))
    
    came_from = {start: None}
    explored = {start}
    nodes_expanded = 0
    
    print("\n--- Task 1: Multi-Goal Greedy Best-First Search (GBFS) ---")
    
    iteration = 0
    start_time = time.time()
    
    while frontier:
        iteration += 1
        
        # POP: Node with the lowest heuristic value (best guess for total cost)
        priority, current = heapq.heappop(frontier)
        nodes_expanded += 1
        
        # --- Frontier Visualization (Task 1 requirement) ---
        frontier_display = sorted([(round(p, 2), n) for p, n in frontier])[:3]
        print(f"Iter {iteration}: Popped {current}. Priority (h)={round(priority, 2)}.")
        print(f"  Frontier (Top 3 scheduled, sorted by h): {frontier_display}")
        
        if current in all_goals:
            end_time = time.time()
            path = reconstruct_path(came_from, start, current)
            cost = calculate_path_cost(path, grid)
            return path, cost, end_time - start_time, nodes_expanded

        # Expand Neighbors
        for next_node in neighbors(current, grid):
            if next_node not in explored:
                # GBFS uses ONLY the heuristic value for priority
                next_priority = multi_goal_heuristic(next_node, all_goals, goal_matrix)
                
                # Push to frontier
                heapq.heappush(frontier, (next_priority, next_node))
                came_from[next_node] = current
                explored.add(next_node)
    
    return None, 0, time.time() - start_time, nodes_expanded

def find_full_multi_goal_tour(start, all_goals, grid):
    """
    Orchestrates the full tour by sequentially searching from the current 
    position to the closest unvisited goal using A* (optimal search).
    """
    current_position = start
    unvisited_goals = set(all_goals)
    total_path = [start]
    total_cost = 0
    total_nodes_expanded = 0
    
    start_time = time.time()
    
    # Loop until all goals are visited
    while unvisited_goals:
        
        # 1. Greedy Selection: Find the path cost to all remaining goals
        min_cost_to_next = MAX_COST
        next_goal = None

        for goal in unvisited_goals:
            # Use A* to find the true cost to the current candidate goal
            cost_to_goal, _, _ = a_star_single_target(current_position, goal, grid, heuristic=manhattan)
            
            if cost_to_goal < min_cost_to_next:
                min_cost_to_next = cost_to_goal
                next_goal = goal
        
        if next_goal is None:
            print("\nError: Could not find a path to the remaining goals. Tour stopped.")
            break

        # 2. Search Leg: Find the full path to the chosen next goal
        # Use Weighted A* with alpha=1.0 (Standard A*) for optimality in each segment
        path_segment, segment_cost, segment_time, segment_nodes = weighted_a_star(
            current_position, next_goal, grid, manhattan, alpha=1.0
        )
        
        # 3. Update Totals
        total_cost += segment_cost
        total_nodes_expanded += segment_nodes
        
        # Append the new path segment (excluding the starting node of this segment)
        total_path.extend(path_segment[1:])
        
        # Update state for the next loop
        current_position = next_goal
        unvisited_goals.remove(next_goal)
        print(f"Segment completed: {total_path[-2]} -> {next_goal}. Cost: {segment_cost}")

    end_time = time.time()
    total_time = end_time - start_time
    
    return total_path, total_cost, total_time, total_nodes_expanded

# --- Task 2.1: Weighted A* ---
def weighted_a_star(start, goal, grid, heuristic, alpha=1.0):
    """Implements Weighted A* (WA*): f(n) = g(n) + alpha * h(n)."""
    frontier = [(0 + alpha * heuristic(start, goal), 0, start)] # (f_cost, g_cost, node)
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_expanded = 0
    
    start_time = time.time()
    while frontier:
        f_cost, g_cost, current = heapq.heappop(frontier)
        nodes_expanded += 1

        if current == goal:
            end_time = time.time()
            path = reconstruct_path(came_from, start, current)
            return path, g_cost, end_time - start_time, nodes_expanded

        for next_node in neighbors(current, grid):
            move_cost = get_cost(next_node, grid)
            new_cost = g_cost + move_cost
            
            if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                cost_so_far[next_node] = new_cost
                h_cost = heuristic(next_node, goal) 
                priority = new_cost + alpha * h_cost
                heapq.heappush(frontier, (priority, new_cost, next_node))
                came_from[next_node] = current
    return None, 0, time.time() - start_time, nodes_expanded

# --- Task 2.2: Bidirectional A* ---
def bidirectional_a_star(start, goal, grid, heuristic=manhattan):
    """Implements Bidirectional A* (BA*). Stops when the two search frontiers meet."""
    if start == goal: return [start], 0, 0, 1
    
    # Forward Search (s_): start to goal
    frontier_s = [(heuristic(start, goal), 0, start)] # (f, g, node)
    cost_so_far_s = {start: 0}
    came_from_s = {start: None}
    nodes_expanded = 0

    # Backward Search (g_): goal to start
    frontier_g = [(heuristic(goal, start), 0, goal)]
    cost_so_far_g = {goal: 0}
    came_from_g = {goal: None}
    
    start_time = time.time()
    meeting_node = None
    min_total_cost = MAX_COST

    def expand_frontier(forward):
        nonlocal nodes_expanded, meeting_node, min_total_cost
        
        current_f = frontier_s if forward else frontier_g
        current_g = cost_so_far_s if forward else cost_so_far_g
        current_c = came_from_s if forward else came_from_g
        other_g = cost_so_far_g if forward else cost_so_far_s
        
        if not current_f: return

        f_cost, g_cost, current = heapq.heappop(current_f)
        nodes_expanded += 1

        # Check for meeting condition
        if current in other_g:
            # Candidate path found. Check if it's the best so far.
            total_cost = g_cost + other_g[current]
            if total_cost < min_total_cost:
                min_total_cost = total_cost
                meeting_node = current
            # Note: BA* should continue searching until the sum of the best f-costs
            # in the two frontiers exceeds min_total_cost, but for simplicity, 
            # we stop when a meeting point is found.

        for next_node in neighbors(current, grid):
            move_cost = get_cost(next_node, grid)
            new_cost = g_cost + move_cost
            
            if new_cost < current_g.get(next_node, MAX_COST):
                current_g[next_node] = new_cost
                current_c[next_node] = current
                
                # Heuristic is always towards the opposing start/goal
                h_cost = heuristic(next_node, goal if forward else start) 
                priority = new_cost + h_cost
                
                heapq.heappush(current_f, (priority, new_cost, next_node))
                
                # Re-check meeting point after adding new node
                if next_node in other_g:
                    total_cost = current_g[next_node] + other_g[next_node]
                    if total_cost < min_total_cost:
                        min_total_cost = total_cost
                        meeting_node = next_node

    while frontier_s and frontier_g:
        # Simple policy: alternate expansion
        if len(frontier_s) <= len(frontier_g):
            expand_frontier(True) # Expand Forward
        else:
            expand_frontier(False) # Expand Backward
        
        if meeting_node is not None and min_total_cost < frontier_s[0][0] + frontier_g[0][0]:
            # Optimal path found (meeting node is the best connection)
            break
            
    end_time = time.time()
    
    if meeting_node:
        path_s = reconstruct_path(came_from_s, start, meeting_node)[:-1] # Exclude meeting node
        path_g = reconstruct_path(came_from_g, goal, meeting_node)[::-1]
        full_path = path_s + path_g
        return full_path, min_total_cost, end_time - start_time, nodes_expanded
    
    return None, 0, end_time - start_time, nodes_expanded

# --- 5. Execution and Experimental Evaluation ---

def run_experiment(algorithm, start, goal, grid, heuristic, alpha=1.0, title=""):
    """
    Runs a single search experiment, extracts performance metrics, 
    and returns them in a dictionary.
    """
    path, cost, time_taken, nodes = None, 0, 0, 0
    # Special handling for WA*
    if algorithm == weighted_a_star:
        path, cost, time_taken, nodes = algorithm(start, goal, grid, heuristic, alpha)
    # Special handling for BA*
    elif algorithm == bidirectional_a_star:
        path, cost, time_taken, nodes = algorithm(start, goal, grid, heuristic)
    # Default (Standard A* via WA* with alpha=1)
    status = "SUCCESS" if path else "FAILURE"
    
    print(f"\n{title} ({heuristic.__name__}, Alpha={alpha})")
    print(f"  Status: {status}")
    print(f"  Cost: {round(cost, 2)}")
    print(f"  Nodes Expanded: {nodes}")
    print(f"  Time (s): {round(time_taken, 5)}")
    return {'cost': cost, 'nodes': nodes, 'time': time_taken}

def receding_horizon_search(start, final_goal, original_grid, horizon_steps=5):
    """
    Simulates a robot moving toward a final_goal using a Receding Horizon approach.
    The robot plans for a small 'horizon' (e.g., 5 steps) and executes only the first step.
    """
    current_position = start
    total_path = [start]
    total_cost = 0
    while current_position != final_goal:
        current_grid = create_dynamic_grid(original_grid, dynamic_prob=0.15, cost_factor=10) # A.Check and Update the Grid
        planned_path, cost_to_goal, _, _ = weighted_a_star(                                  # B.Short Horizon Searc
            current_position, final_goal, current_grid, manhattan, alpha=1.0 # Use standard A*
        )
        
        if not planned_path:
            print("Path planning failed at current position. Search terminated.")
            break
        if len(planned_path) > 1:                                                            # Take One Step
            next_step = planned_path[1] # The next node after current_position
            step_cost = get_cost(next_step, current_grid)
        else:
            next_step = final_goal
            step_cost = 0 # Cost to 'reach' goal is already in path cost calculation

        total_cost += step_cost           # Update state
        current_position = next_step
        total_path.append(current_position)
        print(f"Executed step to {current_position}. Total Cost: {total_cost}")
    return total_path, total_cost

import matplotlib.pyplot as plt
import numpy as np
def visualize_grid_path(grid, path):
    rows, cols = len(grid), len(grid[0])
    image = np.zeros((rows, cols))

    # Define the highest possible cost to normalize the grayscale range (3 * 10 = 30)
    MAX_INFLATED_COST = 30 
    GOAL_MARKERS = ('A', 'B', 'C')

    for r in range(rows):
        for c in range(cols):
            cell_content = grid[r][c]
            
            if cell_content == '#':
                image[r, c] = 0.0  # Wall (Black)
            elif cell_content == 'S':
                image[r, c] = 0.5  # Start (Medium Gray)
            elif cell_content in GOAL_MARKERS:
                image[r, c] = 0.7  # Goal (Light Gray)
            else:
                # --- FIX: Handle ALL numeric costs ('1', '3', '10', '30', etc.) ---
                try:
                    cost = int(cell_content)
                    # Normalize cost: 1.0 (low cost) to 0.0 (high cost/dark shade)
                    normalized_cost = cost / MAX_INFLATED_COST
                    image[r, c] = 1.0 - normalized_cost
                except ValueError:
                    # Fallback for any unexpected non-numeric cell
                    image[r, c] = 1.0  # Default to White

    # Draw the path over the grid
    for r, c in path:
        cell_content = grid[r][c]
        if cell_content not in ('S') and cell_content not in GOAL_MARKERS:
            image[r, c] = 0.35  # Path (Dark Gray line)

    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.title('Pathfinding Result on Dynamic Grid')
    plt.xticks([]); plt.yticks([])
    plt.grid(False)
    plt.show()

# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Task 1 Execution ---
    path_gbfs, cost_gbfs, time_gbfs, nodes_gbfs = multi_goal_gbfs(START_NODE, ALL_GOALS, CITY_GRID, GOAL_COST_MATRIX)
    print(f"\n--- Task 1 Summary ---")
    print(f"First Leg Path Cost: {cost_gbfs}, Nodes Expanded: {nodes_gbfs}, Time: {round(time_gbfs, 5)}s")
    # 2. Run the full iterative tour
    path_full_tour, cost_full_tour, time_full_tour, nodes_full_tour = find_full_multi_goal_tour(
        START_NODE, ALL_GOALS, CITY_GRID
    )
    print(f"Total Path Cost: {cost_full_tour}")
    print(f"Total Nodes Expanded: {nodes_full_tour}")
    print(f"Total Time: {round(time_full_tour, 5)}s")
    if path_full_tour:
        print("\nDisplaying Visualization 1: Full Optimal Multi-Goal Tour")
        visualize_grid_path(CITY_GRID, path_full_tour)
    
    # --- Task 2 & 3 Execution: Performance Comparison (S to A) ---
    
    print("\n" + "="*80)
    print("TASK 2 & 3: Weighted A* and Bidirectional A* Performance Comparison (S to A)")
    print("="*80)
    
    metrics = {}
    
    # --- Static Grid Experiments ---
    print("\n--- A. Static Grid Performance (Goal: A) ---")
    static_grid = CITY_GRID
    
    # Standard A* (alpha=1.0) with Manhattan
    path_a_star, _, _, _ = weighted_a_star(START_NODE, GOAL_A, static_grid, manhattan, 1.0)
    metrics['A*_Manhattan'] = run_experiment(weighted_a_star, START_NODE, GOAL_A, static_grid, manhattan, 1.0, "Standard A*")
    if path_a_star:
        print("\nDisplaying Visualization 2: Standard A* on Static Grid (S to A)")
        # Pass the static grid and the optimal path
        visualize_grid_path(static_grid, path_a_star)
    
    # Weighted A* (alpha=1.5) with Manhattan
    metrics['WA*_1.5'] = run_experiment(weighted_a_star, START_NODE, GOAL_A, static_grid, manhattan, 1.5, "Weighted A* (alpha=1.5)")
    # Weighted A* (alpha=2.0) with Manhattan
    metrics['WA*_2.0'] = run_experiment(weighted_a_star, START_NODE, GOAL_A, static_grid, manhattan, 2.0, "Weighted A* (alpha=2.0)")

    # Bidirectional A* with Manhattan
    metrics['BA*'] = run_experiment(bidirectional_a_star, START_NODE, GOAL_A, static_grid, manhattan, 1.0, "Bidirectional A*")

    # Admissible Heuristic (Zero)
    metrics['A*_Zero'] = run_experiment(weighted_a_star, START_NODE, GOAL_A, static_grid, zero_heuristic, 1.0, "Standard A* (Zero H)")
    
    # Non-Admissible Heuristic (Inflated)
    metrics['A*_Inflated'] =run_experiment(weighted_a_star,START_NODE,GOAL_A,static_grid,inflated_heuristic,1.0,"Standard A* (Inflated H)")

    
    # --- Dynamic Grid Experiments (Task 2 Requirement) ---
    print("\n--- B. Dynamic Grid Performance (Goal: A) ---")
    dynamic_grid = create_dynamic_grid(CITY_GRID, dynamic_prob=0.15, cost_factor=10)
    metrics['A*_Dyn']=run_experiment(weighted_a_star, START_NODE, GOAL_A, dynamic_grid, manhattan, 1.0, "Standard A* (Dynamic)")
    metrics['A*_Inflated']=run_experiment(weighted_a_star,START_NODE,GOAL_A,dynamic_grid,inflated_heuristic,1.0,"Standard A* (Inflated H, Dynamic)")
    
    # Weighted A* (alpha=4.0) on Dynamic Grid
    path_wa_dyn, _, _, _ = weighted_a_star(START_NODE, GOAL_A, dynamic_grid, manhattan, 4.0)
    metrics['WA*_Dyn'] = run_experiment(weighted_a_star, START_NODE, GOAL_A, dynamic_grid, manhattan, 4.0, "Weighted A* (Dynamic)")
    if path_wa_dyn:
        print("\nDisplaying Visualization 3: WA* (Alpha=4.0) on Dynamic Grid (S to A)")
        visualize_grid_path(dynamic_grid, path_wa_dyn)
    
    path_ba_dyn, _, _, _ = bidirectional_a_star(START_NODE, GOAL_A, dynamic_grid, manhattan) # Capture path if you wish to visualize
    metrics['BA*_Dyn'] = run_experiment(bidirectional_a_star, START_NODE, GOAL_A, dynamic_grid, manhattan, 1.0, "Bidirectional A* (Dynamic)")
    
    plt.figure(figsize=(12, 6)); 
    plt.bar(list(metrics.keys()), [m['nodes'] for m in metrics.values()]); 
    plt.title('Nodes Expanded Comparison by Scenario'); 
    plt.ylabel('Nodes Expanded'); 
    plt.xticks(rotation=45, ha='right'); 
    plt.tight_layout(); 
    plt.show()

    # --- Task 3: Measure and Plot ---
    
    import matplotlib.pyplot as plt
    
    # 1. Prepare Data for Plotting
    names = list(metrics.keys())
    costs = [metrics[name]['cost'] for name in names]
    nodes = [metrics[name]['nodes'] for name in names]
    times = [metrics[name]['time'] for name in names]
    
    x_pos = range(len(names))
    
    # 2. Plot: Total Path Cost
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos, costs, align='center', color='skyblue')
    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.ylabel('Total Path Cost')
    plt.title('Algorithm Comparison: Total Path Cost')
    plt.tight_layout()
    plt.show()

    # 3. Plot: Nodes Expanded
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos, nodes, align='center', color='lightcoral')
    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.ylabel('Nodes Expanded')
    plt.title('Algorithm Comparison: Nodes Expanded (Efficiency)')
    plt.tight_layout()
    plt.show()

    # 4. Plot: Execution Time
    plt.figure(figsize=(10, 5))
    plt.bar(x_pos, times, align='center', color='lightgreen')
    plt.xticks(x_pos, names, rotation=45, ha='right')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Algorithm Comparison: Execution Time')
    plt.tight_layout()
    plt.show()
    
    print("\n--- Task 3: Plotting Complete ---")
    print("Three bar charts (Cost, Nodes, Time) have been generated.")