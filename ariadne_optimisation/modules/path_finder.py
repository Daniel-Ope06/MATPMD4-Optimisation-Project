import heapq


class PathFinder:
    """
    Utilizes the A* search algorithm to calculate
    the optimal route across a 2D binary occupancy grid.
    """

    def __init__(self):
        # We restrict movement to 4 cardinal directions (Up, Down, Left, Right)
        # to prevent the UGV from clipping diagonally through wall corners.
        self.movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def calc_manhattan_dist(self, current_node, target_node):
        """
        Calculates the grid-based distance between two points.
        It counts the strict horizontal and vertical steps required.
        """
        return abs(
            target_node[0] - current_node[0]) + abs(
            target_node[1] - current_node[1])

    def find_path(self, binary_matrix, start_node, goal_node):
        """
        Executes the A* algorithm

        Args:
            binary_matrix (numpy array): 2D grid where 1 is path, 0 is wall.
            start_node (tuple): (x, y) grid index of the UGV.
            goal_node (tuple): (x, y) grid index of the Human.

        Returns:
            list: A sequence of (x, y) tuples representing the optimal path.
            Returns an empty list [] if no valid path exists.
        """
        # Keep track of checked cells
        evaluated_nodes = set()

        # Dictionary linking each cell to the cell we came from
        # (used to draw the final path)
        path_history = {}

        # Track the exact number of steps taken from the start
        cost_from_start = {start_node: 0}

        # The priority queue:
        # determines which node to check next based on the lowest total cost.
        # Format: (total_estimated_cost, node_coordinates)
        priority_queue = []
        initial_guess = self.calc_manhattan_dist(start_node, goal_node)
        heapq.heappush(priority_queue, (initial_guess, start_node))

        # Get the map boundaries
        max_y, max_x = binary_matrix.shape

        while priority_queue:
            # Pop the cell with the lowest estimated total cost
            current_node = heapq.heappop(priority_queue)[1]

            # If Target Reached:
            # Trace the path backward using path_history.
            if current_node == goal_node:
                final_path = []
                while current_node in path_history:
                    final_path.append(current_node)
                    current_node = path_history[current_node]
                final_path.append(start_node)

                # Reverse the list so it goes from Start -> Goal
                return final_path[::-1]

            # Else:
            # Mark this cell as checked
            evaluated_nodes.add(current_node)

            # Look at the 4 adjacent cells
            for move_x, move_y in self.movements:
                neighbor_node = (
                    current_node[0] + move_x, current_node[1] + move_y)

                # Boundary Check: Is the neighbor off the edge of the map?
                if not (0 <= neighbor_node[0] < max_x and
                        0 <= neighbor_node[1] < max_y):
                    continue

                # Collision Check: Is the neighbor a wall (0)?
                if binary_matrix[neighbor_node[1]][neighbor_node[0]] == 0:
                    continue

                # Calculate what the cost WOULD be if we moved to this neighbor
                tentative_cost = cost_from_start[current_node] + 1

                # If we've already checked this neighbor AND
                # found a faster way there, skip it
                if (neighbor_node in evaluated_nodes and
                        tentative_cost >= cost_from_start.get(
                            neighbor_node, 0)):
                    continue

                # If this is the first time seeing this neighbor, OR
                # we found a faster route to it:
                is_in_queue = neighbor_node in [
                    item[1] for item in priority_queue]
                if (tentative_cost < cost_from_start.get(neighbor_node, 0) or
                        not is_in_queue):
                    # Record how we got here
                    path_history[neighbor_node] = current_node
                    cost_from_start[neighbor_node] = tentative_cost

                    # Calculate total cost:
                    # exact cost so far + guessed cost to the end
                    total_estimated_cost = tentative_cost + \
                        self.calc_manhattan_dist(neighbor_node, goal_node)

                    # Add it to the queue to be evaluated
                    heapq.heappush(
                        priority_queue, (total_estimated_cost, neighbor_node))

        # If the queue runs out of cells to check, there is no possible route
        return []
