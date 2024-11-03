import copy
import numpy as np
import tik_tac_toe  # Assuming this module provides the game logic
import random


class Node:
    def __init__(self, game, state, parent=None, action=None):
        self.game = game  # Game module reference
        self.state = copy.deepcopy(state)  # Copy of the state to ensure immutability
        self.board = self.state['board']  # Current board state
        self.parent = parent  # Reference to the parent node
        self.action = action  # Action that led to this node from the parent
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Value accumulated from simulations
        self.player = self.state['player']  # Player making the current move
        self.legal_moves = self.get_legal_moves()  # Legal moves for this node
        self.untried_moves = list(self.legal_moves)  # Moves yet to be tried (initially all legal moves)

    def get_legal_moves(self):
        """Extracts legal moves from the game for the current state."""
        return self.game.get_legals(self.state)

    def selection(self):
        """Selects a node using the UCB1 metric."""
        node = self
        while node.children and len(node.untried_moves) == 0:
            node = max(node.children, key=lambda child: child.ucb1())
        return node

    def expansion(self):
        """Expands the node by adding a child corresponding to an unexplored legal move."""
        if not self.untried_moves:
            return None  # If no untried moves are left, return
        move = self.untried_moves.pop()  # Choose an untried move
        next_state = self.get_next_state(move)  # Get the resulting state after applying the move
        child_node = Node(self.game, next_state, parent=self, action=move)  # Create the child node
        self.children.append(child_node)  # Add this child to the list of children
        return child_node



    def simulation(self):

        """Simulates the game by making random moves until the game ends."""
        current_state = copy.deepcopy(self.state)  # Work with a deep copy of the current state
        while self.game.evaluate_state(current_state) is None:
            legal_moves = self.game.get_legals(current_state)  # Get legal moves for the current state
            move = random.choice(legal_moves)  # Randomly select a move
            current_state = game.apply_move(current_state, move)  # Apply the move to get the next state
        return self.game.evaluate_state(current_state)  # Return the result of the game (1, -1, or 0 for draw)


    def backpropagation(self, result):
        """Backpropagates the simulation result up the tree."""
        node = self
        while node:
            node.visits += 1
            node.value += result if node.player == self.player else -result  # Flip value based on player
            node = node.parent

    def ucb1(self, exploration_factor=1.41):
        """Calculates the UCB1 value for a node."""
        if self.visits == 0:
            return float('inf')  # Infinite value for unvisited nodes to encourage exploration
        exploitation = self.value / self.visits
        exploration = exploration_factor * (np.log(self.parent.visits) / self.visits) ** 0.5
        return exploitation + exploration

    def get_next_state(self, move):
        """Returns a new state after applying the move, ensuring state immutability."""
        # Copy the current state deeply
        next_state = copy.deepcopy(self.state)

        # Apply the move in the game to get a new state (ensure apply_move doesn't modify `next_state` in place)
        next_state = self.game.apply_move(next_state, move)

        # Return the new state without changing the current node
        return next_state


def self_play(game, iterations=100):
    """Simulates self-play for a given game until it ends."""
    root = Node(game, game.get_initial_state())  # Initialize the root node
    # Loop until the game ends
    while game.evaluate_state(root.state) is None:  # Game not over
        print("Current Board:")
        print(np.array(root.state['board']))

        # Run Monte Carlo Tree Search for the current position
        best_action = Carlo(root, iterations=iterations)  # Get the best action

        # Apply the best action to get the new state
        root.state = root.get_next_state(best_action)
        root = Node(game, root.state)  # Create a new root node from the new state

    # Get the winner and final board state
    winner = game.evaluate_state(root.state)
    final_board = root.state['board']

    # Print the final board and winner
    print("Final Board:")
    print(np.array(final_board))
    if winner == 1:
        print("Player 1 wins!")
    elif winner == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")
    return root.state

def Carlo(root, iterations=100):
    """Monte Carlo Tree Search loop."""
    for _ in range(iterations):
        # Step 1: Selection
        node = root.selection() # Select the most promising node based on the UCB1 metric

        # Step 2: Expansion
        if node.untried_moves:  # Check if there are any unexplored legal moves
            node = node.expansion()  # Single node expansion AND SELECT INTO IT

        # Step 3: Simulation
        if node and node.game.evaluate_state(node.state) is None:
            result = node.simulation()  # Simulate the game from the new node
            pass
        else:
            # If the game is over, evaluate the result immediately
            result = node.game.evaluate_state(node.state)

        # Step 4: Backpropagation
        node.backpropagation(result)  # Update the value and visit count of the nodes in the path

    # Select the child with the most visits and return its corresponding action
    best_child = max(root.children, key=lambda child: child.visits)
    return best_child.action  # Return the action leading to the best child node

if __name__ == "__main__":
    game = tik_tac_toe.GameModule()  # Initialize the game module
    root = Node(game, game.get_initial_state())  # Initialize the root node
    best_move = Carlo(root, iterations=5)  # Run Monte Carlo Tree Search to find the best move
    print("Best move:", best_move)  # Print the best move found

