import copy

import numpy as np
import tik_tac_toe
import random


class Node:
    def __init__(self, game, state, parent=None):
        self.game = game  # Game module reference
        self.state = state  # Dictionary with "board" and "player"
        self.board = state['board']  # Current board state
        self.parent = parent  # Reference to the parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.value = 0  # Value accumulated from simulations
        self.player = state['player']  # Player making the current move
        self.legal_moves = self.get_legal_moves()  # Legal moves for this node
        self.ucb1_value = 999999

    def get_legal_moves(self):
        """Extracts legal moves from the game for the current state."""
        legals = self.game.get_legals(self.state)
        return legals

    def selection(self):
        """Selects a node using the UCB1 metric."""
        node = self
        while node.children:
            node = max(node.children, key=lambda child: child.ucb1())  # Use node.children instead of self.children
        return node

    def expansion(self):
        """Expands the node by adding a child corresponding to a legal move."""
        if self.legal_moves:
            move = self.legal_moves.pop()
            next_state = self.get_next_state(move)
            child_node = Node(self.game, next_state, parent=self)
            self.children.append(child_node)
            return child_node
        else:
            raise ValueError("No more moves to try.")

    def simulation(self):
        """Simulates the game by making random moves until the game ends."""
        board, player = self.state['board'], self.state['player']
        new = copy.deepcopy(self.state)
        simulated_legals = self.legal_moves
        while self.game.evaluate_state(self.state) == 0:  # Game not over
            new = self.simulate_random_move(board, simulated_legals)
            simulated_legals = self.game.get_legals(new)
        res = self.game.evaluate_state(new)
        return res
    def backpropagation(self, result):
        """Backpropagates the simulation result up the tree."""
        node = self
        while node:
            node.visits += 1
            node.value += result
            node = node.parent

    def ucb1(self, exploration_factor=1.41):
        """Calculates the UCB1 value for a node."""
        if self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_factor * (self.parent.visits / self.visits) ** 0.5

    def get_next_state(self, move):
        """Returns the new state (dictionary) after applying the move."""
        return self.game.apply_move(self.state, move)

    def simulate_random_move(self, state, legal_moves):
        """Randomly selects a legal move."""

        move = random.choice(legal_moves)
        self.state = self.get_next_state(move)
        return self.state

    def unit_children_count(self):
        """Returns the number of direct children."""
        return len(self.children)

    def total_children_count(self):
        """Returns the total count of nodes in the tree starting from this node."""
        count = 1
        for child in self.children:
            count += child.total_children_count()
        return count


def Carlo(root, iterations=100):
    """Monte Carlo Tree Search loop."""
    count = 0
    for _ in range(iterations):
        node = root.selection()
        if node.legal_moves:
            node = node.expansion()
        result = node.simulation()
        node.backpropagation(result)
    count += 1
    return max(root.children, key=lambda child: child.visits)


def self_play(game, iterations=100):
    """Simulates self-play for a given game until it ends."""
    root = Node(game, game.get_initial_state())

    # Loop until the game ends
    while game.evaluate_state(root.state) == 0:  # Game not over
        print("Current Board:")
        print(np.array(root.state['board']))

        # Run full Monte Carlo Tree Search for the current position
        best_node = Carlo(root, iterations=iterations)

        # Move to the best node (the move with the most visits)
        root = best_node

    # Get the winner and final board state
    winner = game.evaluate_state(root.state['board'])
    final_board = root.state['board']

    # Print the final board and winner
    print("Final Board:")
    print(np.array(final_board))  # Assuming the board is represented as a list of lists or 2D array
    if winner == 1:
        print("Player 1 wins!")
    elif winner == -1:
        print("Player 2 wins!")
    else:
        print("It's a draw!")
    return root.state


if __name__ == "__main__":
    game = tik_tac_toe.GameModule()
    root = Node(game, game.get_initial_state())
    next_move = Carlo(root, iterations=100)
