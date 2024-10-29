import numpy as np
import copy

def hash(arr):
    num = str(arr).replace('\n', '').replace(' ', '').replace('[', '').replace(']', '')
    return num


def eightfold_rotate(arr):
    group = []
    group.append(arr)
    group.append(np.rot90(arr, 1))  # Up
    group.append(np.rot90(arr, 2))  # Left
    group.append(np.rot90(arr, 3))  # Down
    group.append(np.fliplr(arr))  # Horizontal reflection
    group.append(np.flipud(arr))  # Vertical reflection
    group.append(np.rot90(np.fliplr(arr), 1))  # Horz Flip then rotate (transpose)
    group.append(np.rot90(np.flipud(arr), 1))  # Vert Flip then rotate

    return group

class GameModule:
    def __init__(self):
        self.initial_board = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    def __repr__(self):
        st = "TicTacToe"
        return st

    def get_initial_state(self):
        """Return the initial game state."""
        return {'board': self.initial_board, 'player': 1}



    def swap_teams(self, state):
        state['board'] *= -1
        state['player'] *= -1
        return state

    def apply_move(self, state, move):
        """Return a new game state after applying a move."""
        state['board'][[move[0]], [move[1]]] = 1
        state = self.swap_teams(state)
        return state

    def evaluate_state(self, state):
        """Evaluate the current state (for backpropagation) and return the result."""
        res = 0  # Default value for ongoing game

        # Check for Player 1 (X) win
        for row in state['board']:
            if np.all(row == 1):
                res = 1
                break
        for col in state['board'].T:
            if np.all(col == 1):
                res = 1
                break
        if np.all(np.diag(state['board']) == 1):
            res = 1
        elif np.all(np.diag(np.fliplr(state['board'])) == 1):
            res = 1

        # Check for Player -1 (O) win
        if res == 0:  # Only check for -1 win if no 1 win was found
            for row in state['board']:
                if np.all(row == -1):
                    res = -1
                    break
            for col in state['board'].T:
                if np.all(col == -1):
                    res = -1
                    break
            if np.all(np.diag(state['board']) == -1):
                res = -1
            elif np.all(np.diag(np.fliplr(state['board'])) == -1):
                res = -1

        # Check for a draw (board full and no winner)
        if res == 0 and np.all(state['board'] != 0):
            res = 2  # Draw

        return res

    def get_legals(self, state, prior=None):
        arr = state['board']
        prior = np.argwhere(arr == 0)
        novel_groups = set()
        legals = set()
        for move in prior:
            new = copy.deepcopy(arr)
            new[move[0], move[1]] = 1
            group = eightfold_rotate(new)
            canonical_group = max(group, key=hash)
            h = hash(canonical_group)
            if h not in novel_groups:
                legals.add(tuple(move))
                novel_groups.add(h)

        return list(legals)



if __name__ == '__main__':
    game = GameModule()
    init = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    print(game.get_legals(init))
