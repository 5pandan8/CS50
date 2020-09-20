"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    count_X = 0
    count_O = 0

    for i in range(3):
        for j in range(3):
            if board[i][j] == X:
                count_X += 1
            elif board[i][j] == O:
                count_O += 1

    if count_X == count_O:
        return X
    else:
        return O

    raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    # To store the set of actions
    action_list = set()

    for i in range(3):
        for j in range(3):

            # If the block is empty it is a possible action
            if board[i][j] == EMPTY:
                action_list.add((i, j))

    return action_list

    raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Making a deepcopy of the board
    new_board = copy.deepcopy(board)

    # Finding which players turn it is
    value = player(new_board)

    # Changing the board according to the action
    if new_board[action[0]][action[1]] == EMPTY:
        new_board[action[0]][action[1]] = value
        return new_board
    else:
        raise NameError('Invalid Action')
    raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check for winning condition in the rows
    for i in range(3):
        if board[i][0] == board[i][1] and board[i][1] == board[i][2] and board[i][2] == board[i][0] and board[i][0] == X:
            return X
        elif board[i][0] == board[i][1] and board[i][1] == board[i][2] and board[i][2] == board[i][0] and board[i][0] == O:
            return O

    # Check for winning condition in the columns
    for i in range(3):
        if board[0][i] == board[1][i] and board[1][i] == board[2][i] and board[2][i] == board[0][i] and board[0][i] == X:
            return X
        if board[0][i] == board[1][i] and board[1][i] == board[2][i] and board[2][i] == board[0][i] and board[0][i] == O:
            return O

    # Check winning condition in diagonals
    if board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] == board[2][2] and board[0][0] == X:
        return X
    elif board[0][0] == board[1][1] and board[1][1] == board[2][2] and board[0][0] == board[2][2] and board[0][0] == O:
        return O
    if board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] == board[2][0] and board[0][2] == X:
        return X
    elif board[0][2] == board[1][1] and board[1][1] == board[2][0] and board[0][2] == board[2][0] and board[0][2] == O:
        return O

    return None

    raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) == X or winner(board) == O:
        return True
    else:
        for i in range(3):
            for j in range(3):
                if board[i][j] == EMPTY:
                    return False

    return True

    raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1
    else:
        return 0

    raise NotImplementedError


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    current_player = player(board)
    # List the value v
    value_list = []

    # List of tuple having (action, value v)
    action_list = []

    if current_player == X:
        for action in actions(board):
            value_list.append(min_value(result(board, action)))
            action_list.append((action, value_list[-1]))

        value = max(value_list)
        for i in range(len(action_list)):
            if action_list[i][1] == value:
                return action_list[i][0]

    if current_player == O:
        for action in actions(board):
            value_list.append(max_value(result(board, action)))
            action_list.append((action, value_list[-1]))

        value = min(value_list)
        for i in range(len(action_list)):
            if action_list[i][1] == value:
                return action_list[i][0]

    raise NotImplementedError


def max_value(board):
    if terminal(board):
        return utility(board)
    v = -100
    for action in actions(board):
        v = max(v, min_value(result(board, action)))
    return v


def min_value(board):
    if terminal(board):
        return utility(board)
    v = 100
    for action in actions(board):
        v = min(v, max_value(result(board, action)))
    return v
