from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from heuristics import Heuristic
    from board import Board


class PlayerController:
    """Abstract class defining a player
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        self.player_id = player_id
        self.game_n = game_n
        self.heuristic = heuristic

    def get_eval_count(self) -> int:
        """
        Returns:
            int: The amount of times the heuristic was used to evaluate a board state
        """
        return self.heuristic.eval_count

    def __str__(self) -> str:
        """
        Returns:
            str: representation for representing the player on the board
        """
        if self.player_id == 1:
            return 'X'
        return 'O'

    @abstractmethod
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        pass


class MinMaxPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth

    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        if self.player_id == 1:     # initialize the best value to minus infinity
            opponent_id = 2
        else:
            opponent_id = 1

        best_value = -np.inf     # initialize the best value to minus infinity
        best_move = 0

        for col in range(board.width):
            if not board.is_valid(col):     # check whether the board is legal
                continue

            new_board: Board = board.get_new_board(col, self.player_id)
            value: int = self.minimax(new_board, opponent_id, self.depth-1)

            if value > best_value:
                best_value = value
                best_move = col

        #best_action = self.heuristic.get_best_action(self.player_id, new_board)
        #print("best_action", best_action)
        print("returned best_move", best_move)
        return best_move

    def minimax(self, board: Board, player_to_move: int, depth: int) -> int:
        state = board.get_board_state()
        result = self.heuristic.winning(state, self.game_n)  # 1,2, -1 or 0 if still no winner
        
        if depth == 0 or result != 0:
            return self.heuristic.evaluate_board(self.player_id, board)
        
        if player_to_move == 1:
            Nextplayer = 2
        else:
            Nextplayer = 1

        if player_to_move == self.player_id:            # If it's our turn → maximize
            best_value = -np.inf
        else:                 # Opponent's turn → minimize
            best_value = np.inf

        for col in range(board.width):
            if not board.is_valid(col):     # check whether the board is legal
                continue
            new_board:  Board = board.get_new_board(col, player_to_move)
            value = self.minimax(new_board, Nextplayer, depth-1)

            if player_to_move == self.player_id:
                if value > best_value:              # Maximizing if it's our turn
                    best_value = value
            else:
                if value < best_value:  # minimizing if it's the opponents turn
                    best_value = value

        print("max_move", best_value)
        return best_value


class AlphaBetaPlayer(PlayerController):
    """Class for the minmax player using the minmax algorithm with alpha-beta pruning
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, depth: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            depth (int): the max search depth
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)
        self.depth: int = depth

    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """

        if self.player_id == 1:     # ensure the players play turn by turn
            opponent_id = 2
        else:
            opponent_id = 1

        best_value = -np.inf    # initialize the best value to minus infinity
        best_move = 0

        for col in range(board.width):  # check whether the board is legal
            if not board.is_valid(col):
                continue

            new_board: Board = board.get_new_board(col, self.player_id)
            value: int = self.minimax_with_ab_pruning(new_board, opponent_id, self.depth-1, -np.inf, np.inf)

            if value > best_value:
                best_value = value
                best_move = col

        print("Best_move is: ", best_move)
        return best_move

    def minimax_with_ab_pruning(self, board: Board, player_to_move: int, depth: int, a, b) -> int:
        state = board.get_board_state()
        result = self.heuristic.winning(state, self.game_n)

        if depth == 0 or result != 0:
            return self.heuristic.evaluate_board(self.player_id, board)

        if player_to_move == 1:
            Nextplayer = 2
        else:
            Nextplayer = 1

        if player_to_move == self.player_id:  # If it is our turn → maximize
            best_value = -np.inf
        else:  # if it is the opponent's turn → minimize
            best_value = np.inf

        for col in range(board.width):
            if not board.is_valid(col):  # check whether the board is legal
                continue

            new_board: Board = board.get_new_board(col, player_to_move)
            value = self.minimax_with_ab_pruning(new_board, Nextplayer, depth - 1, a, b)

            if player_to_move == self.player_id:    # Maximizing if it's our turn
                if value > best_value:      # if the new value is bigger than the previous value
                    best_value = value  # update best value to be the new value
                a = max(a, best_value)  # alpha is the biggest value
                if b <= a:  # if we know this branch will not lead to a better value
                    break   # stop expanding
            else:   # minimizing if it's the opponents turn
                if value < best_value:  # if the new value is smaller than the previous value
                    best_value = value      # update best value to be the new value
                b = min(b, best_value)  # beta is the smallest value
                if b <= a:  # if we know this branch will not lead to a better value
                    break   # stop expanding

        return best_value


class HumanPlayer(PlayerController):
    """Class for the human player
    Inherits from Playercontroller
    """
    def __init__(self, player_id: int, game_n: int, heuristic: Heuristic) -> None:
        """
        Args:
            player_id (int): id of a player, can take values 1 or 2 (0 = empty)
            game_n (int): n in a row required to win
            heuristic (Heuristic): heuristic used by the player
        """
        super().__init__(player_id, game_n, heuristic)

    
    def make_move(self, board: Board) -> int:
        """Gets the column for the player to play in

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        print(board)

        if self.heuristic is not None:
            print(f'Heuristic {self.heuristic} calculated the best move is:', end=' ')
            print(self.heuristic.get_best_action(self.player_id, board) + 1, end='\n\n')

        col: int = self.ask_input(board)

        print(f'Selected column: {col}')
        return col - 1
    

    def ask_input(self, board: Board) -> int:
        """Gets the input from the user

        Args:
            board (Board): the current board

        Returns:
            int: column to play in
        """
        try:
            col: int = int(input(f'Player {self}\nWhich column would you like to play in?\n'))
            assert 0 < col <= board.width
            assert board.is_valid(col - 1)
            return col
        except ValueError: # If the input can't be converted to an integer
            print('Please enter a number that corresponds to a column.', end='\n\n')
            return self.ask_input(board)
        except AssertionError: # If the input matches a full or non-existing column
            print('Please enter a valid column.\nThis column is either full or doesn\'t exist!', end='\n\n')
            return self.ask_input(board)
        