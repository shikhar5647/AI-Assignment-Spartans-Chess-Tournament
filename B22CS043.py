import random
import time
from board import Move
from config import *

class B22CS043:
    """
    A minimax-based agent with alpha-beta pruning.
    """
    def __init__(self, engine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 5
        self.cache = {}

    def get_moves(self):
        """
        Returns legal moves sorted by priority:
        1. Captures (highest value captured first)
        2. Checks (moves that put opponent in check)
        3. Saving moves (move attacked piece to safety)
        4. Quiet moves
        """
        moves = self.engine.get_legal_moves()
        capture_moves = []
        check_moves = []
        saving_moves = []
        quiet_moves = []

        # Find all own pieces under attack
        attacked_squares = set()
        turn_color = 'w' if self.engine.white_to_move else 'b'
        for r, row in enumerate(self.engine.board):
            for c, piece in enumerate(row):
                if piece.startswith(turn_color):
                    if self.engine._is_square_attacked((r, c), turn_color):
                        attacked_squares.add((r, c))

        for move in moves:
            # Captures
            if move.piece_captured != EMPTY_SQUARE:
                capture_moves.append((PIECE_VALUES.get(move.piece_captured, 0), move))
            else:
                # Checks (simulate move and see if opponent is in check)
                self.engine.make_move(move)
                if self.engine.is_in_check():
                    check_moves.append(move)
                self.engine.undo_move()

                # Saving moves: move attacked piece to a safe square
                if (move.start_row, move.start_col) in attacked_squares:
                    self.engine.make_move(move)
                    if not self.engine._is_square_attacked((move.end_row, move.end_col), turn_color):
                        saving_moves.append(move)
                    self.engine.undo_move()
                # Quiet moves
                elif move not in check_moves:
                    quiet_moves.append(move)

        # Sort captures by value descending
        capture_moves.sort(reverse=True, key=lambda x: x[0])
        sorted_moves = [m for _, m in capture_moves] + check_moves + saving_moves + quiet_moves
        return sorted_moves

    def _get_board_hash(self):
        # Hashable representation of board + turn
        board_tuple = tuple(tuple(row) for row in self.engine.board)
        return (board_tuple, self.engine.white_to_move)

    def search(self, depth, alpha=-99999, beta=99999):
        self.nodes_expanded += 1

        board_hash = self._get_board_hash()
        cache_key = (board_hash, depth)
        if cache_key in self.cache:
            return self.cache[cache_key]

        if depth == 0:
            result = (None, self.evaluate_board(self.engine.get_game_state()))
            self.cache[cache_key] = result
            return result

        best_move = None
        best_eval = -99999
        legal_moves = self.get_moves()

        if not legal_moves:
            result = (None, self.evaluate_board(self.engine.get_game_state()))
            self.cache[cache_key] = result
            return result

        for move in legal_moves:
            self.engine.make_move(move)
            _, eval = self.search(depth - 1, -beta, -alpha)
            eval = -eval
            self.engine.undo_move()

            if eval >= best_eval:
                best_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if alpha >= beta:
                break

        result = (best_move, best_eval)
        self.cache[cache_key] = result
        return result

    def get_best_move(self):
        """
        Finds and returns the best move with alpha-beta pruning.
        """
        # time.sleep(5)  # Simulate "thinking"
        self.nodes_expanded = 0
        best_move, _ = self.search(self.depth)
        return best_move

    def evaluate_board(self, game_state):
        if game_state == "checkmate":
            return -99999
        if game_state == "stalemate":
            return 0

        eval = 0
        for row_idx, row in enumerate(self.engine.board):
            for col_idx, piece in enumerate(row):
                eval += PIECE_VALUES.get(piece, 0)
                # Add positional value from Piece-Square Tables (PST)
                if piece == WHITE_PAWN:
                    eval += PAWN_PST[row_idx][col_idx]
                elif piece == BLACK_PAWN:
                    eval += PAWN_PST[BOARD_HEIGHT - 1 - row_idx][col_idx] * -1
                elif piece == WHITE_KNIGHT:
                    eval += KNIGHT_PST[row_idx][col_idx]
                elif piece == BLACK_KNIGHT:
                    eval += KNIGHT_PST[BOARD_HEIGHT - 1 - row_idx][col_idx] * -1
                elif piece == WHITE_BISHOP:
                    eval += BISHOP_PST[row_idx][col_idx]
                elif piece == BLACK_BISHOP:
                    eval += BISHOP_PST[BOARD_HEIGHT - 1 - row_idx][col_idx] * -1

        # Flip if it's black's turn
        return eval if self.engine.white_to_move else -eval