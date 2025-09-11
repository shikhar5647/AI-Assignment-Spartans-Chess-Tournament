import math
from config import *
from board import Move

class B22EE088:
    """
    AI Player with minimax + alpha-beta pruning.
    """
    def __init__(self, board, depth=3, aggressive=False):
        self.board = board
        self.nodes_expanded = 0
        self.depth = depth
        self.aggressive = aggressive  # if True, prioritizes captures

    def get_best_move(self):
        """
        Choose the best move using minimax with alpha-beta pruning.
        """
        best_move = None
        best_score = -math.inf if self.board.white_to_move else math.inf
        alpha, beta = -math.inf, math.inf

        legal_moves = self.board.get_legal_moves()
        if self.aggressive:
            # Sort moves so captures are searched first
            legal_moves.sort(key=lambda m: 0 if m.piece_captured == EMPTY_SQUARE else 1, reverse=True)

        for move in legal_moves:
            self.board.make_move(move)
            score = self._minimax(self.depth - 1, alpha, beta, not self.board.white_to_move)
            self.board.undo_move()

            if self.board.white_to_move:  # White wants max
                if score > best_score:
                    best_score, best_move = score, move
                alpha = max(alpha, best_score)
            else:  # Black wants min
                if score < best_score:
                    best_score, best_move = score, move
                beta = min(beta, best_score)

        return best_move

    def _minimax(self, depth, alpha, beta, maximizing_player):
        self.nodes_expanded += 1
        state = self.board.get_game_state()

        if depth == 0 or state != "ongoing":
            return self.evaluate_board()

        legal_moves = self.board.get_legal_moves()
        if self.aggressive:
            legal_moves.sort(key=lambda m: 0 if m.piece_captured == EMPTY_SQUARE else 1, reverse=True)

        if maximizing_player:  # White
            max_eval = -math.inf
            for move in legal_moves:
                self.board.make_move(move)
                eval = self._minimax(depth - 1, alpha, beta, False)
                self.board.undo_move()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:  # Beta cut-off
                    break
            return max_eval
        else:  # Black
            min_eval = math.inf
            for move in legal_moves:
                self.board.make_move(move)
                eval = self._minimax(depth - 1, alpha, beta, True)
                self.board.undo_move()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:  # Alpha cut-off
                    break
            return min_eval

    def evaluate_board(self):
        """
        Heuristic evaluation with material + piece-square tables.
        """
        score = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue

                base_value = PIECE_VALUES[piece]
                pst_value = 0

                if piece == WHITE_PAWN:
                    pst_value = PAWN_PST[r][c]
                elif piece == BLACK_PAWN:
                    pst_value = -PAWN_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_KNIGHT:
                    pst_value = KNIGHT_PST[r][c]
                elif piece == BLACK_KNIGHT:
                    pst_value = -KNIGHT_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_BISHOP:
                    pst_value = BISHOP_PST[r][c]
                elif piece == BLACK_BISHOP:
                    pst_value = -BISHOP_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_KING:
                    pst_value = KING_PST_LATE_GAME[r][c]
                elif piece == BLACK_KING:
                    pst_value = -KING_PST_LATE_GAME[BOARD_HEIGHT - 1 - r][c]

                score += base_value + pst_value

        return score
