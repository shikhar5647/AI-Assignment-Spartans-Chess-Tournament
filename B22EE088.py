# import math
# from config import *
# from board import Move

# class B22EE088:
#     """
#     Optimized AI Player with minimax + alpha-beta pruning,
#     piece-square tables, mobility, and king safety heuristics.
#     """

#     def __init__(self, board, depth=3, aggressive=False):
#         self.board = board
#         self.nodes_expanded = 0
#         self.depth = depth
#         self.aggressive = aggressive  # if True, prioritizes captures

#     def get_best_move(self):
#         """
#         Choose the best move using minimax with alpha-beta pruning.
#         """
#         board = self.board
#         white_to_move = board.white_to_move
#         best_move = None
#         best_score = -math.inf if white_to_move else math.inf
#         alpha, beta = -math.inf, math.inf

#         legal_moves = board.get_legal_moves()
#         if not legal_moves:
#             return None

#         # Capture-priority sorting
#         if self.aggressive:
#             legal_moves.sort(key=lambda m: PIECE_VALUES.get(m.piece_captured, 0), reverse=True)

#         for move in legal_moves:
#             board.make_move(move)
#             score = self._minimax(self.depth - 1, alpha, beta, not white_to_move)
#             board.undo_move()

#             if white_to_move:  # White wants max
#                 if score > best_score:
#                     best_score, best_move = score, move
#                 alpha = max(alpha, best_score)
#             else:  # Black wants min
#                 if score < best_score:
#                     best_score, best_move = score, move
#                 beta = min(beta, best_score)

#             if beta <= alpha:  # alpha-beta cutoff
#                 break

#         return best_move

#     def _minimax(self, depth, alpha, beta, maximizing_player):
#         self.nodes_expanded += 1
#         board = self.board
#         state = board.get_game_state()

#         if depth == 0 or state != "ongoing":
#             return self.evaluate_board()

#         legal_moves = board.get_legal_moves()
#         if not legal_moves:
#             return self.evaluate_board()

#         # Capture-priority sorting
#         if self.aggressive:
#             legal_moves.sort(key=lambda m: PIECE_VALUES.get(m.piece_captured, 0), reverse=True)

#         if maximizing_player:  # White
#             max_eval = -math.inf
#             for move in legal_moves:
#                 board.make_move(move)
#                 eval_score = self._minimax(depth - 1, alpha, beta, False)
#                 board.undo_move()
#                 max_eval = max(max_eval, eval_score)
#                 alpha = max(alpha, eval_score)
#                 if beta <= alpha:
#                     break
#             return max_eval
#         else:  # Black
#             min_eval = math.inf
#             for move in legal_moves:
#                 board.make_move(move)
#                 eval_score = self._minimax(depth - 1, alpha, beta, True)
#                 board.undo_move()
#                 min_eval = min(min_eval, eval_score)
#                 beta = min(beta, eval_score)
#                 if beta <= alpha:
#                     break
#             return min_eval

#     def evaluate_board(self):
#         """
#         Heuristic evaluation:
#         - Material balance
#         - Piece-square tables
#         - Mobility (number of legal moves)
#         - King safety (penalize exposed kings)
#         """
#         board_matrix = self.board.board
#         score = 0

#         for r in range(BOARD_HEIGHT):
#             for c in range(BOARD_WIDTH):
#                 piece = board_matrix[r][c]
#                 if piece == EMPTY_SQUARE:
#                     continue

#                 base_value = PIECE_VALUES[piece]
#                 pst_value = 0

#                 # Piece-Square Tables
#                 if piece == WHITE_PAWN:
#                     pst_value = PAWN_PST[r][c]
#                 elif piece == BLACK_PAWN:
#                     pst_value = -PAWN_PST[BOARD_HEIGHT - 1 - r][c]
#                 elif piece == WHITE_KNIGHT:
#                     pst_value = KNIGHT_PST[r][c]
#                 elif piece == BLACK_KNIGHT:
#                     pst_value = -KNIGHT_PST[BOARD_HEIGHT - 1 - r][c]
#                 elif piece == WHITE_BISHOP:
#                     pst_value = BISHOP_PST[r][c]
#                 elif piece == BLACK_BISHOP:
#                     pst_value = -BISHOP_PST[BOARD_HEIGHT - 1 - r][c]
#                 elif piece == WHITE_KING:
#                     pst_value = KING_PST_LATE_GAME[r][c]
#                 elif piece == BLACK_KING:
#                     pst_value = -KING_PST_LATE_GAME[BOARD_HEIGHT - 1 - r][c]

#                 score += base_value + pst_value

#         # Mobility bonus
#         mobility = len(self.board.get_legal_moves())
#         score += 0.1 * mobility if self.board.white_to_move else -0.1 * mobility

#         white_king = self.board._find_king('w')
#         black_king = self.board._find_king('b')
#         if white_king:
#             score -= 0.2 * (abs(white_king[0] - BOARD_HEIGHT // 2))  # white king centralization penalty
#         if black_king:
#             score += 0.2 * (abs(black_king[0] - BOARD_HEIGHT // 2))  # black king centralization penalty

#         return score

import math
import random
from config import *
from board import Move

# Agent class expected by your runner (callable as B22EE088(engine))
class B22EE088:
    """
    Minimax + Alpha-Beta agent aligned with Spartans tournament scoring:
      - pawn=20, knight=70, bishop=70 (use PIECE_VALUES from config)
      - check = +2
      - checkmate = +600 (terminal override)
    Constructor matches runner expectation: B22EE088(engine, depth=3, aggressive=False)
    """
    def __init__(self, board, depth=3, aggressive=False):
        self.board = board
        self.depth = depth            # search depth used for minimax
        self.aggressive = aggressive  # if True, prefer captures in ordering
        self.nodes_expanded = 0
        self.transposition_table = {}  # optional: for future use
        # internal evaluation constants (align with scoring)
        # use local mapping so we don't need to modify config
        self.eval_piece_values = {
            WHITE_PAWN: 20, BLACK_PAWN: -20,
            WHITE_KNIGHT: 70, BLACK_KNIGHT: -70,
            WHITE_BISHOP: 70, BLACK_BISHOP: -70,
            # Put a large value for king so search prefers mate (positive for white).
            WHITE_KING: 600, BLACK_KING: -600
        }

        # PSTs are read from config: PAWN_PST, KNIGHT_PST, BISHOP_PST, KING_PST_LATE_GAME
        # small random jitter to break ties
        self.jitter = 0.001
    def _move_ordering(self, move):
        score = 0
        if move.piece_captured != EMPTY_SQUARE:
            score += 100 + PIECE_VALUES.get(move.piece_captured, 0)
        if getattr(move, "promotion", None):
            score += 50
        return score
    # ---------- public API ----------
    def get_best_move(self):
        """Return the best Move for current board using minimax alpha-beta (depth = self.depth)."""
        self.nodes_expanded = 0
        engine = self.board
        white_to_move = engine.white_to_move

        legal_moves = engine.get_legal_moves()
        if not legal_moves:
            return None
        if len(legal_moves) == 1:
            return legal_moves[0]

        best_move = None
        best_score = -math.inf if white_to_move else math.inf
        alpha, beta = -math.inf, math.inf

        # Move ordering: prefer captures (MVV-LVA style) when aggressive
        if self.aggressive:
            legal_moves.sort(key=self._move_ordering, reverse=True)

        for mv in legal_moves:
            engine.make_move(mv)
            score = self._minimax(self.depth - 1, alpha, beta, not white_to_move)
            engine.undo_move()

            if white_to_move:
                if score > best_score:
                    best_score, best_move = score, mv
                alpha = max(alpha, best_score)
            else:
                if score < best_score:
                    best_score, best_move = score, mv
                beta = min(beta, best_score)

            if beta <= alpha:
                break

        return best_move

    # ---------- minimax with alpha-beta ----------
    def _minimax(self, depth, alpha, beta, maximizing_player):
        self.nodes_expanded += 1
        state = self.board.get_game_state()
        board_hash = str(self.board.board) + str(self.board.white_to_move)
        key = (board_hash, depth, maximizing_player)
        if key in self.transposition_table:
            return self.transposition_table[key]
        # terminal or depth cutoff
        if depth == 0 or state != "ongoing":
            return self._evaluate_terminal_or_board(state)

        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return self._evaluate_terminal_or_board(state)

        # optional capture-first ordering for better pruning
        if self.aggressive:
            legal_moves.sort(key=lambda m: self._capture_score(m), reverse=True)

        if maximizing_player:  # White to move in this node
            v = -math.inf
            for mv in legal_moves:
                self.board.make_move(mv)
                val = self._minimax(depth - 1, alpha, beta, False)
                self.board.undo_move()
                v = max(v, val)
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:  # Black to move in this node
            v = math.inf
            for mv in legal_moves:
                self.board.make_move(mv)
                val = self._minimax(depth - 1, alpha, beta, True)
                self.board.undo_move()
                v = min(v, val)
                beta = min(beta, v)
                if alpha >= beta:
                    break
            self.transposition_table[key] = v
            return v

    # ---------- evaluation helpers ----------
    def _evaluate_terminal_or_board(self, state):
        if state == "checkmate":
            if self.board.white_to_move:
                return -600.0  # side to move (White) is mated
            else:
                return 600.0   # side to move (Black) is mated
        elif state == "stalemate":
            return self._evaluate_board_basic()
        else:
            score = self._evaluate_board_basic()
            # Flip perspective: if it's Black's turn, return negative of evaluation
            return score if self.board.white_to_move else -score

    def _evaluate_board_basic(self):
        """
        Evaluate board position (White positive = advantage for White).
        Combines:
          - material using tournament piece values,
          - piece-square tables for positional weight,
          - mobility (difference in legal moves),
          - simple pawn-structure penalties and king safety,
          - small jitter to break ties.
        """
        board = self.board.board
        score = 0.0

        # Material + PST
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = board[r][c]
                if piece == EMPTY_SQUARE:
                    continue

                # material according to tournament scoring (use local eval map)
                score += self.eval_piece_values.get(piece, 0)

                # PST addition (white uses table as-is, black uses mirrored)
                if piece == WHITE_PAWN:
                    score += PAWN_PST[r][c]
                elif piece == BLACK_PAWN:
                    score -= PAWN_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_KNIGHT:
                    score += KNIGHT_PST[r][c]
                elif piece == BLACK_KNIGHT:
                    score -= KNIGHT_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_BISHOP:
                    score += BISHOP_PST[r][c]
                elif piece == BLACK_BISHOP:
                    score -= BISHOP_PST[BOARD_HEIGHT - 1 - r][c]
                elif piece == WHITE_KING:
                    score += KING_PST_LATE_GAME[r][c]
                elif piece == BLACK_KING:
                    score -= KING_PST_LATE_GAME[BOARD_HEIGHT - 1 - r][c]

        # Mobility: difference in number of legal moves (cheap)
        # We compute legal moves for current side and hypothetical opponent by flipping white_to_move
        try:
            current_wtm = self.board.white_to_move
            # count legal moves for white and black by temporarily toggling (cheap approach):
            # Note: we must not disturb real engine state, so call get_legal_moves only in current state:
            # Instead approximate by counting current player's legal moves and using symmetry for opponent:
            # simpler: count current legal moves and also undo simulated toggles are tricky; use direct difference:
            legal_now = len(self.board.get_legal_moves())
            # Quick heuristic: if it's white_to_move, treat mobility advantage as +legal_now - opponent_estimate
            # We estimate opponent moves by briefly toggling turn using make/undo of a null move is not available.
            # So we approximate mobility by counting the moves of the current side only (encourages activity).
            if current_wtm:
                score += 0.12 * legal_now
            else:
                score -= 0.12 * legal_now
        except Exception:
            pass

        # Pawn-structure simple penalties (doubled or isolated)
        score += self._pawn_structure_score()

        # King safety: penalize exposed kings (few defenders near)
        score += self._king_safety_score()

        # Check bonus: if the side NOT to move is in check, that's good for the side who just moved.
        # engine.is_in_check() tells whether current player to move is in check.
        if self.board.is_in_check():
            # current player in check -> bad for them
            if self.board.white_to_move:
                score -= 2  # white in check
            else:
                score += 2  # black in check

        # tiny jitter to break ties
        score += random.uniform(-self.jitter, self.jitter)
        return score

    def _capture_score(self, move):
        """Order captures by victim value - attacker value (MVV-LVA simple)."""
        if move.piece_captured == EMPTY_SQUARE:
            return 0
        victim = PIECE_VALUES.get(move.piece_captured, 0)
        attacker = PIECE_VALUES.get(move.piece_moved, 0)
        # higher better
        return victim - attacker

    def _pawn_structure_score(self):
        """
        Return small penalty for doubled/isolated pawns (white negative reduces score, black positive increases).
        This function returns a signed value from White's perspective.
        """
        w_files = [0]*BOARD_WIDTH
        b_files = [0]*BOARD_WIDTH
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                p = self.board.board[r][c]
                if p == WHITE_PAWN:
                    w_files[c] += 1
                elif p == BLACK_PAWN:
                    b_files[c] += 1

        s = 0
        # doubled pawn penalty
        for f in range(BOARD_WIDTH):
            if w_files[f] > 1:
                s -= 6 * (w_files[f] - 1)
            if b_files[f] > 1:
                s += 6 * (b_files[f] - 1)
        # isolated pawn penalty
        for f in range(BOARD_WIDTH):
            if w_files[f] == 1 and ((f-1 < 0 or w_files[f-1] == 0) and (f+1 >= BOARD_WIDTH or w_files[f+1] == 0)):
                s -= 4
            if b_files[f] == 1 and ((f-1 < 0 or b_files[f-1] == 0) and (f+1 >= BOARD_WIDTH or b_files[f+1] == 0)):
                s += 4
        return s

    def _king_safety_score(self):
        """
        Simple king safety: reward defenders (pawns) near king and penalize king advanced into center early.
        Positive = good for White, negative = good for Black.
        """
        s = 0
        wk = self.board._find_king('w')
        bk = self.board._find_king('b')

        if wk:
            wr, wc = wk
            # prefer white king on back ranks (higher row index ~ back rank) -> small penalty if advanced
            s -= 2 * ((BOARD_HEIGHT - 1) - wr) * -0.0  # negligible here, kept formulaic
            # count white pawn defenders around king
            defenders = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = wr + dr, wc + dc
                    if 0 <= rr < BOARD_HEIGHT and 0 <= cc < BOARD_WIDTH:
                        p = self.board.board[rr][cc]
                        if p == WHITE_PAWN:
                            defenders += 1
            s += 5 * defenders

        if bk:
            br, bc = bk
            defenders = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = br + dr, bc + dc
                    if 0 <= rr < BOARD_HEIGHT and 0 <= cc < BOARD_WIDTH:
                        p = self.board.board[rr][cc]
                        if p == BLACK_PAWN:
                            defenders += 1
            s -= 5 * defenders

        return s
