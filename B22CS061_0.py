import random
import time
from typing import Optional, Tuple, List

from config import *
from board import GameEngine, Move


class B22CS061:
    def __init__(self, engine: GameEngine):
        self.engine = engine
        self.nodes_expanded = 0
        self.depth = 2
        self.max_time_per_move = 0.2
        self._deadline: float = 0.0

    def get_best_move(self) -> Optional[Move]:
        legal_moves = self.engine.get_legal_moves()
        if not legal_moves:
            return None

        self.nodes_expanded = 0
        self._deadline = time.perf_counter() + self.max_time_per_move

        legal_moves = self._order_moves(legal_moves)

        best_move: Optional[Move] = legal_moves[0]
        scored_moves: List[Tuple[int, Move]] = []

        for current_depth in range(1, self.depth + 1):
            try:
                scored_moves = self._search_root_all(legal_moves, current_depth)
                if scored_moves:
                    best_move = scored_moves[0][1]
            except TimeoutError:
                break

        if scored_moves and len(scored_moves) > 1 and random.random() < 0.30:
            return scored_moves[1][1]

        return best_move

    def evaluate_board(self) -> int:
        game_state = self.engine.get_game_state()
        if game_state == "checkmate":
            return -99999 if self.engine.white_to_move else 99999
        if game_state == "stalemate":
            return 0

        score = 0
        for row in self.engine.board:
            for piece in row:
                if piece == EMPTY_SQUARE:
                    continue
                score += PIECE_VALUES.get(piece, 0)
        return score

   
    # search
   
    def _search_root_all(self, moves, depth: int) -> List[Tuple[int, Move]]:
        alpha, beta = -10**9, 10**9
        color = 1 if self.engine.white_to_move else -1
        scored: List[Tuple[int, Move]] = []

        for move in moves:
            self._guard_time()
            self.engine.make_move(move)
            score = -self._negamax(depth - 1, -beta, -alpha, -color)
            self.engine.undo_move()
            scored.append((score, move))
            if score > alpha:
                alpha = score

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _negamax(self, depth: int, alpha: int, beta: int, color: int) -> int:
        self._guard_time()

        game_state = self.engine.get_game_state()
        if game_state == "checkmate":
            return -99999
        if game_state == "stalemate":
            return 0

        if depth == 0:
            self.nodes_expanded += 1
            return color * self.evaluate_board()

        best_score = -10**9
        moves = self.engine.get_legal_moves()
        if not moves:
            return -99999 if self.engine.is_in_check() else 0

        moves = self._order_moves(moves)

        for move in moves:
            self._guard_time()
            self.engine.make_move(move)
            score = -self._negamax(depth - 1, -beta, -alpha, -color)
            self.engine.undo_move()

            if score > best_score:
                best_score = score
            if score > alpha:
                alpha = score
            if alpha >= beta:
                break

        return best_score

    def _order_moves(self, moves):
        
        def move_key(m: Move):
            if m.piece_captured == EMPTY_SQUARE:
                return 0
            return abs(PIECE_VALUES.get(m.piece_captured, 0))

        return sorted(moves, key=move_key, reverse=True)

    def _guard_time(self) -> None:
        if time.perf_counter() >= self._deadline:
            raise TimeoutError


