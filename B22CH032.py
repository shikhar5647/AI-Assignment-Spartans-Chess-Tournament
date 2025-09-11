import sys
import copy
import time
from config import *
from board import GameEngine, Move

class B22CH032:
    """
    AI player using Minimax algorithm with Alpha-Beta pruning.
    Implements comprehensive evaluation function with piece values,
    positional bonuses, and tactical considerations.
    """
    
    def __init__(self, engine):
        self.board = engine
        self.nodes_expanded = 0
        self.depth = 4  # Optimal depth for 60-second games
        self.start_time = 0
        self.time_limit = 0.8  # Reserve time for other operations
        
        # Transposition table for memoization
        self.transposition_table = {}
        
        # Move ordering helpers
        self.killer_moves = [[] for _ in range(10)]  # Store killer moves per depth
        self.history_heuristic = {}
    
    def get_best_move(self):
        """
        Calculates and returns the best move using iterative deepening
        with Alpha-Beta pruning and various optimizations.
        """
        self.start_time = time.time()
        self.nodes_expanded = 0
        self.transposition_table.clear()
        
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        best_move = legal_moves[0]
        
        # Iterative deepening
        for depth in range(1, self.depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break
                
            try:
                current_best = self._iterative_search(legal_moves, depth)
                if current_best:
                    best_move = current_best
            except TimeoutError:
                break
        
        return best_move
    
    def _iterative_search(self, moves, depth):
        """Search with the given depth using iterative deepening."""
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(moves)
        
        for move in ordered_moves:
            if time.time() - self.start_time > self.time_limit:
                raise TimeoutError()
            
            self.board.make_move(move)
            score = self._minimax(depth - 1, alpha, beta, False)
            self.board.undo_move()
            
            if score > best_score:
                best_score = score
                best_move = move
            
            alpha = max(alpha, score)
            if beta <= alpha:
                break
        
        return best_move
    
    def _minimax(self, depth, alpha, beta, is_maximizing):
        """
        Minimax algorithm with Alpha-Beta pruning.
        """
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError()
        
        self.nodes_expanded += 1
        
        # Terminal conditions
        game_state = self.board.get_game_state()
        if depth == 0 or game_state != "ongoing":
            return self.evaluate_board()
        
        # Check transposition table
        board_hash = self._get_board_hash()
        if board_hash in self.transposition_table:
            return self.transposition_table[board_hash]
        
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            score = self.evaluate_board()
            self.transposition_table[board_hash] = score
            return score
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(legal_moves)
        
        if is_maximizing:
            max_score = float('-inf')
            for move in ordered_moves:
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                
                self.board.make_move(move)
                score = self._minimax(depth - 1, alpha, beta, False)
                self.board.undo_move()
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Store killer move
                    if depth < len(self.killer_moves):
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(move)
                            if len(self.killer_moves[depth]) > 2:
                                self.killer_moves[depth].pop(0)
                    break
            
            self.transposition_table[board_hash] = max_score
            return max_score
        else:
            min_score = float('inf')
            for move in ordered_moves:
                if time.time() - self.start_time > self.time_limit:
                    raise TimeoutError()
                
                self.board.make_move(move)
                score = self._minimax(depth - 1, alpha, beta, True)
                self.board.undo_move()
                
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    # Store killer move
                    if depth < len(self.killer_moves):
                        if move not in self.killer_moves[depth]:
                            self.killer_moves[depth].append(move)
                            if len(self.killer_moves[depth]) > 2:
                                self.killer_moves[depth].pop(0)
                    break
            
            self.transposition_table[board_hash] = min_score
            return min_score
    
    def _order_moves(self, moves):
        """
        Order moves for better Alpha-Beta pruning efficiency.
        Priority: captures, checks, killer moves, then by piece value.
        """
        scored_moves = []
        
        for move in moves:
            score = 0
            
            # Prioritize captures (MVV-LVA: Most Valuable Victim - Least Valuable Attacker)
            if move.piece_captured != EMPTY_SQUARE:
                victim_value = abs(PIECE_VALUES.get(move.piece_captured, 0))
                attacker_value = abs(PIECE_VALUES.get(move.piece_moved, 0))
                score += victim_value * 10 - attacker_value
            
            # Check if move gives check
            self.board.make_move(move)
            if self.board.is_in_check():
                score += 50
            self.board.undo_move()
            
            # Killer move heuristic
            for depth_moves in self.killer_moves:
                if move in depth_moves:
                    score += 30
                    break
            
            # History heuristic
            move_key = (move.start_row, move.start_col, move.end_row, move.end_col)
            score += self.history_heuristic.get(move_key, 0)
            
            # Positional bonuses
            score += self._get_positional_bonus(move)
            
            scored_moves.append((score, move))
        
        # Sort by score in descending order
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
    
    def _get_positional_bonus(self, move):
        """Get positional bonus for a move."""
        bonus = 0
        piece_type = move.piece_moved[1]
        end_row, end_col = move.end_row, move.end_col
        
        # Use piece-square tables
        if piece_type == 'P':
            if move.piece_moved.startswith('w'):
                bonus += PAWN_PST[end_row][end_col]
            else:
                bonus -= PAWN_PST[7 - end_row][end_col]
        elif piece_type == 'N':
            bonus += KNIGHT_PST[end_row][end_col] * (1 if move.piece_moved.startswith('w') else -1)
        elif piece_type == 'B':
            bonus += BISHOP_PST[end_row][end_col] * (1 if move.piece_moved.startswith('w') else -1)
        
        return bonus
    
    def evaluate_board(self):
        """
        Comprehensive board evaluation function.
        Returns a score from White's perspective (positive = good for White).
        """
        game_state = self.board.get_game_state()
        
        # Terminal positions
        if game_state == "checkmate":
            return -99999 if self.board.white_to_move else 99999
        elif game_state == "stalemate":
            return 0
        
        score = 0
        
        # Material evaluation
        score += self._evaluate_material()
        
        # Positional evaluation
        score += self._evaluate_positions()
        
        # King safety
        score += self._evaluate_king_safety()
        
        # Piece activity and mobility
        score += self._evaluate_mobility()
        
        # Pawn structure
        score += self._evaluate_pawn_structure()
        
        # Center control
        score += self._evaluate_center_control()
        
        # Endgame considerations
        if self._is_endgame():
            score += self._evaluate_endgame()
        
        return score
    
    def _evaluate_material(self):
        """Evaluate material balance using piece values."""
        score = 0
        for row in self.board.board:
            for piece in row:
                if piece != EMPTY_SQUARE:
                    score += PIECE_VALUES.get(piece, 0)
        return score
    
    def _evaluate_positions(self):
        """Evaluate piece positions using piece-square tables."""
        score = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                
                piece_type = piece[1]
                is_white = piece.startswith('w')
                
                if piece_type == 'P':
                    pst_value = PAWN_PST[r][c] if is_white else PAWN_PST[7-r][c]
                    score += pst_value if is_white else -pst_value
                elif piece_type == 'N':
                    pst_value = KNIGHT_PST[r][c]
                    score += pst_value if is_white else -pst_value
                elif piece_type == 'B':
                    pst_value = BISHOP_PST[r][c]
                    score += pst_value if is_white else -pst_value
                elif piece_type == 'K' and self._is_endgame():
                    pst_value = KING_PST_LATE_GAME[r][c]
                    score += pst_value if is_white else -pst_value
        
        return score
    
    def _evaluate_king_safety(self):
        """Evaluate king safety."""
        score = 0
        
        white_king_pos = self.board._find_king('w')
        black_king_pos = self.board._find_king('b')
        
        if white_king_pos and black_king_pos:
            # Penalize exposed kings in the middle game
            if not self._is_endgame():
                # White king safety
                if white_king_pos[0] < 6:  # King advanced too early
                    score -= 30
                
                # Black king safety
                if black_king_pos[0] > 1:  # King advanced too early
                    score += 30
        
        return score
    
    def _evaluate_mobility(self):
        """Evaluate piece mobility and activity."""
        score = 0
        
        # Current player mobility
        current_moves = len(self.board.get_legal_moves())
        
        # Switch turn to evaluate opponent mobility
        self.board.white_to_move = not self.board.white_to_move
        opponent_moves = len(self.board.get_legal_moves())
        self.board.white_to_move = not self.board.white_to_move
        
        mobility_diff = current_moves - opponent_moves
        if self.board.white_to_move:
            score += mobility_diff * 2
        else:
            score -= mobility_diff * 2
        
        return score
    
    def _evaluate_pawn_structure(self):
        """Evaluate pawn structure."""
        score = 0
        
        white_pawns = []
        black_pawns = []
        
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == WHITE_PAWN:
                    white_pawns.append((r, c))
                elif piece == BLACK_PAWN:
                    black_pawns.append((r, c))
        
        # Reward passed pawns
        for r, c in white_pawns:
            if self._is_passed_pawn(r, c, 'w'):
                score += 15 + (6 - r) * 5  # More reward for advanced passed pawns
        
        for r, c in black_pawns:
            if self._is_passed_pawn(r, c, 'b'):
                score -= 15 + (r - 1) * 5
        
        return score
    
    def _evaluate_center_control(self):
        """Evaluate control of central squares."""
        score = 0
        center_squares = [(3, 1), (3, 2), (4, 1), (4, 2)]
        
        for r, c in center_squares:
            piece = self.board.board[r][c]
            if piece != EMPTY_SQUARE:
                if piece.startswith('w'):
                    score += 10
                else:
                    score -= 10
        
        return score
    
    def _evaluate_endgame(self):
        """Evaluate endgame-specific factors."""
        score = 0
        
        white_king_pos = self.board._find_king('w')
        black_king_pos = self.board._find_king('b')
        
        if white_king_pos and black_king_pos:
            # King activity in endgame
            white_king_centralization = self._king_centralization_score(white_king_pos)
            black_king_centralization = self._king_centralization_score(black_king_pos)
            score += white_king_centralization - black_king_centralization
        
        return score
    
    def _is_endgame(self):
        """Determine if we're in the endgame."""
        piece_count = 0
        for row in self.board.board:
            for piece in row:
                if piece != EMPTY_SQUARE and piece[1] != 'K':
                    piece_count += 1
        return piece_count <= 8  # Endgame when 8 or fewer pieces remain
    
    def _is_passed_pawn(self, row, col, color):
        """Check if a pawn is passed (no enemy pawns blocking its path)."""
        if color == 'w':
            for r in range(row - 1, -1, -1):
                for c in range(max(0, col - 1), min(BOARD_WIDTH, col + 2)):
                    if self.board.board[r][c] == BLACK_PAWN:
                        return False
        else:
            for r in range(row + 1, BOARD_HEIGHT):
                for c in range(max(0, col - 1), min(BOARD_WIDTH, col + 2)):
                    if self.board.board[r][c] == WHITE_PAWN:
                        return False
        return True
    
    def _king_centralization_score(self, king_pos):
        """Calculate king centralization score for endgame."""
        r, c = king_pos
        center_distance = abs(r - 3.5) + abs(c - 1.5)
        return int(10 - center_distance * 2)
    
    def _get_board_hash(self):
        """Generate a hash of the current board position."""
        board_tuple = tuple(tuple(row) for row in self.board.board)
        return hash((board_tuple, self.board.white_to_move))