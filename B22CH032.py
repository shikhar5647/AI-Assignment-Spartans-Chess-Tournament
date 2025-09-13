import sys
import copy
import time
from config import *
from board import Move
class B22CH032:
    """
    AI player using Minimax algorithm with Alpha-Beta pruning.
    """
    def __init__(self, board):
        self.board = board
        self.nodes_expanded = 0
        self.depth = 4  # Search depth
        
        self.transposition_table = {}
        
        # Move ordering helpers
        self.killer_moves = [[] for _ in range(10)]
        self.opening_moves = 0
    
    def get_best_move(self):
        """
        Calculates and returns the best move using Minimax with Alpha-Beta pruning.
        """
        self.nodes_expanded = 0
        
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        # Increment opening counter
        self.opening_moves += 1
        
        best_move = None
        best_score = float('-inf') if self.board.white_to_move else float('inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(legal_moves)
        
        for move in ordered_moves:
            self.board.make_move(move)
            score = self._minimax(self.depth - 1, alpha, beta, not self.board.white_to_move)
            self.board.undo_move()
            
            # Proper score comparison for both colors
            if self.board.white_to_move:
                if score > best_score:
                    best_score = score
                    best_move = move
                alpha = max(alpha, score)
            else:
                if score < best_score:
                    best_score = score
                    best_move = move
                beta = min(beta, score)
            if beta <= alpha:
                break
        return best_move
    
    def _minimax(self, depth, alpha, beta, is_maximizing):

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
                self.board.make_move(move)
                score = self._minimax(depth - 1, alpha, beta, False)
                self.board.undo_move()
                
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    self._store_killer_move(move, depth)
                    break
            self.transposition_table[board_hash] = max_score
            return max_score
        else:
            min_score = float('inf')
            for move in ordered_moves:
                self.board.make_move(move)
                score = self._minimax(depth - 1, alpha, beta, True)
                self.board.undo_move()
                
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    self._store_killer_move(move, depth)
                    break
            
            self.transposition_table[board_hash] = min_score
            return min_score
    
    def _store_killer_move(self, move, depth):
        """Helper to store killer moves."""
        if depth < len(self.killer_moves):
            if move not in self.killer_moves[depth]:
                self.killer_moves[depth].append(move)
                if len(self.killer_moves[depth]) > 2:
                    self.killer_moves[depth].pop(0)
    
    def _order_moves(self, moves):
        """
        Move ordering for better Alpha-Beta pruning efficiency.
        """
        scored_moves = []
        for move in moves:
            score = 0
            
            # Captures with improved evaluation
            if move.piece_captured != EMPTY_SQUARE:
                victim_value = abs(PIECE_VALUES.get(move.piece_captured, 0))
                attacker_value = abs(PIECE_VALUES.get(move.piece_moved, 0))
                # Enhanced capture scoring
                score += victim_value * 12 - attacker_value
                
                # Bonus for capturing with less valuable pieces
                if victim_value > attacker_value:
                    score += 50
            
            # Check bonus - more valuable in middlegame
            self.board.make_move(move)
            if self.board.is_in_check():
                score += 60 if self.opening_moves < 20 else 40
                
                # Extra bonus if check leads to mate threats
                opponent_moves = self.board.get_legal_moves()
                if len(opponent_moves) <= 3:
                    score += 30
            self.board.undo_move()
            
            # Killer move heuristic
            for depth_moves in self.killer_moves:
                if move in depth_moves:
                    score += 35
                    break
            
            # Enhanced positional scoring
            score += self._get_move_positional_bonus(move)
            
            # Opening principles for both colors
            if self.opening_moves < 15:
                score += self._get_opening_bonus(move)
            
            # Endgame bonuses
            piece_count = sum(1 for row in self.board.board for p in row if p != EMPTY_SQUARE and p[1] != 'K')
            if piece_count <= 8:
                score += self._get_endgame_bonus(move)
            
            scored_moves.append((score, move))
        
        # Sort by score in descending order
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
    
    def _get_move_positional_bonus(self, move):
        """Get positional bonus for a move."""
        bonus = 0
        piece_type = move.piece_moved[1]
        end_row, end_col = move.end_row, move.end_col
        is_white = move.piece_moved.startswith('w')
        
        # Center control bonus
        if end_row in [3, 4] and end_col in [1, 2]:
            bonus += 15
        
        # Piece-specific positional bonuses
        if piece_type == 'P':
            # Pawn advancement
            if is_white:
                bonus += (7 - end_row) * 3  # White pawns advance up
            else:
                bonus += end_row * 3        # Black pawns advance down
        
        elif piece_type == 'N':
            # Knights toward center
            center_distance = abs(end_row - 3.5) + abs(end_col - 1.5)
            bonus += int(10 - center_distance * 2)
        
        elif piece_type == 'B':
            # Bishops on long diagonals
            if (end_row + end_col) % 2 == 0:  # Same color diagonal
                bonus += 5
        
        elif piece_type == 'K':
            # King safety in opening/middlegame vs activity in endgame
            piece_count = sum(1 for row in self.board.board for p in row if p != EMPTY_SQUARE)
            if piece_count > 10:  # Not endgame
                # King safety - stay back
                if is_white and end_row > 5:
                    bonus += 10
                elif not is_white and end_row < 2:
                    bonus += 10
            else:  # Endgame
                # King activity - move toward center
                center_distance = abs(end_row - 3.5) + abs(end_col - 1.5)
                bonus += int(15 - center_distance * 3)
        
        return bonus
    
    def _get_opening_bonus(self, move):
        """Opening principles for both colors."""
        bonus = 0
        piece_type = move.piece_moved[1]
        
        # Develop pieces
        if piece_type in ['N', 'B']:
            bonus += 20
        
        # Control center early
        if move.end_row in [3, 4] and move.end_col in [1, 2]:
            bonus += 15
        
        # Don't move same piece twice in opening
        start_square = (move.start_row, move.start_col)
        if piece_type in ['N', 'B'] and self._piece_moved_from_square(start_square):
            bonus -= 25
        
        return bonus
    
    def _get_endgame_bonus(self, move):
        """Endgame-specific bonuses."""
        bonus = 0
        piece_type = move.piece_moved[1]
        
        # King activity
        if piece_type == 'K':
            center_distance = abs(move.end_row - 3.5) + abs(move.end_col - 1.5)
            bonus += int(20 - center_distance * 4)
        
        # Pawn promotion threats
        if piece_type == 'P':
            is_white = move.piece_moved.startswith('w')
            if (is_white and move.end_row <= 2) or (not is_white and move.end_row >= 5):
                bonus += 25
        
        return bonus
    
    def _piece_moved_from_square(self, square):
        """Check if a piece has moved from this square before (simplified)."""
        # Simple heuristic - could be enhanced with actual move history
        return len(self.board.move_log) > 5
    
    def evaluate_board(self):
        """
        Comprehensive board evaluation function optimized for both colors.
        Always returns from White's perspective (positive = White advantage).
        """
        game_state = self.board.get_game_state()
        
        # Handle terminal positions
        if game_state == "checkmate":
            return -99999 if self.board.white_to_move else 99999
        elif game_state == "stalemate":
            return 0
        
        score = 0
        
        # 1. Material evaluation - most important
        material_score = self._evaluate_material()
        score += material_score
        
        # 2. Positional evaluation with piece-square tables
        positional_score = self._evaluate_positional()
        score += positional_score
        
        # 3. King safety - crucial for both colors
        king_safety_score = self._evaluate_king_safety()
        score += king_safety_score
        
        # 4. Piece activity and mobility
        mobility_score = self._evaluate_mobility()
        score += mobility_score
        
        # 5. Pawn structure
        pawn_score = self._evaluate_pawn_structure()
        score += pawn_score
        
        # 6. Game phase specific bonuses
        phase_score = self._evaluate_game_phase()
        score += phase_score
        
        return score
    
    def _evaluate_material(self):
        """Enhanced material evaluation."""
        score = 0
        white_material = 0
        black_material = 0
        
        for row in self.board.board:
            for piece in row:
                if piece != EMPTY_SQUARE:
                    value = abs(PIECE_VALUES.get(piece, 0))
                    if piece.startswith('w'):
                        white_material += value
                        score += PIECE_VALUES.get(piece, 0)
                    else:
                        black_material += value
                        score += PIECE_VALUES.get(piece, 0)
        
        # Material imbalance bonus
        material_diff = white_material - black_material
        if abs(material_diff) > 100:  # Significant material advantage
            score += material_diff * 0.1  # Small bonus for material advantage
        
        return score
    
    def _evaluate_positional(self):
        """Enhanced positional evaluation using piece-square tables."""
        score = 0
        
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                
                piece_type = piece[1]
                is_white = piece.startswith('w')
                
                # Apply piece-square table values
                if piece_type == 'P':
                    if is_white:
                        pst_value = PAWN_PST[r][c]
                    else:
                        pst_value = PAWN_PST[7-r][c]  # Flip for black
                    score += pst_value if is_white else -pst_value
                
                elif piece_type == 'N':
                    pst_value = KNIGHT_PST[r][c]
                    score += pst_value if is_white else -pst_value
                
                elif piece_type == 'B':
                    pst_value = BISHOP_PST[r][c]
                    score += pst_value if is_white else -pst_value
                
                # Additional positional bonuses
                # Center control
                if r in [3, 4] and c in [1, 2]:
                    score += 8 if is_white else -8
        
        return score
    
    def _evaluate_king_safety(self):
        """Enhanced king safety evaluation for both colors."""
        score = 0
        
        white_king_pos = self.board._find_king('w')
        black_king_pos = self.board._find_king('b')
        
        if not white_king_pos or not black_king_pos:
            return score
        
        piece_count = sum(1 for row in self.board.board for p in row if p != EMPTY_SQUARE)
        
        if piece_count > 10:  # Not endgame - prioritize safety
            # White king safety
            if white_king_pos[0] < 6:  # Advanced king is dangerous
                score -= 25
            
            # Black king safety  
            if black_king_pos[0] > 1:  # Advanced king is dangerous
                score += 25
            
            # Penalty for kings in center during middlegame
            if white_king_pos[1] in [1, 2]:  # Center files
                score -= 15
            if black_king_pos[1] in [1, 2]:
                score += 15
        
        else:  # Endgame - prioritize activity
            # Reward active kings in endgame
            white_center_dist = abs(white_king_pos[0] - 3.5) + abs(white_king_pos[1] - 1.5)
            black_center_dist = abs(black_king_pos[0] - 3.5) + abs(black_king_pos[1] - 1.5)
            
            score += int(10 - white_center_dist * 2)
            score -= int(10 - black_center_dist * 2)
        
        return score
    
    def _evaluate_mobility(self):
        """Enhanced mobility evaluation."""
        # Count current player's moves
        current_moves = len(self.board.get_legal_moves())
        
        # Switch turn to count opponent moves
        self.board.white_to_move = not self.board.white_to_move
        opponent_moves = len(self.board.get_legal_moves())
        self.board.white_to_move = not self.board.white_to_move
        
        # Calculate mobility difference
        mobility_diff = current_moves - opponent_moves
        
        # Apply from White's perspective
        if self.board.white_to_move:
            return mobility_diff * 3
        else:
            return -mobility_diff * 3
    
    def _evaluate_pawn_structure(self):
        """Enhanced pawn structure evaluation."""
        score = 0
        
        white_pawns = []
        black_pawns = []
        
        # Collect pawn positions
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == WHITE_PAWN:
                    white_pawns.append((r, c))
                elif piece == BLACK_PAWN:
                    black_pawns.append((r, c))
        
        # Evaluate passed pawns
        for r, c in white_pawns:
            if self._is_passed_pawn(r, c, 'w'):
                advancement = 7 - r  # How far advanced (0-6)
                score += 20 + advancement * 5
        
        for r, c in black_pawns:
            if self._is_passed_pawn(r, c, 'b'):
                advancement = r  # How far advanced (0-6)
                score -= 20 + advancement * 5
        
        # Pawn chains (pawns protecting each other)
        score += self._evaluate_pawn_chains(white_pawns, black_pawns)
        
        return score
    
    def _evaluate_pawn_chains(self, white_pawns, black_pawns):
        """Evaluate pawn chains and pawn support."""
        score = 0
        
        # White pawn chains
        for r, c in white_pawns:
            # Check for supporting pawns
            for dr, dc in [(-1, -1), (-1, 1)]:  # Diagonally behind
                support_pos = (r - dr, c - dc)
                if support_pos in white_pawns:
                    score += 5
        
        # Black pawn chains
        for r, c in black_pawns:
            # Check for supporting pawns
            for dr, dc in [(1, -1), (1, 1)]:  # Diagonally behind
                support_pos = (r - dr, c - dc)
                if support_pos in black_pawns:
                    score -= 5
        
        return score
    
    def _evaluate_game_phase(self):
        """Game phase specific evaluation."""
        score = 0
        piece_count = sum(1 for row in self.board.board for p in row if p != EMPTY_SQUARE and p[1] != 'K')
        
        # Opening phase (many pieces)
        if piece_count > 12:
            # Reward piece development
            developed_pieces = 0
            for r in range(BOARD_HEIGHT):
                for c in range(BOARD_WIDTH):
                    piece = self.board.board[r][c]
                    if piece in [WHITE_KNIGHT, WHITE_BISHOP]:
                        if not ((r == 7 and c in [0, 3]) or (r == 7 and c == 1)):  # Not on starting squares
                            developed_pieces += 1
                    elif piece in [BLACK_KNIGHT, BLACK_BISHOP]:
                        if not ((r == 0 and c in [0, 3]) or (r == 0 and c == 1)):  # Not on starting squares
                            developed_pieces -= 1
            
            score += developed_pieces * 8
        
        # Endgame phase (few pieces)
        elif piece_count <= 8:
            # King centralization bonus already handled in king safety
            # Focus on pawn promotion
            for r in range(BOARD_HEIGHT):
                for c in range(BOARD_WIDTH):
                    piece = self.board.board[r][c]
                    if piece == WHITE_PAWN and r <= 3:  # Advanced white pawn
                        score += (4 - r) * 10
                    elif piece == BLACK_PAWN and r >= 4:  # Advanced black pawn
                        score -= (r - 3) * 10
        
        return score
    
    def _is_passed_pawn(self, row, col, color):
        """Enhanced passed pawn detection."""
        if color == 'w':
            # Check if any black pawns can stop this pawn
            for r in range(row - 1, -1, -1):
                for c in range(max(0, col - 1), min(BOARD_WIDTH, col + 2)):
                    if self.board.board[r][c] == BLACK_PAWN:
                        return False
        else:
            # Check if any white pawns can stop this pawn
            for r in range(row + 1, BOARD_HEIGHT):
                for c in range(max(0, col - 1), min(BOARD_WIDTH, col + 2)):
                    if self.board.board[r][c] == WHITE_PAWN:
                        return False
        return True
    
    def _get_board_hash(self):
        """Generate a hash of the current board position."""
        board_tuple = tuple(tuple(row) for row in self.board.board)
        return hash((board_tuple, self.board.white_to_move))