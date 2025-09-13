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
        self.depth = 4  
        self.is_white = None  # Will be determined during first move
        # Transposition table for memoization
        self.transposition_table = {}
        # Move ordering helpers
        self.killer_moves = [[] for _ in range(10)]  # Store killer moves per depth
    
    def get_best_move(self):
        """
        Calculates and returns the best move using Minimax with Alpha-Beta pruning.
        """
        self.nodes_expanded = 0
        self.transposition_table.clear()
        
        # Determine our color on first move
        if self.is_white is None:
            self.is_white = self.board.white_to_move
        
        legal_moves = self.board.get_legal_moves()
        if not legal_moves:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]
        
        best_move = None
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        # Order moves for better pruning
        ordered_moves = self._order_moves(legal_moves)
        
        for move in ordered_moves:
            self.board.make_move(move)
            score = self._minimax(self.depth - 1, alpha, beta, False)
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
        """
        scored_moves = []
        
        for move in moves:
            score = 0
            
            # Prioritize captures (Most Valuable Victim - Least Valuable Attacker)
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
            
            scored_moves.append((score, move))
        
        # Sort by score in descending order
        scored_moves.sort(key=lambda x: x[0], reverse=True)
        return [move for score, move in scored_moves]
    
    def evaluate_board(self):
        """
        Board evaluation function that adapts to the player's color.
        Returns a score from the current player's perspective.
        """
        game_state = self.board.get_game_state()
        
        # Handle terminal positions
        if game_state == "checkmate":
            return -99999 if self.board.white_to_move == self.is_white else 99999
        elif game_state == "stalemate":
            return 0
        
        score = 0
        piece_count = 0
        
        # Count total pieces for endgame detection
        for row in self.board.board:
            for piece in row:
                if piece != EMPTY_SQUARE:
                    piece_count += 1
        
        is_endgame = piece_count <= 8 
        
        # Combined material and positional evaluation using PSTs
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                
                piece_type = piece[1]
                is_white_piece = piece.startswith('w')
                
                # 1. Material value
                piece_value = PIECE_VALUES.get(piece, 0)
                score += piece_value
                
                pst_value = 0
                
                if piece_type == 'P':
                    pst_value = PAWN_PST[r][c]
                elif piece_type == 'N':
                    pst_value = KNIGHT_PST[r][c]
                elif piece_type == 'B':
                    pst_value = BISHOP_PST[r][c]
                elif piece_type == 'K':
                    if is_endgame:
                        # Use endgame king table for better king activity
                        pst_value = KING_PST_LATE_GAME[r][c]
                    else:
                        # In opening/middlegame, prefer king safety (negative values for advanced positions)
                        pst_value = -KING_PST_LATE_GAME[r][c]
                
                # Apply PST value with correct sign for piece color
                if is_white_piece:
                    score += pst_value
                else:
                    score -= pst_value
        
        # 3. Mobility bonus - reward having more moves
        current_moves = len(self.board.get_legal_moves())
        
        # Switch turn to count opponent moves
        self.board.white_to_move = not self.board.white_to_move
        opponent_moves = len(self.board.get_legal_moves())
        self.board.white_to_move = not self.board.white_to_move
        
        mobility_bonus = (current_moves - opponent_moves) * 2
        if self.board.white_to_move:
            score += mobility_bonus
        else:
            score -= mobility_bonus
        
        # 4. For controlling the centre region . 
        center_squares = [(3, 1), (3, 2), (4, 1), (4, 2)]
        for r, c in center_squares:
            piece = self.board.board[r][c]
            if piece != EMPTY_SQUARE:
                if piece.startswith('w'):
                    score += 15
                else:
                    score -= 15
        
        # 5. Adjusting the score based on player's color
        if not self.is_white:
            score = -score
        
        return score
    
    def _get_board_hash(self):
        """Generate a hash of the current board position for transposition table."""
        board_tuple = tuple(tuple(row) for row in self.board.board)
        return hash((board_tuple, self.board.white_to_move))