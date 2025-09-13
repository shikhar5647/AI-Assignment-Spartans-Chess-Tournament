import sys
import copy
import time
from config import *
from board import *

class B23CM1036:
    """
    class for an AI player. Defines the common interface that all AI
    implementations must follow.
    Follow the naming convention shared before submission!!!
    """
    def __init__(self, board):
        self.board = board
        self.nodes_expanded = 0
        self.depth = 3 ## set depth as you see fit and use it further for your works. 

        
    def get_best_move(self):
        """
        Calculates and returns the best move for the current board state.
        This method must be implemented.
        """
        self.nodes_expanded = 0
        best_move = None
        is_maximizing_player = self.board.white_to_move

        if is_maximizing_player:
            best_value = -float('inf')
        else:
            best_value = float('inf')

        legal_moves = self.board.get_legal_moves()

        # Simple move ordering: check captures first. This significantly improves alpha-beta pruning.
        ordered_moves = sorted(legal_moves, key=lambda move: move.piece_captured != EMPTY_SQUARE, reverse=True)

        for move in ordered_moves:
            self.board.make_move(move)
            # The next turn belongs to the opponent, so we flip the 'is_maximizing_player' flag.
            board_value = self.minimax(self.depth - 1, -float('inf'), float('inf'), not is_maximizing_player)
            self.board.undo_move()

            if is_maximizing_player:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
            else:  # Minimizing player
                if board_value < best_value:
                    best_value = board_value
                    best_move = move
        
        # Fallback in case no best move is found (should only happen in rare edge cases)
        if best_move is None and legal_moves:
            return legal_moves[0]
            
        return best_move
    
    def minimax(self, depth, alpha, beta, is_maximizing_player):
        """
        Recursive helper function for the Minimax algorithm with Alpha-Beta pruning.
        """
        self.nodes_expanded += 1

        game_state = self.board.get_game_state()
        if depth == 0 or game_state != 'ongoing':
            return self.evaluate_board()

        legal_moves = self.board.get_legal_moves()
        # Prioritize captures for more effective pruning
        ordered_moves = sorted(legal_moves, key=lambda move: move.piece_captured != EMPTY_SQUARE, reverse=True)

        if is_maximizing_player:
            max_eval = -float('inf')
            for move in ordered_moves:
                self.board.make_move(move)
                evaluation = self.minimax(depth - 1, alpha, beta, False)
                self.board.undo_move()
                max_eval = max(max_eval, evaluation)
                alpha = max(alpha, evaluation)
                if beta <= alpha:
                    break  # Beta cutoff
            return max_eval
        else:  # Minimizing player
            min_eval = float('inf')
            for move in ordered_moves:
                self.board.make_move(move)
                evaluation = self.minimax(depth - 1, alpha, beta, True)
                self.board.undo_move()
                min_eval = min(min_eval, evaluation)
                beta = min(beta, evaluation)
                if beta <= alpha:
                    break  # Alpha cutoff
            return min_eval
        
    def evaluate_board(self):
        """
        Returns a heuristic score for the current board state.
        This method must be implemented basically the evaluation function for algorithms.
        """

        # Check for terminal states (checkmate/stalemate) first
        game_state = self.board.get_game_state()
        if game_state == 'checkmate':
            # If it's white to move, white is in checkmate, which is a win for black.
            return -float('inf') if self.board.white_to_move else float('inf')
        if game_state == 'stalemate':
            return 0

        total_score = 0
        for r in range(BOARD_HEIGHT):
            for c in range(BOARD_WIDTH):
                piece = self.board.board[r][c]
                if piece == EMPTY_SQUARE:
                    continue
                
                # 1. Material Score (from config.py)
                total_score += PIECE_VALUES.get(piece, 0)

                # 2. Positional Score using Piece-Square Tables (from config.py)
                piece_type = piece[1]
                piece_color = piece[0]
                
                pst = None
                if piece_type == 'P': pst = PAWN_PST
                elif piece_type == 'N': pst = KNIGHT_PST
                elif piece_type == 'B': pst = BISHOP_PST
                elif piece_type == 'K': pst = KING_PST_LATE_GAME
                
                if pst:
                    if piece_color == 'w':
                        total_score += pst[r][c]
                    else:  # Black piece
                        # For black, we flip the board vertically to use the same table
                        total_score -= pst[BOARD_HEIGHT - 1 - r][c]
        
        # Add a bonus for giving a check (+2 as per rules from the original assignment prompt)
        if self.board.is_in_check():
             # If white is to move and is in check, it's bad for white.
             # If black is to move and is in check, it's good for white.
            if self.board.white_to_move:
                total_score -= 2 # Penalty for being in check
            else:
                total_score += 2 # Bonus for putting the opponent in check

        return total_score
