import numpy as np
import struct
import chess
import chess.polyglot
import functools
from enum import Enum
from enum import IntFlag

from tflite_runtime.interpreter import Interpreter

FEATURE_TRANSFORMER_HALF_DIMENSIONS = 256
DENSE_LAYERS_WIDTH = 32

SQUARE_NB = 64

class PieceSquare(IntFlag):
	NONE     =  0,
	W_PAWN   =  1,
	B_PAWN   =  1 * SQUARE_NB + 1
	W_KNIGHT =  2 * SQUARE_NB + 1
	B_KNIGHT =  3 * SQUARE_NB + 1
	W_BISHOP =  4 * SQUARE_NB + 1
	B_BISHOP =  5 * SQUARE_NB + 1
	W_ROOK   =  6 * SQUARE_NB + 1
	B_ROOK   =  7 * SQUARE_NB + 1
	W_QUEEN  =  8 * SQUARE_NB + 1
	B_QUEEN  =  9 * SQUARE_NB + 1
	W_KING   = 10 * SQUARE_NB + 1
	END      = W_KING # pieces without kings (pawns included)
	B_KING   = 11 * SQUARE_NB + 1
	END2     = 12 * SQUARE_NB + 1

	def from_piece(p: chess.Piece, is_white_pov: bool):
		return {
		chess.WHITE: {
			chess.PAWN: PieceSquare.W_PAWN,
			chess.KNIGHT: PieceSquare.W_KNIGHT,
			chess.BISHOP: PieceSquare.W_BISHOP,
			chess.ROOK: PieceSquare.W_ROOK,
			chess.QUEEN: PieceSquare.W_QUEEN,
			chess.KING: PieceSquare.W_KING
		},
		chess.BLACK: {
			chess.PAWN: PieceSquare.B_PAWN,
			chess.KNIGHT: PieceSquare.B_KNIGHT,
			chess.BISHOP: PieceSquare.B_BISHOP,
			chess.ROOK: PieceSquare.B_ROOK,
			chess.QUEEN: PieceSquare.B_QUEEN,
			chess.KING: PieceSquare.B_KING
		}
		}[p.color == is_white_pov][p.piece_type]

    
def orient(is_white_pov: bool, sq: int):
	# Use this one for "flip" instead of "rotate"
	# return (chess.A8 * (not is_white_pov)) ^ sq
	return (63 * (not is_white_pov)) ^ sq

def make_halfkp_index(is_white_pov: bool, king_sq: int, sq: int, p: chess.Piece):
	return orient(is_white_pov, sq) + PieceSquare.from_piece(p, is_white_pov) + PieceSquare.END * king_sq

interpreter = Interpreter(model_path='nnue_data/hidden_layers.tflite', num_threads=1) #use 1-thread for low latency  
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.allocate_tensors()

transformer_weights = np.load('nnue_data/transformer_weights.npy')
transformer_bias = np.load('nnue_data/transformer_bias.npy')

def get_halfkp_indices(board: chess.Board):
	result = []
	is_white_pov = board.turn
	for i, turn in enumerate([board.turn, not board.turn]):
		indices = []
		for sq, p in board.piece_map().items():
			if p.piece_type == chess.KING:
				continue
			indices.append(make_halfkp_index(turn, orient(turn, board.king(turn)), sq, p))
		result.append(indices)

	return np.array(result, dtype=np.intp)

def accum_update(accum, turn, move, kings):

    from_sq = move[0]
    from_pc = move[1]
    to_sq = move[2]
    to_pc = move[3]
    (our_king, opp_king) = (kings[0], kings[1]) if turn else (kings[1], kings[0])
    
    new_accum = np.empty_like(accum)

    piece_index_from_w = make_halfkp_index(turn, orient(turn, our_king), from_sq, from_pc)
    piece_index_from_b = make_halfkp_index(not turn, orient(not turn, opp_king), from_sq, from_pc)
    
    if(to_pc is not None): #empty destination square.
        piece_index_to_b = make_halfkp_index(not turn, orient(not turn, opp_king), to_sq, to_pc)
        piece_index_to_w = make_halfkp_index(turn, orient(turn, our_king), to_sq, to_pc)

    if((from_pc.piece_type == chess.PAWN) and (to_sq > 55)):     # white pawn reached final rank      
        piece_index_to_w_new = make_halfkp_index(turn, orient(turn, our_king), to_sq,  chess.Piece(chess.QUEEN, chess.WHITE))
        piece_index_to_b_new = make_halfkp_index(not turn, orient(not turn, opp_king), to_sq,  chess.Piece(chess.QUEEN, chess.WHITE))
            
    elif((from_pc.piece_type == chess.PAWN) and (to_sq < 8)):  # black pawn reached final rank
        piece_index_to_w_new = make_halfkp_index(turn, orient(turn, our_king), to_sq,  chess.Piece(chess.QUEEN, chess.BLACK))
        piece_index_to_b_new = make_halfkp_index(not turn, orient(not turn, opp_king), to_sq,  chess.Piece(chess.QUEEN, chess.BLACK))
    else:
        piece_index_to_w_new = make_halfkp_index(turn, orient(turn, our_king), to_sq, from_pc)
        piece_index_to_b_new = make_halfkp_index(not turn, orient(not turn, opp_king), to_sq, from_pc)
        
    new_accum[0][0:256] = accum[0][256:512] + transformer_weights[piece_index_to_b_new] - transformer_weights[piece_index_from_b]
    new_accum[0][256:512] = accum[0][0:256] + transformer_weights[piece_index_to_w_new] - transformer_weights[piece_index_from_w]
    if(to_pc is not None):
        new_accum[0][0:256] -= transformer_weights[piece_index_to_b]
        new_accum[0][256:512] -= transformer_weights[piece_index_to_w]

    return new_accum

    
