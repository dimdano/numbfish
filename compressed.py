class Position(namedtuple('Position', 'board score wc bc ep kp')):
    def gen_moves(self):
        for i, p in enumerate(self.board):
            if not p.isupper(): continue
            for d in directions[p]:
                for j in count(i+d, d):
                    q = self.board[j]
                    if q.isspace() or q.isupper(): break
                    if p == 'P' and d in (N, N+N) and q != '.': break
                    if p == 'P' and d == N+N and (i < A1+N or self.board[i+N] != '.'): break
                    if p == 'P' and d in (N+W, N+E) and q == '.' \
                            and j not in (self.ep, self.kp, self.kp-1, self.kp+1): break
                    yield (i, j)
                    if p in 'PNK' or q.islower(): break
                    if i == A1 and self.board[j+E] == 'K' and self.wc[0]: yield (j+E, j+W)
                    if i == H1 and self.board[j+W] == 'K' and self.wc[1]: yield (j+W, j+E)
    def rotate(self):
        return Position(self.board[::-1].swapcase(), -self.score, self.bc, self.wc, 119-self.ep if self.ep else 0, 119-self.kp if self.kp else 0)
    def nullmove(self):
        return Position(self.board[::-1].swapcase(), -self.score, self.bc, self.wc, 0, 0)
    def move(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        put = lambda board, i, p: board[:i] + p + board[i+1:]
        board = self.board
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        score = self.score + self.value(move)
        board = put(board, j, board[i])
        board = put(board, i, '.')
        if i == A1: wc = (False, wc[1])
        if i == H1: wc = (wc[0], False)
        if j == A8: bc = (bc[0], False)
        if j == H8: bc = (False, bc[1])
        if p == 'K':
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, A1 if j < i else H1, '.')
                board = put(board, kp, 'R')
        if p == 'P':
            if A8 <= j <= H8: board = put(board, j, 'Q')
            if j - i == 2*N: ep = i + N
            if j == self.ep: board = put(board, j+S, '.')
        return Position(board, score, wc, bc, ep, kp).rotate()
    def value(self, move):
        i, j = move
        p, q = self.board[i], self.board[j]
        score = pst[p][j] - pst[p][i]
        if q.islower():
            score += pst[q.upper()][119-j]
        if abs(j-self.kp) < 2:
            score += pst['K'][119-j]
        if p == 'K' and abs(i-j) == 2:
            score += pst['R'][(i+j)//2]
            score -= pst['R'][A1 if j < i else H1]
        if p == 'P':
            if A8 <= j <= H8:
                score += pst['Q'][j] - pst['P'][j]
            if j == self.ep:
                score += pst['P'][119-(j+S)]
        return score
Entry = namedtuple('Entry', 'lower upper')
class Searcher:
    def __init__(self):
        self.tp_score = {}
        self.tp_move = {}
        self.history = set()
        self.use_classical = False
    def bound(self, pos, gamma, depth, kings, accum_root, accum_up, pos_prev, move_prev, root=True):
        depth = max(depth, 0)
        if pos.score <= -MATE_LOWER: return -MATE_UPPER
        if not root and pos.board in self.history: return 0
        entry = self.tp_score.get((pos, depth, root), Entry(-MATE_UPPER, MATE_UPPER))
        if entry.lower >= gamma and (not root or self.tp_move.get(pos) is not None): return entry.lower
        if entry.upper < gamma: return entry.upper
        def moves():
            if depth > 0 and not root and any(c in pos.board for c in 'RBNQ'):
                yield None, -self.bound(pos.nullmove(), 1-gamma, depth-3, kings, cur_accum, accum_up, pos, None, root=False)
            if depth == 0:
                if(not self.use_classical):
                    interpreter.set_tensor(input_details[0]["index"], cur_accum)
                    interpreter.invoke()
                    score = ((((interpreter.get_tensor(output_details[0]["index"]))[0][0])// 16) * 100) // 208
                else: score = pos.score
                yield None, score
            killer = self.tp_move.get(pos)
            if killer and (depth > 0 or pos.value(killer) >= QS_LIMIT):
                yield killer, -self.bound(pos.move(killer), 1-gamma, depth-1, kings, cur_accum, pos.board[killer[0]] == 'K', pos, killer, root=False)
            for move in sorted(pos.gen_moves(), key=pos.value, reverse=True):
                if depth > 0 or pos.value(move) >= QS_LIMIT:
                    yield move, -self.bound(pos.move(move), 1-gamma, depth-1, kings, cur_accum, pos.board[move[0]] == 'K', pos, move, root=False)
        best = -MATE_UPPER
        cur_accum = np.empty([1, 512], dtype=np.float32)
        if(not self.use_classical):      
          if accum_up:                 
              board = chess.Board(tools.renderFEN(pos))
              turn = board.turn
              kings = (board.king(turn), board.king(not turn)) if turn else (board.king(not turn), board.king(turn))      
              ind = get_halfkp_indices(board)
              cur_accum[0][:256] = np.sum(transformer_weights[ind[0]],axis=0) 
              cur_accum[0][256:] = np.sum(transformer_weights[ind[1]],axis=0)  
              cur_accum[0][:256] += transformer_bias
              cur_accum[0][256:] += transformer_bias
              accum_up = False
          else: 
              if(move_prev):
                  move_chess = tools.chess_move_from_to(pos_prev, move_prev) 
                  turn = False if pos_prev.board.startswith('\n') else True
                  cur_accum = accum_update(accum_root, turn, move_chess, kings)
              else:                
                  cur_accum[0][0:256] = accum_root[0][256:]
                  cur_accum[0][256:] = accum_root[0][0:256]
        for move, score in moves():
            best = max(best, score)
            if best >= gamma:
                if len(self.tp_move) > TABLE_SIZE: self.tp_move.clear()
                self.tp_move[pos] = move
                break
        if best < gamma and best < 0 and depth > 0:
            is_dead = lambda pos: any(pos.value(m) >= MATE_LOWER for m in pos.gen_moves())
            if all(is_dead(pos.move(m)) for m in pos.gen_moves()):
                in_check = is_dead(pos.nullmove())
                best = -MATE_UPPER if in_check else 0
        if len(self.tp_score) > TABLE_SIZE: self.tp_score.clear()
        if best >= gamma:
            self.tp_score[pos, depth, root] = Entry(best, entry.upper)
        if best < gamma:
            self.tp_score[pos, depth, root] = Entry(entry.lower, best)
        return best
    def search(self, pos, movetime, use_classical = False, history=()):
        self.use_classical = use_classical
        for depth in range (1,100):
            lower, upper = -MATE_UPPER, MATE_UPPER
            while lower < upper - EVAL_ROUGHNESS:
                gamma = (lower+upper+1)//2   
                score = self.bound(pos, gamma, depth, None, np.empty([1, 512], dtype=np.float32) , True, pos, None)
                if score >= gamma: lower = score
                if score < gamma: upper = score  
            self.bound(pos, lower, depth, None, np.empty([1, 512], dtype=np.float32), True, pos, None)
            yield depth, self.tp_move.get(pos), self.tp_score.get((pos, depth, True)).lower, False
