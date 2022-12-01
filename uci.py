#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import division
import importlib
import re
import sys
import time
import logging
import argparse

import tools
import numbfish
import math

from tools import WHITE, BLACK, Unbuffered

PLAY_5min = True

def is_end_game(pos, nnue_eval):
    major_pieces_ours = pos.board.count('R') + pos.board.count('Q') + pos.board.count('N')
    major_pieces_opp = pos.board.count('r') + pos.board.count('q') + pos.board.count('n')
    pieces = sum(c.isalpha() for c in pos.board)
    return (nnue_eval > 1300 and major_pieces_opp <= 1) or (nnue_eval < -1300 and major_pieces_ours <= 1) or pieces <= 8

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('module', help='numbfish.py file (without .py)', type=str, default='numbfish', nargs='?')
    parser.add_argument('--tables', metavar='pst', help='alternative pst table', type=str, default=None)
    args = parser.parse_args()

    numbfish = importlib.import_module(args.module)
    if args.tables is not None:
        pst_module = importlib.import_module(args.tables)
        numbfish.pst = pst_module.pst

    logging.basicConfig(filename='numbfish.log', level=logging.DEBUG)
    out = Unbuffered(sys.stdout)
    def output(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()
    pos = tools.parseFEN(tools.FEN_INITIAL)
    searcher = numbfish.Searcher()
    color = WHITE
    our_time, opp_time = 200000, 200000 # time in centi-seconds
    show_thinking = True

    stack = []
    moveslist = []
    history = set()
    use_classical = False
    while True:
        if stack:
            smove = stack.pop()
        else:
            smove = input()

        logging.debug(f'>>> {smove} ')

        if smove == 'quit':
            break

        elif smove == 'uci':
            output('id name Numbfish')
            output('id author Dimitrios Danopoulos')
            output('uciok')
            
        elif smove == 'isready':
            output('readyok')

        elif smove == 'ucinewgame':
            stack.append('position fen ' + tools.FEN_INITIAL)
            use_classical = False
            history = set()

        # syntax specified in UCI
        # position [fen  | startpos ]  moves  ....

        elif smove.startswith('position'):
            params = smove.split(' ')
            idx = smove.find('moves')

            if idx >= 0:
                moveslist = smove[idx:].split()[1:]
            else:
                moveslist = []

            if params[1] == 'fen':
                if idx >= 0:
                    fenpart = smove[:idx]
                else:
                    fenpart = smove

                _, _, fen = fenpart.split(' ', 2)

            elif params[1] == 'startpos':
                fen = tools.FEN_INITIAL
            else:
                pass

            pos = tools.parseFEN(fen)
            color = WHITE if fen.split()[1] == 'w' else BLACK

            for move in moveslist:
                pos = pos.move(tools.mparse(color, move))
                color = 1 - color    

        elif smove.startswith('go'):
            #  default options
            depth = 20
            movetime = 20000
            moves_remain = 60
            our_time = 300000
            
            
            _, *params = smove.split(' ')
            for param, val in zip(*2*(iter(params),)):
                if param == 'depth':
                    depth = int(val)
                if param == 'movetime':
                    movetime = int(val)
                if param == 'wtime':
                    our_time = int(val)
                if param == 'btime':
                    opp_time = int(val)
            
            our_time = opp_time if pos.board.startswith('\n') else our_time
            
            if(PLAY_5min):
                #add current position to history
                history.add(pos.board)
                
                if(our_time < 60000):
                    #maybe overkill but i liked this curve below 60s
                    movetime = 1000*(1.6/(1 + math.exp(-0.09*((our_time/1000)-40))))  
                else:
                    movetime = 1900
            
            start = time.time()
            ponder = None
            
            for sdepth, _move, _score, from_book in searcher.search(pos, movetime, use_classical, history):
                if(from_book):
                    output('bestmove ' + str(_move))
                    continue
                    
                moves = tools.pv(searcher, pos, include_scores=False)

                if show_thinking:
                    entry = searcher.tp_score.get((pos, sdepth, True))
                    score = int(round((entry.lower + entry.upper)/2))
                    usedtime = int((time.time() - start) * 1000)
                    moves_str = moves if len(moves) < 15 else moves[0:25]
                    output('info depth {} score cp {} time {} nodes {} nps {} pv {}'.format(sdepth, score, usedtime, searcher.nodes,  (searcher.nodes*1000)//usedtime, moves_str))

                if len(moves) > 5:
                    ponder = moves[1]
                
                if movetime > 0 and (time.time() - start) * 1000 > movetime:
                    break
                '''
                if (time.time() - start) * 1000 > our_time/moves_remain:
                    break
                '''
                
                if sdepth >= depth:
                    break
            if(not from_book):
                entry = searcher.tp_score.get((pos, sdepth, True))
                m, s = searcher.tp_move.get(pos), entry.lower

                if(is_end_game(pos,s)):
                    use_classical = True
                    print('searching with classical eval...')
                #if s == -numbfish.MATE_UPPER:
                #    output('resign')
                #else:
                moves = moves.split(' ')
                if len(moves) > 1:
                    output(f'bestmove {moves[0]} ponder {moves[1]}')
                else:
                    output('bestmove ' + moves[0])

        elif smove.startswith('time'):
            our_time = int(smove.split()[1])

        elif smove.startswith('otim'):
            opp_time = int(smove.split()[1])

        else:
            pass

if __name__ == '__main__':
    main()

