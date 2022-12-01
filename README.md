<img src="https://github.com/dimdano/numbfish/blob/main/logo/numbfish_logo.jpg" width=60% height=60%>

## Introduction
Numbfish is a simple but strong pythonic chess engine. Numbfish is based on Sunfish but with several additional features, the most important of which is an Efficiently Updatable Neural Network (NNUE) implemented and optimized first time using numpy. 
NNUE is *efficiently* implemented using incremental updates of the input layer outputs in make and unmake moves just like Stockfish does in C++. The additional positional information entailed from NNUE makes this engine *probably* **the strongest python engine** running on 1-thread CPU.


## About Numbfish
 Numbfish keeps a very simple but optimized python interface, taking up just 140 lines of code! (see [`compressed.py`](https://github.com/dimdano/numbfish/blob/master/compressed.py)) <br /> Yet it plays at 2300 ratings at Lichess. You can challenge [numbfish_bot](https://lichess.org/@/numbfish_bot) for 5+0 min games!

Because Numbfish is small and strives to be simple, the code provides a great platform for experimenting. My NNUE implementation is a bit complex to understand and a more explanatory guide is needed of how I made it work in python. However people can fork my project and experiment with other features as well.  Make sure to drop a :star:. If you make Numbfish stronger I will merge your improvements!


## Run it!

This project needs python >= 3.7. Also install the required packages with `pip install -r requirements.txt` <br />
Numbfish is optimized for 5+0 min games through `uci.py`. It communicates through UCI protocol by the command `python3 -u uci.py`. You can also play it through the chess GUI of you choice such as [cutechess](https://cutechess.com/).

If you want to just test it with UCI commands through terminal set PLAY_5min flag to False in `uci.py`.

Last, if you want to casually play from terminal run `python3 play_terminal.py` instead. <br />
*(just make sure to set OPENING_BOOK to False in `numbfish.py`)*

#### Interface from terminal play
<img src="https://github.com/dimdano/numbfish/blob/main/logo/terminal_play.png" width=40% height=40%>

## Features

1. Built around the simple, but deadly efficient MTD-bi search algorithm.
2. Filled with classic as well as modern 'chess engine tricks' for simpler and faster code.
3. Evaluation is based on smart NNUE inference as well as classic Piece Square Tables evaluation for end games.
4. Uses standard Python collections and data structures for clarity and efficiency.

## About NNUE implementation in python

<img src="https://github.com/dimdano/numbfish/blob/main/logo/HalfKP.png" width=70% height=70%>

Numbfish has a very efficient implementation of NNUE written with numpy operations along with neural network inference using [tf-lite](https://www.tensorflow.org/lite/). <br /> 
- First, the model of HalfKP structure was implemented in Tensorflow and the weights and biases of trained NNUE were loaded. Then the first half of the model up to the 2x256 layer output was implemented using fast numpy operations. The accumulator has a "white king" half and a "black king" half, where each half is a 256-element vector, which is equal to the sum of the weights of the "active" features plus the biases. The active features are gathered fast from the transformer weights using the indices of the pieces as the operations are sparse. Also an accum_update function was implemented to do incremental updates of the accumulator on new moves, changing only the indices of the changed pieces when needed. Although int16 operations can be used, no perfomance increase was observed so I left float32 datatypes. <br /> 
- The second part of the model which begins with a clipped ReLU function and 2x256 input, and ends with the final evaluation was implemented using tf-lite for very low latency. Here float16 datatype for tflite model was used. <br /> <br /> 

**Weights and biases are taken from* [nn-62ef826d1a6d.nnue](https://tests.stockfishchess.org/nns?network_name=nn-62ef826d1a6d.nnue&user=&master_only=on)


## Performance and Limitations 

Numbfish supports castling, en passant, and promotion. It doesn't however do minor promotions to rooks, knights or bishops.
Also, the NNUE inference (specifically the accumulator update) does not take into account the en passant cases. Left for future work.
Last, in chess matches the engine adds previous positions in history to avoid 3-fold repetitions. Although with this implementation it avoids 2-fold repetitions which might not be good.

Numbfish is strong when it runs on a fast CPU core especially when reaching >5 depths in games. Below are the results from 5+0 min games of Numbfish vs Sunfish running both on a i7-8700 CPU. <br /> 
It's worth mentioning that Numbfish achieves ~14K NPS, almost 1/4 of the NPS of Sunfish (~54K NPS), however it wins in most of the matches. Below are some winning percentages after several tests I did but more games are needed to have a final conclusion.

    Name                          Games   Wins   Draws
    Numbfish vs Sunfish           30      90%     7%



## How to make Numbfish stronger

- add dedicated capture generation or check detection
- add the en passant case for the accumulator update function in order to avoid blunders
- move everything to bitboards using chess library
- implement a more recent NNUE structure such as HalfKAv2
- implement parallel search

## Why Numbfish?

The name Numbfish actually refers to the [numbfish](https://en.wikipedia.org/wiki/Narcinidae), which has numb as first name and more or less refers to numpy. Although numba is a more similar word, I sinfully :smiling_imp: left the numba implementation as a future work (should give minor improvements though).
The use of a fish at the end is in the spirit of the great open source chess engines such as Stockfish!


## Credit

[Sunfish](https://github.com/thomasahle/sunfish)

[Stockfish](https://github.com/official-stockfish/Stockfish) 

[Stockfish NNUE](https://www.chessprogramming.org/Stockfish_NNUE)

[Chess Opening Books](https://sites.google.com/site/computerschess/download?pli=1)

*Feel free to contact me if you have any questions*
