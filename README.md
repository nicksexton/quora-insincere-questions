# Quora Insincere Questions Classification 
LSTM with Attention classifier models for Kaggle Quora Insincere Questions Classification competition dataset. See here for a definition of [sincere and insincere](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/77691#456365). 

Disclaimer: none of this code will win you the competition, probably ;-) 

## Attention Sandwich Model
attention-sandwich-model/attention-sandwich-model.ipynb --- A number of the [public kernels](https://www.kaggle.com/shujian/different-embeddings-with-attention-fork-fork) use the same code for creating a drop-on attention layer that sits nicely on top of the two LSTMs that make up the recurrent part of the model and does improve the performance of the LSTMs on this NLP classification task.

I wanted to dig around in the guts of how to implement an attention mechanism for NLP classification. In this notebook, I  implement an attention algorithm that sits between two LSTMs (or more, if we decide to continue stacking, why not) to create an attention sandwich, and serves to make the hidden states of the first LSTM for each timestep accessible to each timestep of the second LSTM, based on attention weights (which are also learned). This has the advantage that for each point in its sequence, the second LSTM (which we'll call, for no particular reason, LSTM_q) is able to look at a much wider input (context) than just it's own hidden state and the current output of the previous LSTM (which we'll call LSTM_p), and means that the two LSTMs aren't necessarily aligned (i.e., they don't need to have the same number of timesteps).

The resulting model is quite slow to train but without ensembling it does get better results than the alternative Attention class in the public kernels, whether this is simply due to having more units in my LSTMs, I'm not sure, but I think it's worth persisting with.

## Install instructions
0. (optional) create yourself a new, clean Anaconda environment with something like `conda create --name kaggle --clone myenv` or where myenv is the environment you use for deep learning (i.e. tensorflow, keras, GPU enabled if you have one)
1. clone this repo into an appropriate place using `git clone git@github.com:nicksexton/quora-insincere-questions`
2. `cd quora-insincere-questions`
2. (if you don't have it already) install kaggle CLI with `pip install kaggle`
3. download the dataset by running `kaggle competitions download quora-insincere-questions-classification`. Note that this will trigger a very large (~8GiB?) download including four word embeddings files so you could maybe tweak this (see [kaggle cli api](https://github.com/Kaggle/kaggle-api).
4. unzip the embeddings files
5. the directory structure should such that input directory sits alongside the kernel directories, i.e.: `quora-insincere-questions/input/embeddings/...` and `quora-insincere-questions/attention-sandwich-model/...` etc.
