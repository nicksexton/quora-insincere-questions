# Quora Insincere Questions Classification 
Kaggle competition for NLP classification of Quora questions into into [sincere and insincere
](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/77691#456365). 

Disclaimer: none of this code will win you the competition, probably ;-) 


# Install instructions
0. (optional) create yourself a new, clean Anaconda environment with something like `conda create --name kaggle --clone myenv` or where myenv is the environment you use for deep learning (i.e. tensorflow, keras, GPU enabled if you have one)
1. clone this repo into an appropriate place using `git clone git@github.com:nicksexton/quora-insincere-questions`
2. `cd quora-insincere-questions`
2. (if you don't have it already) install kaggle CLI with `pip install kaggle`
3. download the dataset by running `kaggle competitions download quora-insincere-questions-classification`. Note that this will trigger a very large (~8GiB?) download including four word embeddings files so you could maybe tweak this (see [kaggle cli api](https://github.com/Kaggle/kaggle-api).
4. unzip the embeddings files
5. the directory structure should such that input directory sits alongside the kernel directories, i.e.: quora-insincere-questions/input/embeddings/... and quora-insincere-questions/attention-sandwich-model/... etc.
 

## Attention Sandwith Model
attention-sandwich-model/attention-sandwich-model.ipynb --- A number of the public kernels (e.g. this [forked model](https://www.kaggle.com/nicksexton/different-embeddings-with-attention-fork-fork)) use the same code for creating an attention layer that sits nicely on top of the two LSTMs that make up the recurrent part of the model. 

I wanted to dig around in the guts of how to implement an attention mechanism for type of NLP classification. In this notebook, I  implement an attention algorithm that sits between two LSTMs (or more, if we decide to continue stacking, why not) to create an attention sandwich. This has the advantage that for each point in its sequence, the second LSTM (which we'll call LSTM_q) is able to look at a much wider input (context) than just it's own hidden state and the current output of the previous LSTM (which we'll call LSTM_p), and means that the two LSTMs aren't necessarily aligned (i.e., they don't need to have the same number of timesteps).

The resulting model is quite slow to train but does get better results than the alternative Attention class in the public kernels, whether this is simply due to having more units in my LSTMs, I'm not sure, but I think it's worth persisting with.
