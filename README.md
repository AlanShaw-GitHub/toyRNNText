# toyRNNText

Requirements:

- python  >= 3.6
- tensorflow >= 1.10
- jieba
- gensim
- numpy
- pickle

Shutouts:

./demo.jpg

This is a toy implementation of a common **text classification** model called RNNText.

It uses the end2end architecture which takes a sentence as input, and directly predicts the labels it belongs to.

Differ from the traditional methods like SVM etc. It uses neural networks to encode the huge information and corelations between sentences and corresponding tags.

The model is extremely simple(main model part takes less than 50 lines), we argue that the results mainly achieved by tuning the hyper-parameters and empirical tricks.

We also found that adding L2 normalization punishment to the final loss function significantly benefits the results on valid set, it's probably because the neural-network-like models easily get overfitted on the training set.

The original dataset is from NLPCC website, check this link:http://tcci.ccf.org.cn/conference/2018/taskdata.php

The word embedding use pretrained Google word2vec model on open source wikipedia(chinese) dumps, and is fine-tuned during the training process, which also benefits the results on valid set.

I will release the pretained model on 100k sentences(10k different  labels) and the preprocessed data(also 100k ,pickle format).Note that the original dataset contains over 700k sentences(20k labels) .

To use the pretained model(100k), you need to first download the cleaned dataset and tensorflow checkpoint on www.freedomworld.cn/toyRNNText , then put the dataset on root path(./) , and the checkpoints on ./model_path_large.