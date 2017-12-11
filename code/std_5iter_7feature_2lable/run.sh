#!/bin/bash
python ../preprocess.py train.txt
python ../preprocess.py test.txt
python perceptron.py ../pro_train.txt ../pro_test.txt
perl ../benchmark/score ../perl_word_list.txt ../test.answer.txt mytest.answer.txt > output_score.txt
