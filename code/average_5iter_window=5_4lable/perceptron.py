#encoding: utf-8
import codecs
from collections import defaultdict
import numpy as np
import sys        
class Debug:
    def __init__(self, flag):
        self.flag = flag
        
    def debug(self, *l):
        if self.flag:
            l = map(str, l)
            print("\t".join(l))
        else:
            pass

class Perceptron:
    def __init__(self, debug_flag = True):
        self.weight = defaultdict(int)
        self.weight_delta = []

    def get_max_index(self, feature):
        max_res = -1
        max_index = -1
        lable = ['b', 'm', 'e', 's']
        for i in range(len(feature)):
            sum = 0.0
            for s in feature[i]:
                sum += self.weight[s]
            if sum > max_res:
                max_res = sum
                max_index = i
        return lable[max_index]

    def update_weights(self, feature, right_res, wrong_res):
        tmp_delta = defaultdict(int)
        if feature[right_res] == feature[wrong_res]:
            self.weight_delta.append(tmp_delta)
            return
        for i in feature[right_res]:
            self.weight[i] += 1
            tmp_delta[i] = 1
        for i in feature[wrong_res]:
            self.weight[i] -= 1
            tmp_delta[i] = -1
        self.weight_delta.append(tmp_delta)

class Test:
    def __init__(self, source_path, dest_path, p, debug_flag = True):
        self.debuger = Debug(debug_flag)
        self.p = p
        self.source_path = source_path
        self.dest_path = dest_path
        self.words = []
        self.feature = []
        self.res = []
        
    def get_words(self):
        f = codecs.open(self.source_path, "r", "utf-8")
        for line in f.readlines():
            sentence_word = ["B"]
            line  = line.strip().split()
            for l in line:
                for k in l:
                    sentence_word.append(k)
            sentence_word.append("E")
            self.words.append(sentence_word)

    def get_feature(self):
        for sentence_word in self.words:
            sentence_feature = []
            for i in range(1, len(sentence_word) - 1):
                word_feature = []
                lables = ['b', 'm', 'e', 's']
                for lable_index, lable in enumerate(lables):
                    feature_lis = []
                    if  i > 1 and i < len(sentence_word) - 2:
                        for unigram_tag, unigram_index in enumerate([-2, -1, 0, 1, 2]):
                            feature_lis.append(sentence_word[i + unigram_index] + '_' + \
                            str(unigram_tag) + '_' + lable)
                        for bigram_tag, bigram_index in enumerate([(-2, -1), 
                            (-1, 0), (0, 1), (1, 2)]):
                            feature_lis.append(sentence_word[i + bigram_index[0]] + '_' + \
                            sentence_word[i + bigram_index[1]] + '_' + str(bigram_tag) + '_' + lable)
                        for trigram_tag, trigram_index in enumerate([(-2, -1, 0), (-1, 0, 1), (0, 1, 2)]):
                            feature_lis.append(sentence_word[i + trigram_index[0]] + '_' + \
                            sentence_word[i + trigram_index[1]] + '_' + \
                            sentence_word[i + trigram_index[2]] + '_' + str(trigram_tag) + '_' + lable)
                        for quagram_tag, quagram_index in enumerate([(-2, -1, 0, 1),  (-1, 0, 1, 2)]):
                            feature_lis.append(sentence_word[i + quagram_index[0]] + '_' + \
                            sentence_word[i + quagram_index[1]] + '_' + \
                            sentence_word[i + quagram_index[2]] + '_' + \
                            sentence_word[i + quagram_index[3]] + '_' + str(quagram_tag) + '_' + lable)

                        feature_lis.append(sentence_word[i - 2] + '_' + sentence_word[i - 1] + '_' + \
                            sentence_word[i] + '_' + sentence_word[i + 1] + '_' + sentence_word[i + 2] + '_' + str(0) + '_' + lable)
                    else:
                        for unigram_tag, unigram_index in enumerate([-1, 0, 1]):
                            feature_lis.append(sentence_word[i + unigram_index] + '_' + \
                            str(unigram_tag) + '_' + lable)
                        for bigram_tag, bigram_index in enumerate([ (-1, 0), (-1, 1),(0, 1)]):
                            feature_lis.append(sentence_word[i + bigram_index[0]] + '_' + \
                            sentence_word[i + bigram_index[1]] + '_' + str(bigram_tag) + '_' + lable)
                        for trigram_tag, trigram_index in enumerate([(-1, 0, 1)]):
                            feature_lis.append(sentence_word[i + trigram_index[0]] + '_' + \
                            sentence_word[i + trigram_index[1]] + '_' + \
                            sentence_word[i + trigram_index[2]] + '_' + str(trigram_tag) + '_' + lable)
                    word_feature.append(feature_lis)       
                sentence_feature.append(word_feature)
            self.feature.append(sentence_feature)

    def get_perceptron_res(self):
        for i in range(len(self.feature)):
            sentence_res = [None]
            for j in range(len(self.feature[i])):
                sentence_res.append(self.p.get_max_index(self.feature[i][j]))
            sentence_res.append(None)
            self.res.append(sentence_res)

    def write_dest_file(self):
        f = codecs.open(self.dest_path, "w", "utf-8")
        for i in range(len(self.words)):
            for j in range(1, len(self.words[i]) - 2):
                f.write(self.words[i][j])
                if self.res[i][j] in ['s', 'e']:
                    f.write(' ')
            f.write(self.words[i][-2])
            f.write('\n')
        f.close()

class Train:
    def __init__(self, source_path, p, debug_flag = True):
        self.debuger = Debug(debug_flag)
        self.p = p
        self.source_path = source_path
        self.feature = []
        self.lable = []
        self.words = []

    def get_words_and_lables(self):
        f = codecs.open(self.source_path, "r", "utf-8")
        cnt = 0
        for line in f.readlines():
            sentence_words = []
            sentence_lable = []
            vocab = line.strip().split()
            sentence_words.append('B')
            sentence_lable.append(None)
            for v in vocab:
                if len(v) == 1:
                    sentence_lable.append('s')
                    sentence_words.append(v)
                elif len(v) == 2:
                    sentence_lable.append('b')
                    sentence_words.append(v[0])
                    sentence_lable.append('e')
                    sentence_words.append(v[1])
                else:
                    sentence_lable.append('b')
                    sentence_words.append(v[0])
                    for item in v[1:-1]:
                        sentence_lable.append('m')
                        sentence_words.append(item)
                    sentence_lable.append('e')
                    sentence_words.append(v[-1])                                      
            sentence_words.append('E')
            sentence_lable.append(None) 
            self.words.append(sentence_words)
            self.lable.append(sentence_lable) 
            assert sentence_words != sentence_lable
            self.debuger.debug("sentence", cnt, "getting word_lis and lable")
            cnt += 1
    
    def get_feature(self):
        for sentence_word in self.words:
            sentence_feature = []
            for i in range(1, len(sentence_word) - 1):
                word_feature = []
                lables = ['b', 'm', 'e', 's']
                for lable_index, lable in enumerate(lables):
                    feature_lis = []
                    if  i > 1 and i < len(sentence_word) - 2:
                        for unigram_tag, unigram_index in enumerate([-2, -1, 0, 1, 2]):
                            feature_lis.append(sentence_word[i + unigram_index] + '_' + \
                            str(unigram_tag) + '_' + lable)
                        for bigram_tag, bigram_index in enumerate([(-2, -1), 
                            (-1, 0), (0, 1), (1, 2)]):
                            feature_lis.append(sentence_word[i + bigram_index[0]] + '_' + \
                            sentence_word[i + bigram_index[1]] + '_' + str(bigram_tag) + '_' + lable)
                        for trigram_tag, trigram_index in enumerate([(-2, -1, 0), (-1, 0, 1), (0, 1, 2)]):
                            feature_lis.append(sentence_word[i + trigram_index[0]] + '_' + \
                            sentence_word[i + trigram_index[1]] + '_' + \
                            sentence_word[i + trigram_index[2]] + '_' + str(trigram_tag) + '_' + lable)
                        for quagram_tag, quagram_index in enumerate([(-2, -1, 0, 1),  (-1, 0, 1, 2)]):
                            feature_lis.append(sentence_word[i + quagram_index[0]] + '_' + \
                            sentence_word[i + quagram_index[1]] + '_' + \
                            sentence_word[i + quagram_index[2]] + '_' + \
                            sentence_word[i + quagram_index[3]] + '_' + str(quagram_tag) + '_' + lable)

                        feature_lis.append(sentence_word[i - 2] + '_' + sentence_word[i - 1] + '_' + \
                            sentence_word[i] + '_' + sentence_word[i + 1] + '_' + sentence_word[i + 2] + '_' + str(0) + '_' + lable)
                    else:
                        for unigram_tag, unigram_index in enumerate([-1, 0, 1]):
                            feature_lis.append(sentence_word[i + unigram_index] + '_' + \
                            str(unigram_tag) + '_' + lable)
                        for bigram_tag, bigram_index in enumerate([ (-1, 0), (-1, 1),(0, 1)]):
                            feature_lis.append(sentence_word[i + bigram_index[0]] + '_' + \
                            sentence_word[i + bigram_index[1]] + '_' + str(bigram_tag) + '_' + lable)
                        for trigram_tag, trigram_index in enumerate([(-1, 0, 1)]):
                            feature_lis.append(sentence_word[i + trigram_index[0]] + '_' + \
                            sentence_word[i + trigram_index[1]] + '_' + \
                            sentence_word[i + trigram_index[2]] + '_' + str(trigram_tag) + '_' + lable)
                    word_feature.append(feature_lis)       
                sentence_feature.append(word_feature)
            self.feature.append(sentence_feature)

    def train(self, iteration):
        self.iteration = iteration
        for k in range(iteration):
            self.debuger.debug("train iteration: ", k)
            for i in range(len(self.feature)):
                for j in range(len(self.feature[i])):
                    our_res = self.p.get_max_index(self.feature[i][j])
                    std_res = self.lable[i][j + 1]
                    delta = []
                    trans = { 'b':0, 'm':1, 'e':2, 's':3 }
                    our_res = trans[our_res]
                    std_res = trans[std_res]
                    assert our_res not in trans
                    assert std_res not in trans
                    self.p.update_weights(self.feature[i][j], std_res, our_res)
        average_weight = defaultdict(int)
        cnt = len(self.p.weight_delta)
        for w_delta in self.p.weight_delta:
            for j in w_delta:
                average_weight[j] += (1.0*cnt/len(self.p.weight_delta)) * w_delta[j]
            cnt -= 1
        self.p.weight = average_weight

if __name__ == "__main__":
    p = Perceptron()
    train_model = Train(sys.argv[1], p)
    print "train_model built!"
    train_model.get_words_and_lables()
    print "get words and lables from train_model!"
    train_model.get_feature()
    print "extract feature from train_model!"
    train_model.train(5)
    print "train phase complete"
    test_model = Test(sys.argv[2], "mytest.answer.txt", p)
    print "test_model built!"
    test_model.get_words()
    print "get test words!"
    test_model.get_feature()
    print "extract feature from test_model!"
    test_model.get_perceptron_res()
    print "get result from test_model!"
    test_model.write_dest_file()
    print "write back!"












                






