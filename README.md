# NLP期中大作业 汉语分词

###实验目标
本次大作业能够实现：

1. 利用非结构化感知机对中文分词
2. 利用结构化感知机对中文分词

*****************
###文件内容

1. benchmark文件夹（存放评测工具）
2. 两个用非平均化感知机实现分词的代码文件夹（标签数目不同）
3. 两个用平均化感知机实现分词的代码文件夹（特征个数不同）
4. 用结构化感知机实现分词的代码文件夹
5. readme.md
6. 若干实验训练与测试文件

**************
###运行流程
1. 进入上述5个代码文件夹，在linux终端键入,对实验数据预处理（去掉'\r'），生成pro\_train.txt、pro\_test.txt、pro\_test.answer.txt3个预处理后的文件
```
$ python ../preprocess.py train.txt
$ python ../preprocess.py test.txt
$ python ../preprocess.py test.answer.txt
```

2. 进入上述5个代码文件夹，在linux终端键入以下命令，可以在当前文件夹下得到分词结果，存放在mytest.answer.txt中
```
$ python perceptron.py ../pro_train.txt ../pro_test.txt
```

3. 评测环节， 在linux 终端键入以下命令，得到output_score.txt，文件结尾存放着分词的准确率、召回率和F1分数等评测信息
```
$ perl ../benchmark/score ../perl_word_list.txt ../test.answer.txt mytest.answer.txt > output_score.txt
``` 

4. 也可以直接运行脚本run.sh 执行上述所有操作
```
$ chmod u+x run.sh
$ ./run.sh
```
