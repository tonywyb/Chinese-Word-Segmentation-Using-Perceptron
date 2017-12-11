#encoding:utf-8
import sys
import codecs
f1 = codecs.open(sys.argv[1], "r", "utf-8")
sentence_lis = f1.readlines()
f1.close()
for i in range(len(sentence_lis)):
    sentence_lis[i] = sentence_lis[i].replace("\r",'')
f2 = codecs.open('pro_' + sys.argv[1], "w", "utf-8")
for line in sentence_lis:
    f2.write(line)
f2.close()