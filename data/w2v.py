'''
word2vec生成sem_t的静态词向量，见colab
'''
from gensim.models import Word2Vec
import csv

#将sem_t变成列表句子..
text_li=[]
label_li=[]
with open('sem/sem_t.tsv', 'r', encoding='utf-8') as fi:
    rowes = csv.reader(fi, delimiter='\t')
    for row in rowes:
        text = row[0]
        text_li.append(text.split(" "))
        label = row[1]
        label_li.append(label)

#sem_t变成word2vec词向量
model = Word2Vec(sentences=text_li, vector_size=200, window=1, sample=0.001, seed=42, workers=1, epochs=15)
model.wv.save_word2vec_format('sem/sem_w2v_200.txt',binary = False)