import tensorflow as tf
from konlpy.tag import Okt
from gensim.models import Word2Vec
import numpy as np

okt = Okt()

word2vec = Word2Vec.load("./model/word2vec.model")

def embedding(word):
    if word in word2vec.wv.vocab:
        return word2vec.wv[word]
    else:
        return np.random.normal(size=300)

sess = tf.Session()
path = "./model/BiLSTM"
saver = tf.train.import_meta_graph(path+".meta")
saver.restore(sess, path)
graph = tf.get_default_graph()

print("Model was loaded")

input_size = 300
max_length = 100
batch_size = 1

input_X = graph.get_tensor_by_name("BiLSTM/input_X:0")
input_y = graph.get_tensor_by_name("BiLSTM/input_y:0")
dropout_keep_prob = graph.get_tensor_by_name("BiLSTM/dropout_keep_prob:0")
output = graph.get_tensor_by_name("BiLSTM/output:0")
seq_len = graph.get_tensor_by_name("BiLSTM/seq_len:0")
acc = graph.get_tensor_by_name("BiLSTM/acc:0")

while True:
    X = input("리뷰를 입력하세요.(exit => 종료)")
    if X == "exit":
        break
    X = [X]
    token_X = [["/".join(tag) for tag in okt.pos(sentence, norm=True, stem=True)] for sentence in X]
    batch_X = [[embedding(word) for word in sentence] for sentence in token_X]
    batch_X_padded = np.zeros(shape=(batch_size, max_length, input_size))
    for b in range(batch_size):
        batch_X_padded[b, :len(batch_X[b])] = batch_X[b]
    seq_len_ = [len(x) for x in X]
    feed_dict = {
        input_X: batch_X_padded,
        dropout_keep_prob: 1.0,
        seq_len: seq_len_
    }
    pred = sess.run(output, feed_dict=feed_dict)
    print("리뷰 \"{}\" 은 약 {}%의 확률로 긍정 리뷰입니다.".format(X[0], int(pred[0]*100)))