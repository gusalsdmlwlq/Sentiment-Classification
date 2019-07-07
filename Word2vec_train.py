import Word2vec
from konlpy.tag import Okt
import time
import numpy as np

def read_data(path):
    f = open(path, encoding="utf-8")
    data = [line.split("\t")[1:] for line in f.read().splitlines()]
    return data[1:]

start = time.time()
train_data = read_data("./data/ratings_train_txt")
test_data = read_data("./data/ratings_test_txt")
end = time.time()
print("Data was read : {} seconds".format(end-start))

def delete_null(data):
    for idx, d in enumerate(data):
        if d[0] == "":
            del data[idx]

delete_null(train_data)
delete_null(test_data)

okt = Okt()

def tokenize(data):
    return [["/".join(tag) for tag in okt.pos(sentence[0], norm=True, stem=True)] for sentence in data]

print("Tokenizing data ...")
start = time.time()
train_tokens = tokenize(train_data)
test_tokens = tokenize(test_data)
end = time.time()
print("Tokenizing finished : {} seconds".format(end-start))

word2vec = Word2vec.Word2vec()
word2vec.build(train_tokens)

word2vec.train(train_tokens, 30)

print("Preprocessing data ...")
start = time.time()
train_X = [[word2vec.embedding(word) for word in sentence] for sentence in train_tokens]
train_y = [[int(data[1])] for data in train_data]
test_X = [[word2vec.embedding(word) for word in sentence] for sentence in test_tokens]
test_y = [[int(data[1])] for data in test_data]
np.save("./data/train_X.npy", train_X)
np.save("./data/train_y.npy", train_y)
np.save("./data/test_X.npy", test_X)
np.save("./data/test_y.npy", test_y)
end = time.time()
print("Preprocessing finished : {} seconds".format(end-start))