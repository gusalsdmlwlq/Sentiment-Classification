import tensorflow as tf
import BiLSTM
import numpy as np
import time

input_size = 300
max_length = 100
batch_size = 10000
path = "./model/BiLSTM"

test_X = np.load("./data/test_X.npy")
test_y = np.load("./data/test_y.npy")
print("Data was read")
batches = int(len(test_X)/batch_size)
test_sequence_length = [len(x) for x in test_X]

sess = tf.Session()
print("Session was created")
saver = tf.train.import_meta_graph(path+".meta")
saver.restore(sess, path)
graph = tf.get_default_graph()
print("Model was loaded")

input_X = graph.get_tensor_by_name("BiLSTM/input_X:0")
input_y = graph.get_tensor_by_name("BiLSTM/input_y:0")
sequence_length = graph.get_tensor_by_name("BiLSTM/sequence_length:0")
dropout_keep_prob = graph.get_tensor_by_name("BiLSTM/dropout_keep_prob:0")
output = graph.get_tensor_by_name("BiLSTM/output:0")
accuracy = graph.get_tensor_by_name("BiLSTM/accuracy:0")

accuracy_total = 0
steps = 0
for batch in range(batches):
    batch_X, batch_y = test_X[steps*batch_size:(steps+1)*batch_size], test_y[steps*batch_size:(steps+1)*batch_size]
    batch_sequence_length = test_sequence_length[steps*batch_size:(steps+1)*batch_size]
    batch_X_padded = np.zeros(shape=(batch_size, max_length, input_size))
    for b in range(batch_size):
        batch_X_padded[b, :len(batch_X[b])] = batch_X[b]

    feed_dict = {
        input_X: batch_X_padded,
        input_y: batch_y,
        sequence_length: batch_sequence_length,
        dropout_keep_prob: 1.0
    }

    accracy_batch = sess.run(accuracy, feed_dict=feed_dict)
    accuracy_total += accracy_batch
    steps += 1
    print("Batch {} Accuracy : {}".format(steps, accracy_batch))

if batch_size*steps < len(test_X):
    batch_X, batch_y = test_X[steps*batch_size:], test_y[steps*batch_size:]
    batch_sequence_length = test_sequence_length[steps * batch_size:]
    batch_size = len(batch_X)
    batch_X_padded = np.zeros(shape=(batch_size, max_length, input_size))
    for b in range(batch_size):
        batch_X_padded[b, :len(batch_X[b])] = batch_X[b]

    feed_dict = {
        input_X: batch_X_padded,
        input_y: batch_y,
        sequence_length: batch_sequence_length,
        dropout_keep_prob: 1.0
    }

    accracy_batch = sess.run(accuracy, feed_dict=feed_dict)
    accuracy_total += accracy_batch
    steps += 1
    print("Batch {} Accuracy : {}".format(steps, accracy_batch))

print("Total Accuracy : {}".format(accuracy_total/steps))