import tensorflow as tf
import BiLSTM
import numpy as np
import time
import sys

input_size = 300
layers = 1
hidden_units = 128
batch_size = 32
dropout_keep_prob = 0.75
max_length = 100
learning_rate = 0.001
epochs = 3
path = "./model/BiLSTM"

train_X = np.load("./data/train_X.npy")
train_y = np.load("./data/train_y.npy")
test_X = np.load("./data/test_X.npy")
test_y = np.load("./data/test_y.npy")
print("Data was read")
batches = int(len(train_X)/batch_size)
train_sequence_length = [len(x) for x in train_X]
test_sequence_length = [len(x) for x in test_X]

sess = tf.Session()
print("Session was created")
model = BiLSTM.BiLSTM(input_size, layers, hidden_units, max_length, learning_rate)
print("Model was initialized")

saver = tf.train.Saver()
writer = tf.summary.FileWriter("./log", sess.graph)
merge_graph = model.merge_graph
global_steps = 0

def train_step(batch_X, batch_y, sequence_length, steps, epoch):
    batch_X_padded = np.zeros(shape=(batch_size, max_length, input_size))
    for b in range(batch_size):
        batch_X_padded[b, :len(batch_X[b])] = batch_X[b]

    feed_dict = {
        model.input_X: batch_X_padded,
        model.input_y: batch_y,
        model.sequence_length: sequence_length,
        model.dropout_keep_prob: dropout_keep_prob
    }

    _, loss, accuracy = sess.run([model.train, model.loss, model.accuracy], feed_dict=feed_dict)
    if steps % 100 == 0:
        summary = sess.run(merge_graph, feed_dict=feed_dict)
        writer.add_summary(summary, global_step=global_steps)
    if steps % 1000 == 0:
        print("Steps: {}, Epochs: {}/{}, Loss: {}, Accuracy: {}".format(steps, epoch, epochs, loss, accuracy))

sess.run(tf.global_variables_initializer())
print("Training model ...")
start = time.time()
for epoch in range(epochs):
    steps = 0
    for batch in range(batches):
        batch_X, batch_y = train_X[steps*batch_size:(steps+1)*batch_size], train_y[steps*batch_size:(steps+1)*batch_size]
        sequence_length = train_sequence_length[steps*batch_size:(steps+1)*batch_size]
        try:
            train_step(batch_X, batch_y, sequence_length, steps, epoch)
        except ValueError:
            print(ValueError)
        except KeyboardInterrupt:
            saver.save(sess, path)
            sys.exit(1)
        steps += 1
        global_steps += 1
    saver.save(sess, path)
saver.save(sess, path)
end = time.time()
print("Model was trained : {} seconds".format(end-start))