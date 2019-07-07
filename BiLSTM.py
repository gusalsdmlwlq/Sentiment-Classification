import tensorflow as tf

class BiLSTM:
    def __init__(self, input_size, layers, hidden_units, max_length, learning_rate):
        with tf.VariableScope(name="BiLSTM", reuse=tf.AUTO_REUSE):
            self.input_X = tf.placeholder(dtype=tf.float32, shape=[None, max_length, input_size], name="input_X")
            self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="input_y")
            self.sequence_length = tf.placeholder(dtype=tf.int32, shape=[None], name="sequence_length")
            self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, shape=(), name="dropout_keep_prob")

            self.output = self.build_bilstm(self.input_X, layers, hidden_units, self.dropout_keep_prob)

            self.loss = -(self.input_y * tf.log(self.output) + (1-self.input_y) * tf.log(1-self.output))
            self.train = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

            self.prediction = tf.cast(tf.to_int32(self.loss >= 0.5), dtype=tf.float32, name="prediction")
            self.accuracy = tf.multiply(tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.prediction), dtype=tf.float32)), 100, name="accuracy")

            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            self.merge_graph = tf.summary.merge_all()

    def build_bilstm(self, X, layers, hidden_units, dropout_keep_prob):
        fw_cell_stack = []
        for layer in range(layers):
            fw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout_keep_prob)
            fw_cell_stack.append(fw_cell)
        fw_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(fw_cell_stack)

        bw_cell_stack = []
        for layer in range(layers):
            bw_cell = tf.nn.rnn_cell.LSTMCell(hidden_units)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell)
            bw_cell_stack.append(bw_cell)
        bw_lstm_cell = tf.nn.rnn_cell.MultiRNNCell(bw_cell_stack)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm_cell, cell_bw=bw_lstm_cell, inputs=X, dtype=tf.float32, sequence_length=self.sequence_length)

        fw_output = outputs[0][:,-1]
        bw_output = outputs[1][:,0]
        output = tf.concat([fw_output, bw_output], axis=1)

        W = tf.get_variable("W", shape=[hidden_units*2, 1], initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        b = tf.get_variable("b", shape=(), initializer=tf.zeros_initializer(), dtype=tf.float32)

        output = tf.nn.sigmoid(tf.matmul(output, W) + b, name="output")
        return output

