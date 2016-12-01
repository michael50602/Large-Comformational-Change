import numpy as np
import csv
import tensorflow as tf
import profile_parser
import answer_parser
from tensorflow.python.ops import rnn, rnn_cell, math_ops
import sklearn


def feature_max_length(fea, protein):
    tmp_max = 0
    for v in protein.values():
        if len(v[fea]) > tmp_max:
            tmp_max = len(v[fea])
    return tmp_max

def fill_mx(max_len, feature, protein_prop, keys):
    dense_mx = []
    for k in sorted(np.intersect1d(protein_prop.keys(), keys)):
        if len(protein_prop[k][feature]) != max_len:
            tmp = protein_prop[k][feature]
            for i in range(len(protein_prop[k][feature]), max_len):
                tmp.append(0)
            dense_mx.append(tmp)
    return np.array(dense_mx)

def RNN(x, weight, bias, n_hidden, feature, split_size):
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    x = tf.split(0, split_size, x)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weight['out']) + bias['out']

def run(feature):
    # global parameter
    sess = tf.InteractiveSession()
    learning_rate = 0.01
    training_iter = 100000
    batch_size = 704 # set to amount of protein sequence
    display_step = 1
    split_size = 1
    n_hidden = 256 # temporarily, haven't decide yet
    n_class = 2
    weight = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_class]))
    }
    bias = {
        'out': tf.Variable(tf.random_normal([n_class]))        
    } 
    protein_prop = profile_parser.profile_parsing()
    max_sequence_len = feature_max_length(feature, protein_prop)
    x = tf.placeholder("float", [None, max_sequence_len])
    y = tf.placeholder("float", [None, n_class])
    ans = answer_parser.answer_parsing()
    dense_mx = fill_mx(max_sequence_len, feature, protein_prop, ans.keys())
    pred = RNN(x, weight, bias, n_hidden, feature, split_size)
    pred = math_ops.mul(pred, tf.constant([1, 4], dtype=tf.float32))
    cost  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.arg_max(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.initialize_all_variables()
    train_res = []
    for k in sorted(np.intersect1d(protein_prop.keys(), ans.keys())):
        if ans[k] == 1:
            train_res.append([0, 1])
        else:
            train_res.append([1, 0])
    sess.run(init)
    step = 1
    i = 0
    while step * batch_size < training_iter:
        batch_x = dense_mx
        batch_y = train_res[i*batch_size/split_size: (i+1)*batch_size/split_size]
        sess.run(optimizer, feed_dict = {x:batch_x, y:batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
        if i >= batch_size/split_size :
            i = 0
    validation_table = open('data/va50-lst', 'r')
    test_protein_id = []
    for row in csv.reader(validation_table, delimiter='\t'):
        test_protein_id.append(row[0])
    test_sq = fill_mx(max_sequence_len, feature, protein_prop, test_protein_id)
    result = sess.run(tf.argmax(pred,1), feed_dict={x:test_sq})
    output_file = open('./test-output', 'w')
    for i, entity in enumerate(result):
        if entity == 0:
            result[i] = -1
        output_file.write(str(result[i]) + ",")
if __name__ == '__main__':
    run("molecular_weight")
