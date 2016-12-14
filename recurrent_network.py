from __future__ import print_function

import tensorflow as tf
import numpy as np
import csv
from tensorflow.python.ops import rnn, rnn_cell

import profile_parser
import answer_parser



# for feature max_len
def feature_max_length(feas, protein):
    length = []
    for fea in feas:
      tmp_max = 0
      for v in protein.values():
          if len(v[fea]) > tmp_max:
              tmp_max = len(v[fea])
      length.append(tmp_max)
    return length
# add 0 to feature data to len = max_len
def fill_mx(max_len, features, protein_prop, keys):
    dense_mx = []
    train_res = []
    for k in sorted(np.intersect1d(protein_prop.keys(), keys)):
        try: 
          if ans[k] == 1:
            train_res.append([0, 1])
          else:
            train_res.append([1, 0])
        except:
            pass
        temp = []
        for feature in features:
            sequence_length = int((max_len / 50) + 1)
            tmp = np.zeros(sequence_length * 50)
            tmp[:len(protein_prop[k][feature])] = protein_prop[k][feature]
            tmp2 = np.empty(sequence_length)
            for i in range(sequence_length):
                tmp2[i] = np.sum(tmp[i:i+49]) 
            temp.append(tmp2)
        dense_mx.append(np.transpose(temp))
    print(np.array(dense_mx).shape)
    return np.array(dense_mx), np.array(train_res)

# prepare data
protein_prop = profile_parser.profile_parsing()
ans = answer_parser.answer_parsing()
f_length = feature_max_length(['molecular_weight', 'carboxyl', 'cyclic_structure'], protein_prop) 
x_data ,y_data  = fill_mx(max(f_length), ['molecular_weight', 'carboxyl', 'cyclic_structure'], protein_prop, ans.keys())

# Parameters
learning_rate = 0.001
training_iters = 10000
batch_size = 704
display_step = 10

# Network Parameters
n_input = 3
sequence_len = int(max(f_length) / 50) + 1 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 2
# tf Graph input
x = tf.placeholder("float", [None, sequence_len, n_input])
y = tf.placeholder("float", [None, n_classes])

validation_table = open('data/va50-lst', 'r')
test_protein_id = []
for row in csv.reader(validation_table, delimiter='\t'):
  test_protein_id.append(row[0])
test_x, tmp = fill_mx(max(f_length), ['molecular_weight','carboxyl','cyclic_structure'], protein_prop, test_protein_id)
# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, sequence_len, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)
# Define loss and optimizer
softmax = tf.nn.log_softmax(pred)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# Initializing the variables
init = tf.initialize_all_variables()
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: x_data, y: y_data})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: x_data, y: y_data})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: x_data, y: y_data})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    soft = sess.run(softmax, feed_dict={x:x_data})
    print(sess.run(pre,feed_dict={x:x_data})[1,:])
    print(soft[1,:])
    result = sess.run(tf.argmax(pred,1), feed_dict={x:test_x})
    output_file = open('./test-output', 'w')
    for i, entity in enumerate(result):
        if entity == 0:
            result[i] = -1
        output_file.write(str(result[i]) + ",")
    #print("Testing Accuracy:", \
    #sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
