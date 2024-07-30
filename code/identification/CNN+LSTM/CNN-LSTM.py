import tensorflow as tf
import numpy as np
import os

def load_X(path):
    X_signals = []
    files = os.listdir(path)
    for my_file in files:
        fileName = os.path.join(path,my_file)
        file = open(fileName, 'r')
        X_signals.append(
            [np.array(cell, dtype=np.float32) for cell in [
                row.strip().split(' ') for row in file
            ]]
        )
        file.close()
    return np.transpose(np.array(X_signals), (1, 2, 0))
def load_y(y_path):
    file = open(y_path, 'r')
    y_ = np.array(
        [elem for elem in [
            row.replace('  ', ' ').strip().split(' ') for row in file
        ]],
        dtype=np.int32
    )
    file.close()
    y_ = y_ - 1
    y_ = y_.reshape(len(y_))
    n_values = int(np.max(y_)) + 1
    return np.eye(n_values)[np.array(y_, dtype=np.int32)]
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1) 
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
class Config(object):
    """
    define a class to store parameters,
    the input should be feature mat of training and testing

    Note: it would be more interesting to use a HyperOpt search space:
    https://github.com/hyperopt/hyperopt
    """

    def __init__(self, X_train, X_test):
        self.n_layers = 2   
        self.train_count = len(X_train)  
        self.test_data_count = len(X_test)  
        self.n_steps = len(X_train[0])


        self.learning_rate = 0.0025
        self.lambda_loss_amount = 0.0015
        self.training_epochs = 300
        self.batch_size = 1500

        
        self.n_inputs = len(X_train[0][0])  
        self.n_hidden = 64  
        self.n_classes = 118  
        self.W = {
            'hidden': tf.Variable(tf.random_normal([self.n_inputs, self.n_hidden])),
            'output': tf.Variable(tf.random_normal([self.n_hidden, self.n_classes]))
        }
        self.biases = {
            'hidden': tf.Variable(tf.random_normal([self.n_hidden], mean=1.0)),
            'output': tf.Variable(tf.random_normal([self.n_classes]))
        }


def LSTM_Network(_X, config):
    """Function returns a TensorFlow RNN with two stacked LSTM cells

    Two LSTM cells are stacked which adds deepness to the neural network.
    Note, some code of this notebook is inspired from an slightly different
    RNN architecture used on another dataset, some of the credits goes to
    "aymericdamien".

    Args:
        _X:     ndarray feature matrix, shape: [batch_size, time_steps, n_inputs]
        config: Config for the neural network.

    Returns:
        This is a description of what is returned.

    Raises:
        KeyError: Raises an exception.

      Args:
        feature_mat: ndarray fature matrix, shape=[batch_size,time_steps,n_inputs]
        config: class containing config of network
      return:
              : matrix  output shape [batch_size,n_classes]
    """
   
    _X = tf.transpose(_X, [1, 0, 2])  
    _X = tf.reshape(_X, [-1, config.n_inputs])
   
    _X = tf.nn.relu(tf.matmul(_X, config.W['hidden']) + config.biases['hidden'])
   
    _X = tf.split(_X, config.n_steps, 0)
   
    lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cell_2 = tf.contrib.rnn.BasicLSTMCell(config.n_hidden, forget_bias=1.0, state_is_tuple=True)
    lstm_cells = tf.contrib.rnn.MultiRNNCell([lstm_cell_1, lstm_cell_2]*config.n_layers, state_is_tuple=True)
    
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cells, _X, dtype=tf.float32)

    lstm_last_output = outputs[-1]

    return  lstm_last_output

def CNN_NetWork(X_):
    CNN_input = tf.reshape(X_,[-1, 128, 6, 1])
   
    W_conv1 = weight_variable([3, 3, 1, 32])  
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.elu(
        tf.nn.conv2d(CNN_input, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

 
    W_conv2 = weight_variable([3, 3, 32, 64])  
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.elu(
        tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 1, 1], padding='SAME')

 
    W_fc1 = weight_variable([32 * 3 * 64, 64])
    b_fc1 = bias_variable([64])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 32 * 3 * 64])
    h_fc1 = tf.nn.elu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
   
    return h_fc1

def last_full_connection_layer(lstm_output,cnn_output):
    eigen_input = tf.concat([lstm_output, cnn_output],1)
   
    W_fc2 = weight_variable([128, 118])
    b_fc2 = bias_variable([118])
    
    return tf.nn.softmax(tf.matmul(eigen_input, W_fc2) + b_fc2)  


X_ = tf.placeholder(tf.float32, [None, 128, 6])  
label_ = tf.placeholder(tf.float32, [None, 118])  


X_train = load_X('./data/train/record')
X_test = load_X('./data/test/record')

train_label = load_y('./data/train/label.txt')
test_label = load_y('./data/test/label.txt')

config = Config(X_train, X_test)
lstm_output = LSTM_Network(X_,config)
cnn_output = CNN_NetWork(X_)
pred_Y = last_full_connection_layer(lstm_output,cnn_output)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(label_ * tf.log(pred_Y+1e-10), reduction_indices=[1])) 
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy) 
correct_prediction = tf.equal(tf.argmax(pred_Y,1), tf.argmax(label_,1)) 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer()) 

best_accuracy = 0
for i in range(100):
    batch_size = 512
    for start,end in zip(range(0,len(train_label),batch_size),
                         range(batch_size,len(train_label)+1,batch_size)):
        sess.run(train_step,feed_dict={
            X_:X_train[start:end],
            label_:train_label[start:end]
        })
        # Test completely at every epoch: calculate accuracy
    accuracy_out, loss_out = sess.run(
        [accuracy, cross_entropy],
        feed_dict={
            X_:X_test,
            label_:test_label
        }
    )
    if accuracy_out > best_accuracy:
        best_accuracy = accuracy_out
    print(str(i)+'th cross_entropy:',str(loss_out),'accuracy:',str(accuracy_out))

print("best accuracy:"+str(best_accuracy))