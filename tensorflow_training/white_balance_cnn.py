import tensorflow as tf
import numpy as np
import scipy.io as sio

batch_szie = 8;
beta = 8 * 8 * 8 * 10
hist_size = 256
lr = 0.0001


data = sio.loadmat('../data/data_7.0.mat');
train_data = np.array(data['train_data'])
train_label = np.array(data['train_label'])

print('batch train data size:', train_data.shape)
print('batch train label size:', train_label.shape)

train_data = train_data.reshape((568, 256, 256, 1))
train_label = train_label.reshape((568, 256, 256, 1))

tf_x = tf.placeholder(tf.float32, [batch_szie, hist_size, hist_size, 1])     # input x
tf_y = tf.placeholder(tf.float32, [batch_szie, hist_size, hist_size, 1])     # input y

# Variables.
kernel = tf.Variable(tf.constant(0, shape = [64, 64, 1, 1], dtype = tf.float32))

# neural network layers
# CNN
conv1 = tf.nn.conv2d(
    tf_x, 
    kernel, 
    strides = [1, 1, 1, 1],
    padding = 'SAME')

soft = tf.nn.softmax(conv1)

loss = tf.norm(tf.multiply(soft, tf_y), 1) + beta * tf.norm(kernel, 2);
train_op = tf.train.GradientDescentOptimizer(lr).minimize(loss)

# Session
sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
sess.run(init_op)

for iter in range(5000):
    step = batch_szie * 8;
    if (step + batch_szie == 568):
        step = 0
    b_x = train_data[step:(step + 8), :, :, :]
    b_y = train_label[step:(step + 8), :, :, :]
    _, loss_ = sess.run([train_op, loss], {tf_x: b_x, tf_y: b_y})
    if iter % 100 == 0:
        print('Iter:', iter, '| train loss: %.4f' % loss_)
    if iter % 1000 == 0:
        k = {}
        k_result = sess.run(kernel)
        k['kernel'] = k_result
        sio.savemat('../data/kernel_%d.mat' % (iter), k)
        lr = lr / 10;

