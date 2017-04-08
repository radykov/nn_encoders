import tensorflow as tf
import visualiser

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

INPUT_NEURONS = 784
ENCODED_NEURONS = 10
OUTPUT_NEURONS = INPUT_NEURONS
OPTIMIZER_STEP_SIZE = 1e-3

x_ = tf.placeholder(tf.float32, [None, INPUT_NEURONS])

Wi1 = tf.Variable(tf.random_uniform([INPUT_NEURONS, ENCODED_NEURONS], 0.001, 0.01))
bi1 = tf.Variable(tf.zeros([ENCODED_NEURONS]))

Ei1 = tf.nn.relu(tf.matmul(x_, Wi1) + bi1)

Wo1 = tf.Variable(tf.random_uniform([ENCODED_NEURONS, OUTPUT_NEURONS], -0.1, 0.1))
bo1 = tf.Variable(tf.zeros([OUTPUT_NEURONS]))

# Sigmoid the output to have it fall within 0 to 1, makes smoother images then leaving it as matmul or relu
y = tf.sigmoid(tf.matmul(Ei1, Wo1) + bo1)
y_ = tf.placeholder(tf.float32, [None, OUTPUT_NEURONS])

# Square the difference in output vs expected, the closer the output is to expected, the less error there is
cost = tf.reduce_sum(tf.square(tf.subtract(y_, y)))
train_step = tf.train.AdamOptimizer(OPTIMIZER_STEP_SIZE).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i%1000 == 0:
        print i
    sess.run(train_step, feed_dict={x_: batch_xs, y_: batch_xs})

print visualiser.visualise_image(sess, mnist, x_, y_, y)