from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#input
x = tf.placeholder(tf.float32, [None, 784])
#weight
W = tf.Variable(tf.zeros([784, 10]))
#bais
b = tf.Variable(tf.zeros([10]))
#matrix multiply
y = tf.nn.softmax(tf.matmul(x, W) + b)

#a new placeholder to input the correct answers
y_ = tf.placeholder(tf.float32, [None, 10])

#loss func
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#back prop
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(1000)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

#tf.argmax is to find max within an array.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

#tf.cast is to cast to float
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
