import tensorflow as tf
import numpy as np
tf.set_random_seed(777)

xy = np.loadtxt('data-03-diabetes.csv', delimiter=',',dtype=np.float32)

x_data = xy[:,0:-1]
y_data = xy[:,[-1]]

X = tf.placeholder(tf.float32,shape=[None,8])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([8,1]),name='weight')
B = tf.Variable(tf.random_normal([1]),name='bias')

Hypothesis = tf.sigmoid(tf.matmul(X,W)+B,name='Hypothesis')
cost = -tf.reduce_mean(Y*tf.log(Hypothesis) + (1-Y)*tf.log(1-Hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

predicted = tf.cast(Hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(20001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step%200 == 0:
            print(step, cost_val)

    hy_val, predic_val, accu = sess.run([Hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("HY_val: ", hy_val, "\nPredicted: ",predic_val, "\naccuracy: ",accu)