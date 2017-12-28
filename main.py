import tensorflow as tf
import numpy as np
tf.set_random_seed(777)


filename_queue = tf.train.string_input_producer(['data-03-diabetes.csv'],shuffle=False, name='filename_queue')

reader = tf.TextLineReader()
key,value = reader.read(filename_queue)

record_defaults = [[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

train_x_batch, train_y_batch = tf.train.batch([xy[0:-1],xy[-1:]],batch_size=10)


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
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(20001):
        x_data, y_data = sess.run([train_x_batch,train_y_batch])
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        if step%200 == 0:
            print(step, cost_val)

    coord.request_stop()
    coord.join(threads)
    hy_val, predic_val, accu = sess.run([Hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
    print("HY_val: ", hy_val, "\nPredicted: ",predic_val, "\naccuracy: ",accu)