import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Global variables #
batch_size = 32
max_iterations = 5000   

# Load the data # 
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
x_train, y_train = mnist.train.next_batch(batch_size)

###############################################################
############## BUILDING THE COMPUTATIONAL GRAPH ###############
###############################################################
# Declare parameters #
weights = {
    'w1': tf.Variable(tf.random_normal([784, 256])),
    'w2': tf.Variable(tf.random_normal([256, 128])),
    'w3': tf.Variable(tf.random_normal([128, 10]))
    }
biases = {
    'b1': tf.Variable(tf.random_normal([256])),
    'b2': tf.Variable(tf.random_normal([128])),
    'b3': tf.Variable(tf.random_normal([10]))
    }

# Inputs
x_in = tf.placeholder(tf.float32, [None, 784])
y_in = tf.placeholder(tf.float32, [None, 10])

# Inference. Compute everything until the predictions 
y = tf.nn.sigmoid(tf.matmul(x_in, weights['w1']) + biases['b1']) # INPUT LAYER 
y = tf.nn.sigmoid(tf.matmul(y, weights['w2']) + biases['b2'])    # HIDDEN LAYER 1
y = tf.matmul(y, weights['w3']) + biases['b3']                   # HIDDEN LAYER 2 
y_pred = tf.nn.softmax(y)                                        # PREDICTION 

# Cost Function. Compare predictions with true labels
# cross_entropy = tf.reduce_mean(- y_pred * tf.log(y_in))
# We use the cross entropy of tensorflow because it's numerically stable 
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_in, logits=y))

# Optimization 
optimizer  = tf.train.GradientDescentOptimizer(learning_rate=0.5)
gradients  = optimizer.compute_gradients(cross_entropy)
train_step = optimizer.apply_gradients(gradients)

# Evaluation 
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_in, 1))
accuracy           = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Visualization (you can ignore this)
ac_sum = tf.summary.scalar('accuracy', accuracy)
ce_sum = tf.summary.scalar('loss', cross_entropy)
merged = tf.summary.merge_all()

###############################################################
############## RUNNING THE COMPUTATIONAL GRAPH ################
############################################################### 
with tf.Session() as sess: 
    writer = tf.summary.FileWriter('./3e', sess.graph)  # TensorBoard writer 
    # We need to initialize the variables 
    tf.global_variables_initializer().run()

    # Train the model 
    for i in range(max_iterations):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, loss, summ = sess.run([train_step, cross_entropy, merged], 
                                  feed_dict={x_in: batch_xs, y_in: batch_ys})
        writer.add_summary(summ, i)

    # Test trained model
    ac = sess.run(accuracy, feed_dict={x_in: mnist.test.images, y_in: mnist.test.labels})
    print('Accuracy:', ac)

