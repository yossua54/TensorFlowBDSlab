import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

# Global variables #
batch_size = 32
max_iterations = 30000  

# Load the data # 
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
x_train, y_train = mnist.train.next_batch(batch_size)

###############################################################
############## BUILDING THE COMPUTATIONAL GRAPH ###############
###############################################################
# Declare parameters #
weights = {
    'encoder_w1': tf.Variable(tf.random_normal([784, 256])),
    'encoder_w2': tf.Variable(tf.random_normal([256, 64])),
    'decoder_w1': tf.Variable(tf.random_normal([64, 256])),
    'decoder_w2': tf.Variable(tf.random_normal([256, 784]))
    }
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([256])),
    'encoder_b2': tf.Variable(tf.random_normal([64])),
    'decoder_b1': tf.Variable(tf.random_normal([256])),
    'decoder_b2': tf.Variable(tf.random_normal([784]))
    }

# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_w1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_w2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_w1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_w2']),
                                   biases['decoder_b2']))
    return layer_2

# Inputs
x_in = tf.placeholder(tf.float32, [None, 784])

# Inference. Compute everything until the predictions 
encoder_op = encoder(x_in)
decoder_op = decoder(encoder_op) 
y_pred     = decoder_op

# Cost Function. Compare predictions with true labels
# In this case with use the L2 distance 
loss = tf.reduce_mean(tf.pow(x_in - y_pred, 2))

# Optimization 
optimizer  = tf.train.RMSPropOptimizer(learning_rate=0.01)
gradients  = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(gradients)

###############################################################
############## RUNNING THE COMPUTATIONAL GRAPH ################
############################################################### 
with tf.Session() as sess: 
    writer = tf.summary.FileWriter('./4e', sess.graph)  # TensorBoard writer 
    # We need to initialize the variables 
    tf.global_variables_initializer().run()

    # Train the model 
    for i in range(max_iterations):
        batch_xs, _ = mnist.train.next_batch(batch_size)
        _, l = sess.run([train_step, loss], feed_dict={x_in: batch_xs})
        
        if (i % 1000) == 0:
            print('Iteration', i, ' Loss:', l)

    n = 4
    canvas_orig = np.empty((28 * n, 28 * n))
    canvas_recon = np.empty((28 * n, 28 * n))
    for i in range(n):
        # MNIST test set
        batch_x, _ = mnist.test.next_batch(n)
        # Encode and decode the digit image
        g = sess.run(decoder_op, feed_dict={x_in: batch_x})

        # Display original images
        for j in range(n):
            # Draw the original digits
            canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                batch_x[j].reshape([28, 28])
        # Display reconstructed images
        for j in range(n):
            # Draw the reconstructed digits
            canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = \
                g[j].reshape([28, 28])

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()


