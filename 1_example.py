import tensorflow as tf 

###############################################################
############## BUILDING THE COMPUTATIONAL GRAPH ###############
###############################################################
node1 = tf.constant(3.0, dtype=tf.float32, name='node1') # Tensor
node2 = tf.constant(4.0, dtype=tf.float32, name='node2') # Tensor 
node3 = tf.add(node1, node2) # Operation, which returns a Tensor 

print(node1, node3) # Check BEFORE running the computational graph

###############################################################
############## RUNNING THE COMPUTATIONAL GRAPH ################
############################################################### 
sess = tf.Session() # Initialize session

# Run the computational graph until the node3, we should get the result 
# of the operation node1 + node2 
result = sess.run([node3]) # RUN the computational graph 
print('Result:', result) # Check AFTER running the computational graph

# Code to obtain a visualization in tensorboard (you can ignore this) 
writer = tf.summary.FileWriter('./e1', sess.graph)  # TensorBoard writer 
writer.close()  # Ensure the data is saved 
sess.close()    # Free the resources 
