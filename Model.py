import random
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
from scipy.interpolate import spline



def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
	m = X.shape[0]                  # number of training examples
	mini_batches = []
	np.random.seed(seed)
	
	# Step 1: Shuffle (X, Y)
	permutation = list(np.random.permutation(m))
	shuffled_X = X[permutation,:]
	shuffled_Y = Y[permutation,:]

	# Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
	num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
	for k in range(0, num_complete_minibatches):
		mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size:]
		mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	# Handling the end case (last mini-batch < mini_batch_size)
	if m % mini_batch_size != 0:
		mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m:]
		mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m:]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	
	return mini_batches



class Model():
	
	def __init__(self):
		self._sess = tf.Session()

	def train(self, X_train, x_validation, Y_train, y_validation, learning_rate = 0.001, num_epochs = 20, minibatch_size = 32):
		tf.set_random_seed(1)
		m = X_train.shape[0] 
		seed = 3
		costs = [] 
		epochs = []
		self.X = tf.placeholder(tf.float32, shape=[None,600,400,3])
		self.Y = tf.placeholder(tf.float32, shape=[None,8,])

		#CNN structure(convolution + pooling)
		conv1 = tf.layers.conv2d(inputs=self.X, filters=10, kernel_size=[3,3], padding="same", activation=tf.nn.relu)
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
		conv2 = tf.layers.conv2d(inputs=pool1,filters=5,kernel_size=[3,3],padding="same",activation=tf.nn.relu)
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
	
		#fully-connected
		flattened = tf.contrib.layers.flatten(pool2)
		self.out = tf.contrib.layers.fully_connected(inputs = flattened, num_outputs = 8, activation_fn=None)

		cost = tf.losses.mean_squared_error(labels=self.Y, predictions=self.out)
		optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

		self._sess.run(tf.global_variables_initializer())
		loss_list = []
		epoch_number = []
		for epoch in range(1, num_epochs+1):
				epoch_cost = 0.
				seed = seed + 1
				num_minibatches = int(m / minibatch_size)
				minibatches = random_mini_batches(X_train, Y_train, mini_batch_size = minibatch_size, seed = 0)
				for minibatch in minibatches:

					(minibatch_X, minibatch_Y) = minibatch

					_ , minibatch_cost = self._sess.run([optimizer, cost], feed_dict={self.X: minibatch_X, self.Y: minibatch_Y})

					epoch_cost += minibatch_cost / num_minibatches

				if epoch % 1 == 0:
					epochs.append(epoch)
					self._sess.run(cost, feed_dict={self.X: x_validation, self.Y: new_y_validation})
					print ("Cost after epoch %i: %f" % (epoch, epoch_cost))

				costs.append(epoch_cost)
		# plot the cost
		x = np.array(epochs)
		xnew = np.linspace(x.min(),x.max(),20)
		power_smooth = spline(x,costs,xnew)
		plt.plot(xnew,power_smooth)
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
	def predict(self, X_test):
		return self._sess.run(self.out, feed_dict={self.X:X_test})