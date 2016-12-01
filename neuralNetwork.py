import tensorflow as tf
import numpy as np

# level is the shape of the neural net
# ex. a shape of [5, 2, 1]
# this takes an input of 5, has a two hidden nodes, and one output node
# dropoutProb is the probability that a node is dropped for each level
# ex. a shape of [5, 2, 1] may have a dropoutProb of [ .6, .8, .5 ]
# trying to allow user to choose what the activations are for each
# layer
class neuralNet:
	def	__init__(self,level,learningRate = 0.1, activations = None,lossFunction = "squared_mean" ):
		self.shape = level
		if activations == None:
			activations = ["sigmoid" for act in range(len(level))]
		# input is the patient's genes that are given to the neural network
		self.input = tf.placeholder(tf.float32,[None,level[0]],name='Input')
		# y_ = the true y for the patient only if training otherwise if testing no need to put y_
		self.y_ = tf.placeholder(tf.float32,[None,level[-1]],name='Y')
		# the mask is used to filter out the drug values that don't exist
		self.filter = tf.placeholder(tf.float32,[None,level[-1]],name='Mask')
		self.weights = []
		self.bias = []
		self.dropoutProbList = []
		left = level[0] #genes
		self.y = self.input #genes are given to the input
		i = 1
		# create each layer of the neural network
		for lvl,act in zip(level[1:],activations):
			wght = tf.Variable(tf.zeros([left,lvl]),name=('wght_'+str(i)))
			keep_prob = tf.placeholder(tf.float32)
			drop = tf.nn.dropout(wght,keep_prob)
			bia = tf.Variable(tf.zeros([lvl]),name=('bias_'+str(i)))
			print(act)
			if act == "sigmoid":
				self.y = tf.sigmoid(tf.matmul(self.y,wght)+bia,name=('sigmoid_'+str(i)))
			elif act == "relu":
				self.y = tf.nn.relu(tf.matmul(self.y,wght)+bia,name=('sigmoid_'+str(i)))
			elif act == "linear":
				self.y = tf.matmul(self.y,wght)+bia

			self.dropoutProbList.append(keep_prob)
			self.weights.append(wght)
			self.bias.append(bia)
			left = lvl
			i += 1
		# filter is applied to the predictedy
		# example: y = {0.12,0.6,0.3,0.9} 	actualY = 	{0,1,1,No value}
		# 									filter = 	{1,1,1,0}
		self.y = self.y*self.filter
		#self.cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.y),reduction_indices=[1]))
		# compares the actual drug effectiveness to predicted drug effectiveness
		# by subtracting the two values 			( 0-0 = 0, 1-1=0 ) same
		# and then adding all the incorrect values	( 0-1 = -1, 1-0=1) different
		if lossFunction == "squared_mean":
			self.lossFunction = tf.reduce_mean(tf.reduce_sum(np.square(self.y_-self.y),reduction_indices=[1]))
		elif lossFunction == "mean_absolute":
			self.lossFunction = tf.reduce_mean(tf.reduce_sum(abs(self.y_-self.y),reduction_indices=[1]))
		elif lossFunction == "hinge_loss":
			self.lossFunction = tf.reduce_mean(tf.reduce_sum(abs(1 - self.y_*self.y),reduction_indices=[1]))
		elif lossFunction == "cross_entropy":
			self.lossFunction = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.y, self.y_))
		self.train_step = tf.train.GradientDescentOptimizer(learningRate).minimize(self.lossFunction)
		correct_prediction = tf.equal(self.y_,tf.round(self.y))
		self.accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32),reduction_indices=[0])
		self.init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(self.init)
		self.activations = activations
		#self.saver = tf.train.Saver()
		#self.saver.save(self.sess,"sess.ckpt")
	def training_run(self,x,y,filt = None,dropoutProb = None,epochs = None):
		if dropoutProb == None:
			dropoutProb = [1]*len(self.shape)
		if epochs == None:
			epochs = 1
		if filt == None:
			filt = []
			if len(np.shape(y)) == 1:
				filt = [1]*self.shape[-1]
			else:
				filt = [[1]*self.shape[-1]]*len(y)
		dictionary = {self.input: x, self.y_: y, self.filter: filt}
		for dr,dp in zip(self.dropoutProbList,dropoutProb):
			dictionary[dr] = dp
		ret = []
		tenth = int(epochs/10)
		for i in range(epochs):
			a = self.sess.run([self.train_step,self.y,self.lossFunction,self.accuracy],feed_dict=dictionary)[2]
			if i%tenth == 0:
				ret.append(a)
		return ret
	def run(self,x,y,filt = None,dropoutProb = None):
		if dropoutProb == None:
			dropoutProb = [1]*len(self.shape)
		if filt == None:
			filt = []
			if len(np.shape(x)) == 1:
				filt = [1]*self.shape[-1]
			else:
				filt = [[1]*self.shape[-1]]*len(x)
		dictionary = {self.input: x, self.filter: filt, self.y_: y}
		for dr,dp in zip(self.dropoutProbList,dropoutProb):
			dictionary[dr] = dp
		# returns a prediction for the patient
		return self.sess.run([self.y,self.accuracy],feed_dict=dictionary)
	def reset(self):
		#self.restore(self.sess,"sess.ckpt")
		self.sess.run(self.init)
	def get_weights(self):
		return self.sess.run([*self.weights])
