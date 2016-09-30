import tensorflow as tf

class neuralNet:
	def	__init__(self,level):
		# input is the patient's genes that are given to the neural network
		self.input = tf.placeholder(tf.float32,[None,level[0]],name='Input')
		# y_ = the true y for the patient only if training otherwise if testing no need to put y_
		self.y_ = tf.placeholder(tf.float32,[None,level[-1]],name='Y')
		# the mask is used to filter out the drug values that don't exist
		self.filter = tf.placeholder(tf.float32,[None,level[-1]],name='Mask')
		self.weights = []
		self.bias = []
		left = level[0] #genes
		self.y = self.input #genes are given to the input
		i = 1
		# create each layer of the neural network
		for lvl in level[1:]:
			wght = tf.Variable(tf.random_normal([left,lvl],0,1),name=('wght'+str(i)))
			bia = tf.Variable(tf.random_normal([lvl],0,1),name=('bias'+str(i)))
			self.y = tf.sigmoid(tf.matmul(self.y,wght)+bia,name=('sigmoid'+str(i)))
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
		self.cross_entropy = tf.reduce_mean(tf.reduce_sum(abs(self.y_-self.y),reduction_indices=[1]))
		self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)
		init = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init)
	def training_run(self,x,y,filt):
		# does a training run
		return self.sess.run([self.train_step,self.cross_entropy,self.y],feed_dict={self.input: x, self.y_: y, self.filter: filt})[1:]
	def run(self,x,filt):
		# returns a prediction for the patient
		return self.sess.run(self.y,feed_dict={self.input: x,self.filter: filt})
		
		
		
		
		
		
		
		
		
		
		
		
