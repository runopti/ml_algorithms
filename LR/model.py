import tensorflow as tf
import numpy as np

class Config(object):
    def __init__(self):
        self.input_dim = 100
        self.feature_dim = 150
        self.batch_size = 100

class LR(object):
    def __init__(self, config):
        n = config.input_dim
        p = config.feature_dim
        batch_size = config.batch_size
        self.x = tf.placeholder(tf.float32,[batch_size,p])
        self.y = tf.placeholder(tf.float32,[batch_size])
        
        # Initialize weights
        w = tf.Variable(tf.zeros([batch_size, p]),tf.float32) 
        bias = tf.Variable(tf.zeros([batch_size]),tf.float32) 

        # Define model
        self.h1 = tf.matmul(self.x, tf.transpose(w)) + bias 
        self.loss = tf.reduce_sum(tf.square(self.h1 - self.y))
        # For backprop call
        self.train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(self.loss)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def forward_backprop(self, data, targets):
        cost, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.x: data, self.y: targets})
        return cost

    def getWeights(self):
        pass

