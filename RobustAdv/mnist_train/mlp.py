import tensorflow as tf
import numpy as np
import math

class MLP:
    def __init__(self, input_size, target_size, hidden_size1, hidden_size2, batch_size):
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.input_size = input_size
        self.target_size = target_size 
        print(self.hidden_size1)

        self.input = tf.placeholder(tf.float32, shape=(batch_size, input_size))
        self.target = tf.placeholder(tf.int32, shape=(batch_size))
            
        # feed the data to the model and get the output 
        self.output = output  =  self._inference(self.input)

        # get loss value
        #loss = self._loss(output, target)

        # get train_op
        #self._train_op = self._training(loss)

    def _inference(self, input):
        with tf.name_scope("hidden1") as scope:
            weights = tf.Variable(tf.truncated_normal((self.input_size, self.hidden_size1), stddev=1.0/math.sqrt(float(self.input_size)), name="weights"))
            biases = tf.Variable(tf.zeros([self.hidden_size1]), name="biases")
            hidden1 = tf.nn.relu(tf.matmul(input, weights) + biases)
        with tf.name_scope("hidden2") as scope:
            weights = tf.Variable(tf.truncated_normal((self.hidden_size1, self.hidden_size2), stddev=1.0/math.sqrt(float(self.hidden_size1)), name="weights"))
            biases = tf.Variable(tf.zeros([self.hidden_size2]), name="biases")
            hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
        with tf.name_scope("soft_max") as scope:
            weights = tf.Variable(tf.truncated_normal((self.hidden_size2, self.target_size), stddev=1.0/math.sqrt(float(self.target_size)), name="weights"))
            biases = tf.Variable(tf.zeros([self.target_size]), name="biases")
            output = tf.matmul(hidden2, weights) + biases
            return output
              
    def test(self):
        with tf.name_scope("test") as scope:
            weights = tf.Variable(tf.truncated_normal((2,4)))
            return weights

    @property
    def train_op(self):
        return self._train_op

    @property
    def output_op(self):
        return self.output
    
    @property
    def input_data(self):
        return self.input

    @property
    def target(self):
        return self.target


