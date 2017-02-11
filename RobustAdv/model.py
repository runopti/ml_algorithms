import tensorflow as tf
import numpy as np
import math

class Config(object):
    def __init__(self):
        self.sample_size = 10000
        self.n_input = 784
        self.n_h1 = 400
        self.n_h2 = 200
        self.n_label = 10
        self.batch_size = 100
        self.lr = 0.001
        

class MLP(object):
    def __init__(self, config):
        self.input = tf.placeholder(tf.float32, shape=[config.batch_size, config.n_input])
        self.target = tf.placeholder(tf.int32, shape=[config.batch_size, config.n_label])
            
        # feed the data to the model and get the output 
        self.y_hat = self._inference(self.input, config)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y_hat, self.target))
        # For backprop call
        self.train_op = tf.train.GradientDescentOptimizer(config.lr).minimize(self.loss)

        # Start Session
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def _inference(self, input, config):
        h1 = self.add(input, config.n_input, config.n_h1, "ReLU", "hidden1")
        h2 = self.add(h1, config.n_h1, config.n_h2, "ReLU", "hidden2")
        logits = self.add(h2, config.n_h2, config.n_label, "Softmax","hidden3")
        return logits

    def add(self, prev, n_in, n_out, activation, name_scope):
        with tf.name_scope(name_scope) as scope:
            weights = tf.Variable(tf.truncated_normal([n_in, n_out], stddev=1.0/math.sqrt(float(n_in)), name="weights"))
            biases = tf.Variable(tf.zeros([n_out]), name="biases") 
            #weights = tf.get_variable("weights", [n_in, n_out], tf.random_normal_initializer())
            #biases = tf.get_variable("biases", [n_out], tf.random_normal_initializer())
            if activation=="ReLU":
                hidden = tf.nn.relu(tf.matmul(prev, weights) + biases)
                return hidden
            elif activation=="Softmax":
                # softmax will be done in the loss calc by tf.nn.softmax_cross_entropy_with_logits
                hidden = tf.matmul(prev, weights) + biases
                return hidden 
            else:
                print("Didn't specify activation!!!")
                raise NotImplementedError()
            

    def forward_backprop(self, data, targets):
        cost, _ = self.sess.run([self.loss, self.train_op], feed_dict={self.input:data, self.target: targets})
        return cost

    def load(self):
        pass

    def save(self):
        pass
              
    def getWeights(self):
        pass

