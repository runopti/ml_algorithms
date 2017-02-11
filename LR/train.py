import tensorflow as tf
import numpy as np
from model import LR, Config

def data_load(n,p):
    # random X and random beta. y will be Xb + Gaussian noise
    # and our goal is to reconstruct beta from X and y
    X = np.random.rand(n*p).reshape(n,p)         
    beta = np.random.rand(p).reshape(p)
    y = np.matmul(X, beta) + np.random.rand(n)
    return X, y
 
def get_minibatch(batch_size, X, y):
    n,_ = X.shape
    block_range = int(n / batch_size) 
    start = 0 
    batch_X = X[start:start+batch_size,:] 
    batch_y = y[start:start+batch_size]
    return batch_X, batch_y 
        

def main():
    g1 = tf.Graph()
    with g1.as_default():
        config = Config()
        n = config.input_dim
        p = config.feature_dim
        n_epoch = 10
        batch_size = config.batch_size
        trainX, trainy = data_load(n,p)

        model = LR(config)
        tf.get_default_graph().finalize()
        for i in range(n_epoch):
            batch_X, batch_y = get_minibatch(batch_size, trainX, trainy)
            cost = model.forward_backprop(batch_X, batch_y)
            cost_per_sample = cost / batch_size
            print(cost_per_sample)

if __name__ == "__main__":
    main()
         
