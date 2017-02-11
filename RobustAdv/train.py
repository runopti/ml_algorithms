import tensorflow as tf
import os,sys,math
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from args import args
from model import MLP, Config
from utils.progress import Progress

def main():
    # load data
    # images, labels = dataset.load_train_images()
    mnist = input_data.read_data_sets(args.dataset_dir+'MNIST_data', one_hot=True)
    #image = mnist.test.images[0]
    #plt_out = plt.imshow(image.reshape(28,28), cmap="gray") 
    #plt.show()

    images = mnist.train.images
    labels = mnist.train.labels
    
    max_epoch = 1000
    with tf.Graph().as_default():
        config = Config()
        model = MLP(config)
        tf.get_default_graph().finalize()
        progress = Progress()
        # TODO: define n_batch_loop correctly
        n_batch_loop = int(50000/config.batch_size)
        for epoch in range(max_epoch):
            sum_cost = 0
            progress.start_epoch(epoch, max_epoch)

            for t in range(n_batch_loop):
                # batch_X: batch_size x n_input
                # batch_y: batch_size
                # batch_X, batch_y = get_minibatch(batch_size, images, labels)
                batch_X, batch_y = mnist.train.next_batch(config.batch_size)
                cost = model.forward_backprop(batch_X, batch_y)
                cost_per_sample = cost / config.batch_size
                sum_cost += cost_per_sample

                if t % 10 == 0:
                    progress.show(t, n_batch_loop, {})

        # TODO: Wrote model saving in mlp.py
        # model.save(args.model_dir)
        
        # Validation
        # TODO: Finish this part

def test():
    mnist = input_data.read_data_sets(args.dataset_dir+'MNIST_data', one_hot=True)
    batch_size = 100
    batch_X, batch_y = mnist.train.next_batch(batch_size)
    print(batch_X.shape)
    print(batch_y.shape)

if __name__=="__main__":
    main()
    #test()
