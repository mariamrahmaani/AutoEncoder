# This program read data from MNST Dataset and use Fully connected autoencoders to regenerate the same data (digits)
# ae6-learning0.final.gif
# imports
# %matplotlib inline
# %pylab osx
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
# Some additional libraries which we'll use just
# to produce some visualizations of our training
from libs.utils import montage
from libs import gif
import IPython.display as ipyd
plt.style.use('ggplot')
# Bit of formatting because I don't like the default inline code style:
from IPython.core.display import HTML
HTML("""<style> .rendered_html code { 
    padding: 2px 4px;
    color: #c7254e;
    background-color: #f9f2f4;
    border-radius: 4px;
} </style>""")
from libs.datasets import MNIST

# MNIST_read will read 10,000 images from MNIST dataset and reshape it to 28X28 images
# returns a list of 10,000 images 

#def MNIST_mean_std(imgs):
 #   imgs = MNIST_read()
  #  mean_img = np.mean(imgs , axis=0)
   # std_img = np.std(imgs, axis=0)
   # plt.imshow(mean_img)#, cmap='gray')
   # plt.imshow(std_img)#, cmap='gray')
   # plt.show()
   # return

def create_vanilla_auto_encoder(n_features, dimensions):
    ds = MNIST()
    # X is the list of all the images in MNIST dataset 
    imgs = ds.X[:1000].reshape((-1, 28, 28))
    # Then create a montage and draw the montage
    plt.imshow(montage(imgs), cmap='gray')
    plt.show()
    mean_img = np.mean(ds.X, axis=0)
    std_img = np.std(imgs, axis=0)
    X = tf.placeholder(tf.float32, [None, n_features])
    current_input = X
    n_input = n_features
    Ws = []
    for layer_i, dimension_i in enumerate(dimensions, start=1):
        with tf.variable_scope("encoder/layer/{}".format(layer_i)):
            w = tf.get_variable(name='W', 
                                shape=[n_input, dimension_i], 
                                initializer=tf.random_normal_initializer(mean=0.0, stddev=2.0))
            b = tf.get_variable(name='b',
                                shape=[dimension_i],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
        
            h = tf.nn.bias_add(name='h',
                               value=tf.matmul(current_input, w),
                               bias=b)
            current_input = tf.nn.relu(h)
            Ws.append(w)
            n_input = dimension_i

    Ws = Ws[::-1]
    dimensions = dimensions[::-1][1:]+[n_features]
    print('dimensions=',dimensions)
    for layer_i, dimension_i in enumerate(dimensions):
            with tf.variable_scope("decoder/layer/{}".format(layer_i)):
                w= tf.transpose(Ws[layer_i])
                b = tf.get_variable(name='b',
                                    shape=[dimension_i],
                                    dtype=tf.float32,
                                    initializer=tf.constant_initializer(0.0))
                
                print('current_input= ', current_input)
                print('w = ', w)

                h = tf.nn.bias_add(name='h',
                                   value=tf.matmul(current_input, w),
                                   bias=b)
                current_input = tf.nn.relu(h)
                n_input = dimension_i
            
    Y = current_input
    cost = tf.reduce_mean(tf.squared_difference(X, Y), 1)
    print('cost.getshape',cost.get_shape())
    cost = tf.reduce_mean(cost)
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    batch_size = 100
    n_epochs = 60                                                                                            
    # We'll try to reconstruct the same first 100 images and show how The network does over the course of training.
    examples = ds.X[:100]
    mean_img = np.mean(examples, axis=0)
    #recon0 = np.clip(examples.reshape((-1, 28, 28)), 0, 255)
    #img_or = montage(recon0).astype(np.uint8)
    #img_or.append('0')
    #gif.build_gif(img_or, saveto='example.{}.gif'.format(np.random.rand()), cmap='gray')
    #plt.show()
    # We'll store the reconstructions in a list
    imgs = []
    fig, ax = plt.subplots(1, 1)
    for epoch_i in range(n_epochs):
        for batch_X, _ in ds.train.next_batch():
            sess.run(optimizer, feed_dict={X: batch_X - mean_img})
        recon = sess.run(Y, feed_dict={X: examples - mean_img})
        recon = np.clip((recon + mean_img).reshape((-1, 28, 28)), 0, 255)
        img_i = montage(recon).astype(np.uint8)
        imgs.append(img_i)
        ax.imshow(img_i, cmap='gray')
        fig.canvas.draw()
        #plt.imshow(img_i, cmap='gray')
        #plt.show()
        print(epoch_i, sess.run(cost, feed_dict={X: batch_X - mean_img}))
    gif.build_gif(imgs, saveto='ae6-learning0.{}.gif'.format(np.random.rand()), cmap='gray')
    ipyd.Image(url='ae.gif?{}'.format(np.random.rand()),height=500, width=500)
    return Y



n_features = 784 # n_feature is the number of features or pixels in one image which is  ds.X.shape[1]=784
dimensions = [512,256,128,64]
Y = create_vanilla_auto_encoder(n_features, dimensions)



########################
## CLEAN UP
########################



