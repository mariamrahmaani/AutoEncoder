import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
import libs.utils as utils
import libs.datasets as datasets
from skimage.transform import resize
from skimage.color import rgb2gray
from libs.utils import montage_filters
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

def images_to_gray(img_lst):
    imgs_t = [rgb2gray(img) for img in img_lst] 
    imgs = np.array(imgs_t).reshape(-1, 100, 100)
    print('imgs.shape= ', imgs.shape)
    return(imgs)

dirname=".\\MyPic"
filenames = [os.path.join(dirname, fname) for fname in os.listdir(dirname)]
# Make sure we have exactly 100 image files!
filenames = filenames[:100]
assert(len(filenames) == 100)
#print(filenames)

# Read every filename as an RGB image
imgs = [plt.imread(fname)[..., :3] for fname in filenames]

# Crop every image to a square
imgs = [utils.imcrop_tosquare(img_i) for img_i in imgs]

# Then resize the square image to 100 x 100 pixels; mode='reflect'
imgs = [resize(img_i, (100, 100), mode='reflect') for img_i in imgs]

# Then convert the list of images to a 4d array (e.g. use np.array to convert a list to a 4d array):
Xs = np.array(imgs).astype(np.float32)
#print(Xs.shape)
assert(Xs.ndim == 4 and Xs.shape[1] <= 100 and Xs.shape[2] <= 100)
ds = datasets.Dataset(Xs)

mean_img = ds.mean()
#plt.imshow(mean_img)
#plt.show()
# If your image comes out entirely black, try w/o the `astype(np.uint8)`
# that means your images are read in as 0-255, rather than 0-1 and 
# this simply depends on the version of matplotlib you are using.
std_img = ds.std()
#plt.imshow(std_img)
#plt.show()
#print(std_img.shape)
std_img = np.mean(std_img, axis=2)
#plt.imshow(std_img)
#plt.show()
#plt.imshow(ds.X[0])
#plt.show()
#print(ds.X.shape)

#reset_default_graph()

# X is the list of all the images 

imgs = images_to_gray(ds.X[:100])

# And we'll create a placeholder in the tensorflow graph that will be able to get any number of n_feature inputs.
# Then create a montage and draw the montage
plt.imshow(montage(imgs), cmap='gray')
plt.show()
n_features = 10000
mean_img = np.mean(imgs, axis=0)
std_img = np.std(imgs, axis=0)

plt.imshow(mean_img)
plt.show()
plt.imshow(std_img)
plt.show()

X = tf.placeholder(tf.float32, [None, n_features])
X_tensor = tf.reshape(X, [-1, 100, 100, 1])
n_filters = [16, 16, 16]
filter_sizes = [4, 4, 4]
current_input = X_tensor

# notice instead of having 784 as our input features, we're going to have
# just 1, corresponding to the number of channels in the image.
# We're going to use convolution to find 16 filters, or 16 channels of information in each spatial location we perform convolution at.
n_input = 1

# We're going to keep every matrix we create so let's create a list to hold them all
Ws = []
shapes = []

# We'll create a for loop to create each layer:
for layer_i, n_output in enumerate(n_filters):
    # just like in the last session,
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("encoder/layer/{}".format(layer_i)):
        # we'll keep track of the shapes of each layer
        # As we'll need these for the decoder
        shapes.append(current_input.get_shape().as_list())

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = tf.get_variable(
            name='W',
            shape=[
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input,
                n_output],
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.02))

        # Now we'll convolve our input by our newly created W matrix
        h = tf.nn.conv2d(current_input, W,
            strides=[1, 2, 2, 1], padding='SAME')

        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)

        # Finally we'll store the weight matrix so we can build the decoder.
        Ws.append(W)

        # We'll also replace n_input with the current n_output, so that on the
        # next iteration, our new number inputs will be correct.
        n_input = n_output
# We'll first reverse the order of our weight matrices
Ws.reverse()
# and the shapes of each layer
shapes.reverse()
# and the number of filters (which is the same but could have been different)
n_filters.reverse()
# and append the last filter size which is our input image's number of channels
n_filters = n_filters[1:] + [1]
print(n_filters, filter_sizes, shapes)
# and then loop through our convolution filters and get back our input image
# we'll enumerate the shapes list to get us there
for layer_i, shape in enumerate(shapes):
    # we'll use a variable scope to help encapsulate our variables
    # This will simply prefix all the variables made in this scope
    # with the name we give it.
    with tf.variable_scope("decoder/layer/{}".format(layer_i)):

        # Create a weight matrix which will increasingly reduce
        # down the amount of information in the input by performing
        # a matrix multiplication
        W = Ws[layer_i]

        # Now we'll convolve by the transpose of our previous convolution tensor
        h = tf.nn.conv2d_transpose(current_input, W,
            tf.stack([tf.shape(X)[0], shape[1], shape[2], shape[3]]),
            strides=[1, 2, 2, 1], padding='SAME')
        # And then use a relu activation function on its output
        current_input = tf.nn.relu(h)
Y = current_input
Y = tf.reshape(Y, [-1, n_features])
cost = tf.reduce_mean(tf.reduce_mean(tf.squared_difference(X, Y), 1))
learning_rate = 0.001

# pass learning rate and cost to optimize
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Session to manage vars/train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Some parameters for training
batch_size = 100
n_epochs = 10

# We'll try to reconstruct the same first 100 images and show how
# The network does over the course of training.
#examples = ds.X[:100]
#gray = [rgb2gray(img) for img in examples] 
#print('examples.shape', examples.shape)
#print('gray.shape', gray[0].shape)
examples = imgs
examples = examples.reshape((-1, 10000))

# We'll store the reconstructions in a list
imgs = []
fig, ax = plt.subplots(1, 1)
for epoch_i in range(n_epochs):
    for batch_X, _ in ds.train.next_batch(): # need to change the output of ds.train.next_batch() to gray-> batch_X.shape (100, 100, 100, 3)
        batch_X_gray = images_to_gray(batch_X)
        batch_X_gray = batch_X_gray.reshape((-1, 10000))
        mean_img = mean_img.reshape(10000,)
        print('batch_X_gray.shape', np.array(batch_X_gray).shape) # batch_X.shape should be (100,10000)
        print('mean_img.shape', np.array(mean_img).shape)
        sess.run(optimizer, feed_dict={X: batch_X_gray - mean_img})
    recon = sess.run(Y, feed_dict={X: examples - mean_img})
    recon = np.clip((recon + mean_img).reshape((-1, 100, 100)), 0, 255)
    img_i = montage(recon).astype(np.uint8)
    imgs.append(img_i)
    ax.imshow(img_i, cmap='gray')
    fig.canvas.draw()
    print(epoch_i, sess.run(cost, feed_dict={X: batch_X_gray - mean_img}))
gif.build_gif(imgs, saveto='Mypic-conv-ae.gif', cmap='gray')
ipyd.Image(url='Mypic-conv-ae.gif?{}'.format(np.random.rand()), height=500, width=500)

# Visualize the filters
# W1 = sess.run(Ws[1])
#plt.figure(figsize=(10, 10))
#plt.imshow(montage_filters(W1), cmap='coolwarm', interpolation='nearest')
#W2 = sess.run(Ws[2])
#plt.imshow(montage_filters(W2 / np.max(W2)), cmap='coolwarm')
