from libs import datasets
import matplotlib.pyplot as plt
import numpy as np
# ds = datasets.MNIST(one_hot=True)
ds = datasets.MNIST(one_hot=True)
# let's look at the first label
print(ds.Y[0])
# okay and what does the input look like
plt.imshow(np.reshape(ds.X[0], (28, 28)), cmap='gray')
plt.show()
