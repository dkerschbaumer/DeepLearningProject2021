import numpy as num
import pandas as pd
import matplotlib.pyplot as mat
import numpy.random as ran

# x_train = num.load('data/repo-kannada-mnist/X_kannada_MNIST_train.npy')
# x_test = num.load('data/repo-kannada-mnist/X_kannada_MNIST_test.npy')

train_ = pd.read_csv('data/train.csv')
test_ = pd.read_csv('data/Dig-MNIST.csv') # the test set does not contain a label, which makes it useless for us for evaluation, we use Dig-MNIST for test set

x_train = train_.iloc[:, 1:].to_numpy()
x_test = test_.iloc[:, 1:].to_numpy()

x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

n_train = x_train.shape[0]
n_test = x_test.shape[0]

n_x = x_train.shape[1]
n_y = x_train.shape[2]
nn = n_x * n_y


def distribute():
    pass


def distort(x, d):
    # additive Gaussian noise on the input image, etc.

    # mat.subplot(121)
    # mat.imshow(x, cmap='Greys')

    # flipping horizontally
    if d[0]:
        x = num.flipud(x)
    # flipping vertically
    if d[1]:
        x = num.fliplr(x)
    # rotating
    if d[2]:
        x = num.rot90(x, k=ran.randint(1, 4))
    # gaussian noise
    if d[3]:
        x = x + ran.normal(loc=0.0, scale=20.0, size=(n_x, n_y))
    # brightness
    if d[4]:
        xx, yy = num.meshgrid(num.arange(n_x), num.arange(n_y))
        z = num.sqrt((xx - ran.randint(n_x)) ** 2 + (yy - ran.randint(n_y)) ** 2)
        x = x + ran.randint(80) / (1 + 0.1 * z)
    # occlusion
    if d[5]:
        s = ran.randint([0, n_x / 4, 0, n_y / 4], [n_x / 2 + 1, n_x / 2 + 1, n_y / 2 + 1, n_y / 2 + 1])
        x[s[0]: s[0] + s[1], s[2]: s[2] + s[3]] = 255

    # mat.subplot(122)
    # mat.imshow(x, cmap='Greys')
    # mat.title(d)
    # mat.show()
    return x



"""
c = [14, 14]
x, y = num.meshgrid(num.arange(n_x), num.arange(n_y))
z = num.sqrt((x - c[0])**2 + (y - c[1])**2)
print(x)
print(y)
mat.imshow(z)
mat.show()
b = ran.randint(30) / (1 + z)
mat.imshow(b)
mat.show()
"""


distortions = ['up-down flip', 'left-right flip', 'rotation by 90??, 180??, or 270??',
               'gaussian noise', 'brightness gradient', 'occlusion']
x_dist = []
d_dist = []
x_dist_test = []
d_dist_test = []


def addDistortion3(x_dist_full):
    x_dist = []
    x_dist_test = []
    for i in range(n_train):
        dist = num.zeros(6)
        while max(dist) == 0:
            dist[ran.randint(2)] = 1
            dist[3 + ran.randint(2)] = 1
            dist[5] = ran.randint(2)
        x_dist.append(distort(x_train[i], dist))
        d_dist.append(dist)

    for i in range(n_test):
        dist = num.zeros(6)
        while max(dist) == 0:
            dist[ran.randint(2)] = 1
            dist[3 + ran.randint(2)] = 1
            dist[5] = ran.randint(2)
        x_dist_test.append(distort(x_test[i], dist))
        d_dist_test.append(dist)

    x_dist = num.rint(num.clip(x_dist, 0, 255)).astype(num.uint8)
    x_dist_test = num.rint(num.clip(x_dist_test, 0, 255)).astype(num.uint8)

    num.save('data/distorted/X_kannada_MNIST_train_multipl_distorted.npy', x_dist)
    num.save('data/distorted/X_kannada_MNIST_test_multipl_distorted.npy', x_dist_test)
    x_dist_full.extend(x_dist)


def addDistortion1(x_dist_full):
    x_dist = []
    x_dist_test = []
    for i in range(n_train):
        dist = num.zeros(6)
        while max(dist) == 0:
            dist[ran.randint(6)] = 1
        x_dist.append(distort(x_train[i], dist))
        d_dist.append(dist)

    for i in range(n_test):
        dist = num.zeros(6)
        while max(dist) == 0:
            dist[ran.randint(6)] = 1
        x_dist_test.append(distort(x_test[i], dist))
        d_dist_test.append(dist)

    x_dist = num.rint(num.clip(x_dist, 0, 255)).astype(num.uint8)
    x_dist_test = num.rint(num.clip(x_dist_test, 0, 255)).astype(num.uint8)

    num.save('data/distorted/X_kannada_MNIST_train_single_distorted.npy', x_dist)
    num.save('data/distorted/X_kannada_MNIST_test_single_distorted.npy', x_dist_test)
    x_dist_full.extend(x_dist)

addDistortion3(x_dist)
addDistortion1(x_dist)


d_all = num.concatenate((num.array(d_dist), num.array(d_dist_test)), axis=0)

# uncomment to view the distribution of distortions

mat.hist(num.sum(d_all, axis=1))
mat.title('number of distortions')
mat.show()
mat.hist(d_all, label=distortions)
mat.title('type of distortions')
mat.legend()
mat.show()

for i in range(30):
    mat.subplot(121)
    mat.imshow(x_train[i], cmap='Greys')
    mat.subplot(122)
    mat.imshow(x_dist[i], cmap='Greys')
    mat.title(d_dist[i])
    mat.show()

