import numpy as num
# import matplotlib.pyplot as mat
import numpy.random as ran

path = 'C:\\Users\\Felix\\PycharmProjects\\deeplearning\\Kannada_MNIST-master\\data\\output_tensors\\MNIST_format\\'

x_train = num.load(path + 'X_kannada_MNIST_train.npy')
x_test = num.load(path + 'X_kannada_MNIST_test.npy')

# print(x_train.shape)
# print(x_test.shape)

n_train = x_train.shape[0]
n_test = x_test.shape[0]

n_x = x_train.shape[1]
n_y = x_train.shape[2]
nn = n_x * n_y


def distribute():
    pass


def distort(x, d):
    # additive Gaussian noise on the input image, etc.

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
    # occlusion
    if d[3]:
        s = ran.randint([0, n_x / 4, 0, n_y / 4], [n_x / 2 + 1, n_x / 2 + 1, n_y / 2 + 1, n_y / 2 + 1])
        x[s[0]: s[0] + s[1], s[2]: s[2] + s[3]] = 255
    # gaussian noise
    if d[4]:
        x = x + ran.normal(loc=0.0, scale=30.0, size=(n_x, n_y))
    # brightness
    if d[5]:
        x = x * ran.rand()

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


distortions = ['left-right flip', 'up-down flip', 'rotation by 90°, 180°, or 270°',
               'occlusion', 'gaussian noise', 'brightness gradient (unimplemented)']
x_dist = []
d_dist = []
x_dist_test = []
d_dist_test = []

for i in range(n_train):
    dist = num.zeros(6)
    while max(dist) == 0:
        dist[ran.randint(3)] = 1
        dist[3] = int(num.round(ran.random()))
        dist[4] = int(num.round(ran.random()))
    x_dist.append(distort(x_train[i], dist))
    d_dist.append(dist)

for i in range(n_test):
    dist = num.zeros(6)
    while max(dist) == 0:
        dist[ran.randint(3)] = 1
        dist[3] = int(num.round(ran.random()))
        dist[4] = int(num.round(ran.random()))
    x_dist_test.append(distort(x_test[i], dist))
    d_dist_test.append(dist)

num.save(path + 'X_kannada_MNIST_train_distorted.npy', x_dist)
num.save(path + 'X_kannada_MNIST_test_distorted.npy', x_dist_test)

d_all = num.concatenate((num.array(d_dist), num.array(d_dist_test)), axis=0)

# uncomment to view the distribution of distortions
"""
mat.hist(num.sum(d_all, axis=1))
mat.title('number of distortions')
mat.show()
mat.hist(d_all, label=distortions)
mat.title('type of distortions')
mat.legend()
mat.show()
"""
