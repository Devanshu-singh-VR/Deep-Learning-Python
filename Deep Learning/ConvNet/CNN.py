import numpy as np

train = np.random.rand(3, 5, 5)
label = np.array([[0.4], [0.8], [0.9]])
weight1 = np.random.rand(3, 3)

linear_wt = train.shape[1]-weight1.shape[0]+1
weight2 = np.random.rand(linear_wt*linear_wt, 1)

def sigmoid(z):
    return 1/(1+np.exp(-z))

def diff_sigmoid(z):
    return (np.exp(-z)/((1+np.exp(-z))**2)).reshape(z.shape[0], -1)

def diff_relu(z):
    a = np.ones(z.shape)
    a[z<=0] -= 1
    return a.reshape(a.shape[0], -1)

def filter_cal(x, w):
    nl = x.shape[1]-w.shape[0]+1
    z1 = np.zeros((x.shape[0], nl, nl))
    for k in range(x.shape[0]):
        for i in range(nl):
            for j in range(nl):
                z1[k, i, j] = np.sum(x[k, i:w.shape[0]+i, j:w.shape[0]+j] * w, axis=(0, 1))
    return z1

def op_filter_cal(x, f):
    nl = x.shape[1]-f.shape[1]+1
    wt = np.zeros((nl, nl))
    for i in range(nl):
        for j in range(nl):
            wt[i, j] = np.sum(x[:, i:f.shape[1]+i, j:f.shape[1]+j] * f, axis=(0, 1, 2))

    return wt

def forward(a0, w1, w2):
    '''
    z1 = np.concatenate(
        (np.expand_dims(filter_cal(a0, w1), axis=-1),
         np.expand_dims(filter_cal(a0, w2), axis=-1)
         ),
        axis=3)
    '''

    z1 = filter_cal(a0, w1)
    a1 = sigmoid(z1)
    linear1 = a1.reshape(a1.shape[0], -1)
    z2 = linear1.dot(w2)
    hypo = sigmoid(z2)

    return hypo, z2, linear1, z1

def cost(real, pred):
    return (1 / (2*real.shape[0])) * sum((real - pred)**2)

iteration = 2000

for itr in range(iteration):
    hypo, z2, linear, z1 = forward(train, weight1, weight2)

    delta2 = (1 / train.shape[0]) * (label - hypo) * diff_sigmoid(z2)
    dw2 = -linear.T.dot(delta2)

    delta1 = (delta2.dot(weight2.T) * diff_sigmoid(z1)).reshape(3, 3, 3)
    dw1 = -op_filter_cal(train, delta1)

    weight2 -= dw2
    weight1 -= dw1

    print(cost(label, hypo))

print(hypo)
