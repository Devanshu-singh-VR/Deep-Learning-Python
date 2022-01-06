import numpy as np

train = np.array([[1, 2, 3], [2, 3, 4], [4, 5, 6]], dtype='float')
label = np.array([[1, 0], [0, 1], [1, 0]], dtype='float')
hidden_size = 124
weight1 = np.random.rand(train.shape[1], hidden_size)
weight2 = np.random.rand(hidden_size, 2)
bias1 = np.random.rand(1, hidden_size)
bias2 = np.random.rand(1, 2)

def softmax(z):
    return (np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True))

def sigmoid(z):
    return 1/(1+np.exp(-z))

def diff_sigmoid(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def forward(a0, w1, w2, b1, b2, keep_prob):
    z1 = a0.dot(w1) + b1

    # dropout
    dp = (np.random.rand(z1.shape[0], z1.shape[1]) < keep_prob) * 1
    a1 = sigmoid(z1) * dp
    a1 = a1 / keep_prob # we don't wat to change the value of a1
    # because it will affect the hypo and z2 value,
    # ( after which model will try to regain those value by (may be changing weights) of something else)
    z2 = a1.dot(w2) + b2
    hypo = softmax(z2)
    ak = softmax(z2)
    return hypo, ak, z2, a1, a0

def cost(real, pred):
    loss = -sum(real * np.log(pred))
    return sum(loss)

iteration = 10000
alp = 0.1

for itr in range(iteration):
    hypo, ak, z2, a1, a0 = forward(train, weight1, weight2, bias1, bias2, keep_prob=1)

    ak[label==1] -= 1

    delta2 = (1 / hypo.shape[0]) * ak
    dw2 = sigmoid(train.dot(weight1)).T.dot(delta2)
    db2 = np.ones((a1.shape[0], 1)).T.dot(delta2) * (1 / hypo.shape[0])

    delta1 = delta2.dot(weight2.T) * (diff_sigmoid(train.dot(weight1)))
    dw1 = train.T.dot(delta1)
    db1 = np.ones((a0.shape[0], 1)).T.dot(delta1) * (1 / hypo.shape[0])

    weight1 -= alp * dw1
    weight2 -= alp * dw2
    bias1 -= alp * db1
    bias2 -= alp * db2

    print(cost(label, hypo))

print(hypo)