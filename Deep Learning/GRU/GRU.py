import numpy as np

# training data
train = np.array([[[2, 9, 1, 1], [2, 1, 1, 2], [1, 1, 1, 1]], [[2, 9, 1, 1], [2, 1, 1, 1], [1, 2, 3, 4]]])
label = np.array([[[1], [1], [0]], [[1], [1], [0]]])
hidden_layer_size = 10
hidden_unit_size = hidden_layer_size
hidden_unit = np.zeros((train.shape[0], 1, hidden_unit_size))

# weights
weight_update = np.random.rand(train.shape[2]+hidden_unit.shape[2], hidden_layer_size)
weight_reset = np.random.rand(train.shape[2]+hidden_unit.shape[2], hidden_layer_size)
weight_hx = np.random.rand(train.shape[2]+hidden_unit.shape[2], hidden_layer_size)
weight_y = np.random.rand(hidden_layer_size, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def diff_sigmoid(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def tanh(z):
    return(np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

def diff_tanh(z):
    return 4/((np.exp(z)+np.exp(-z))**2)

def forward(a, wu, wr, whx, wy, h):
    z1 = np.concatenate((a, h), axis=1)
    update = sigmoid(z1.dot(wu))
    reset = sigmoid(z1.dot(wr))
    reset = np.concatenate((np.ones((h.shape[0], a.shape[1])), reset), axis=1)
    a1_to = tanh((z1*reset).dot(whx))
    a1 = ((1 - update) * a1_to) + (update * h)
    z2 = a1.dot(wy)
    hypo = sigmoid(z2)

    return hypo, z1, update, reset, a1_to, a1, z2

def cost(real, hypo):
    return (0.5/hypo.shape[0]) * np.sum((real - hypo)**2, axis=(0, 1))

iteration = 1000

for itr in range(iteration):
    loss = 0
    hidden = hidden_unit[:, 0, :]
    for i in range(train.shape[1]):
        hypo, z1, update, reset, a1_to, a1, z2 = forward(train[:, i, :], weight_update, weight_reset, weight_hx, weight_y, hidden)

        delta_y = (1 / train.shape[0]) * (label[:, i, :] - hypo) * diff_sigmoid(z2)
        delta_hx = delta_y.dot(weight_y.T) * diff_sigmoid(z2) * (1-update) * diff_tanh((z1*reset).dot(weight_hx))
        delta_wr = delta_hx * diff_sigmoid(z1.dot(weight_reset))
        delta_wu = delta_y.dot(weight_y.T) * (hidden - a1_to) * diff_sigmoid(z1.dot(weight_update))

        dwy = -a1.T.dot(delta_y)
        dwhx = -(z1*reset).T.dot(delta_hx)
        dwr = -z1.T.dot(delta_wr)
        dwu = -z1.T.dot(delta_wu)

        weight_y -= dwy
        weight_hx -= dwhx
        weight_reset -= dwr
        weight_update -= dwu

        loss += cost(label[:, i, :], hypo)

        hidden = a1

    print(loss)

# prediction
hidden = hidden_unit[:, 0, :]
pred = np.zeros((2, 3, 1))
for i in range(train.shape[1]):
    hypo, z1, update, reset, a1_to, a1, z2 = forward(train[:, i, :], weight_update, weight_reset, weight_hx, weight_y, hidden)
    hidden = a1

    pred[:, i] = hypo

print(pred)