import numpy as np

train = np.array([[[2, 9, 1, 1], [2, 1, 1, 2], [1, 1, 1, 1]], [[2, 9, 1, 1], [2, 1, 1, 1], [1, 2, 3, 4]]])
label = np.array([[[1], [1], [0]], [[1], [1], [0]]])
hidden_layer_size = 10
hidden_unit_size = hidden_layer_size
hidden_unit = np.zeros((train.shape[0], 1, hidden_unit_size))
weight_x = np.random.rand(train.shape[2], hidden_layer_size)
weight_h = np.random.rand(hidden_unit_size, hidden_layer_size)
weight_y = np.random.rand(hidden_layer_size, 1)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def diff_sigmoid(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

def forward(a, wx, wh, wy, h):
    z1_h = h.dot(wh)
    z1_a = a.dot(wx)
    a1 = sigmoid(z1_h + z1_a)
    z2 = a1.dot(wy)
    hypo = sigmoid(z2)

    return hypo, z1_a, z1_h, a1, z2

def cost(real, hypo):
    return (0.5/hypo.shape[0]) * np.sum((real - hypo)**2, axis=(0, 1))

iteration = 1000

for itr in range(iteration):
    loss = 0
    hidden = hidden_unit[:, 0, :]
    for i in range(train.shape[1]):
        hypo, z1_a, z1_h, a1, z2 = forward(train[:, i, :], weight_x, weight_h, weight_y, hidden)

        delta_y = (1 / train.shape[0]) * (label[:, i, :] - hypo) * diff_sigmoid(z2)
        dwy = -a1.T.dot(delta_y)

        delta_a = delta_y.dot(weight_y.T) * diff_sigmoid(z1_a+z1_h)
        delta_h = delta_y.dot(weight_y.T) * diff_sigmoid(z1_h+z1_a)

        dwa = -train[:, i, :].T.dot(delta_a)
        dwh = -hidden.T.dot(delta_h)

        weight_h -= dwh
        weight_x -= dwa
        weight_y -= dwy

        hidden = a1

        loss += cost(label[:, i, :], hypo)

    print(loss)

# prediction
hidden = hidden_unit[:, 0, :]
pred = np.zeros((2, 3, 1))
for i in range(train.shape[1]):
    hypo, z1_a, z1_h, a1, z2 = forward(train[:, i, :], weight_x, weight_h, weight_y, hidden)
    hidden = a1

    pred[:, i] = hypo

print(pred)