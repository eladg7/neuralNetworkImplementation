import sys
import numpy as np

MINI_BATCH_SIZE = 200
LEARNING_RATE = 0.75
EPOCHS = 100
HIDDEN_LAYER_SIZE = 41


def read_file(train_x_name, train_y_name, test_x_name):
    # for each x normalize it and then add it to the list
    with open(train_x_name, 'r') as file:
        train_x = []
        for line in file.read().strip().split('\n'):
            v = np.fromstring(line, dtype=float, sep=' ')
            train_x.append(v)
    with open(train_y_name, 'r') as file:
        train_y = np.fromstring(file.read().strip(), dtype=int, sep=' ')
        train_y = one_hot_encoding(train_y)
    with open(test_x_name, 'r') as file:
        test_x = []
        for line in file.read().strip().split('\n'):
            v = np.fromstring(line, dtype=float, sep=' ')
            test_x.append(v)
    return np.array(train_x), train_y, np.array(test_x)


def add_bias(train_x, test_x):
    # add bias for both train x and test x
    train_x = np.c_[train_x, np.ones(len(train_x)).astype(int)]
    test_x = np.c_[test_x, np.ones(len(test_x)).astype(int)]
    return train_x, test_x


def predict(train_set_x, train_set_y, test_set_x, epochs):
    params = train(train_set_x, train_set_y, epochs)
    fprop_cache = forward_propagation(test_set_x, params)
    return fprop_cache['h2'].T


def train(train_x, train_y, epochs):
    permutation = np.random.permutation(train_x.shape[1])
    # initiate the weights and b with random values
    train_x_len = len(train_x)
    params = {'w1': np.random.randn(HIDDEN_LAYER_SIZE, train_x_len) * np.sqrt(1. / train_x_len),
              'b1': np.ones((HIDDEN_LAYER_SIZE, 1)) * np.sqrt(1. / train_x_len),
              'w2': np.random.randn(10, HIDDEN_LAYER_SIZE) * np.sqrt(1. / HIDDEN_LAYER_SIZE),
              'b2': np.ones((10, 1)) * np.sqrt(1. / HIDDEN_LAYER_SIZE)}
    train_x_shuffled = train_x[:, permutation]
    train_y_shuffled = train_y[:, permutation]
    batches_count = int(np.ceil(train_x.shape[1] / (MINI_BATCH_SIZE * 1.)))
    for epoch in range(epochs):
        for i in range(batches_count):
            start = i * MINI_BATCH_SIZE
            end = min(start + MINI_BATCH_SIZE, train_x.shape[1] - 1)
            batch_array_x = train_x_shuffled[:, start:end]
            batch_array_y = train_y_shuffled[:, start:end]

            fprop_cache = forward_propagation(batch_array_x, params)
            gradients = back_propagation(batch_array_x, batch_array_y, fprop_cache)

            # update weights and b by gradients
            params['w1'] -= LEARNING_RATE * gradients['dW1']
            params['b1'] -= LEARNING_RATE * gradients['db1']
            params['w2'] -= LEARNING_RATE * gradients['dW2']
            params['b2'] -= LEARNING_RATE * gradients['db2']
    return params


def one_hot_encoding(y):
    digits = 10
    examples = y.shape[0]
    y = y.reshape(1, examples)
    Y_new = np.eye(digits)[y.astype(int)]
    Y_new = Y_new.T.reshape(digits, examples)
    return Y_new.astype(int)


def back_propagation(mini_batch_x, mini_batch_y, fprop_cache):
    dz2 = (fprop_cache['h2'] - mini_batch_y)  # dL/dz2
    dW2 = (1 / MINI_BATCH_SIZE) * np.matmul(dz2, fprop_cache['h1'].T)  # dL/dz2 * dz2/dW2
    db2 = (1 / MINI_BATCH_SIZE) * np.sum(dz2, axis=1, keepdims=True)  # dL/dz2 * dz2/db2

    dH1 = np.matmul(fprop_cache['w2'].T, dz2)  # dL/dz2 * dz2/dh1
    z1_sigmoid = sigmoid(fprop_cache['z1'])
    dz1 = dH1 * z1_sigmoid * (1 - z1_sigmoid)  # dL/dz2 * dz2/dh1 * dh1/dz1

    dW1 = (1 / MINI_BATCH_SIZE) * np.matmul(dz1, mini_batch_x.T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
    db1 = (1 / MINI_BATCH_SIZE) * np.sum(dz1, axis=1, keepdims=True)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1

    return {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}


def forward_propagation(mini_batch, params):
    # create w: each array represents a weights matrix for the ith layer
    fprop_cache = {'z1': np.matmul(params['w1'], mini_batch) + params['b1']}
    fprop_cache['h1'] = sigmoid(fprop_cache['z1'])
    fprop_cache['z2'] = np.matmul(params['w2'], fprop_cache['h1']) + params['b2']
    fprop_cache['h2'] = softmax(fprop_cache['z2'])
    for key in params:
        fprop_cache[key] = params[key]
    return fprop_cache


def softmax(x):
    # do next line in order to avoid overflow
    temp_x = x - np.max(x)
    return np.exp(temp_x) / np.sum(np.exp(temp_x), axis=0)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    if len(sys.argv) < 3:
        print('Not enough arguments were supplied!')
        return
    train_x, train_y, test_x = read_file(*sys.argv[1:])
    # normalize data
    train_x = np.true_divide(train_x, 255)
    test_x = np.true_divide(test_x, 255)
    train_x = train_x.T
    test_x = test_x.T

    predictions = predict(train_x, train_y, test_x, EPOCHS)
    write_to_file(predictions)
    # evaluate.evaluate_algorithm(train_x, train_y, predict, 5, EPOCHS)


def write_to_file(predictions):
    # convert the one hot encoding to a number
    content = [str(np.argmax(row)) for row in predictions]
    with open('test_y', 'w') as file:
        file.write('\n'.join(content))


if __name__ == '__main__':
    main()
