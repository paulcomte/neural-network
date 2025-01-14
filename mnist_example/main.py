import numpy as np
from keras.datasets import mnist

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-8), axis=1))

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]

def accuracy(y_true, y_pred):
    return np.mean(np.argmax(y_true, axis=1) == np.argmax(y_pred, axis=1))


class Conv2D:
    def __init__(self, num_filters, kernel_size, input_shape, padding='valid'):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.input_shape = input_shape
        self.padding = padding
        self.weights = np.random.randn(num_filters, kernel_size, kernel_size, input_shape[-1]) * 0.1
        self.biases = np.zeros((num_filters, 1))
        self.mask = None

    def forward(self, x):
        self.input = x
        batch_size, height, width, channels = x.shape
        out_height = height // self.kernel_size
        out_width = width // self.kernel_size
        self.output = np.zeros((batch_size, out_height, out_width, self.num_filters))
        self.mask = np.zeros_like(x) 

        for i in range(out_height):
            for j in range(out_width):
                region = x[:, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size, :]

                max_vals = np.max(region, axis=(1, 2), keepdims=True) 
                self.output[:, i, j, :] = max_vals.reshape(batch_size, -1)

                self.mask[:, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size, :] = (region == max_vals)

        return self.output

    
    def backward(self, grad_output, learning_rate=None):
        grad_input = np.zeros_like(self.input)
        out_height, out_width = grad_output.shape[1:3]
        for i in range(out_height):
            for j in range(out_width):
                grad_region = grad_output[:, i, j, :, None, None] * self.mask[:, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size, :]
                grad_input[:, i*self.kernel_size:(i+1)*self.kernel_size, j*self.kernel_size:(j+1)*self.kernel_size, :] += grad_region
        
        if learning_rate:
            self.weights -= learning_rate * np.dot(self.input.T, grad_input)
            self.biases -= learning_rate * np.sum(grad_input, axis=0, keepdims=True)

        return grad_input

class MaxPooling2D:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, x):
        self.input = x
        batch_size, height, width, channels = x.shape
        out_height = height // self.pool_size
        out_width = width // self.pool_size
        self.output = np.zeros((batch_size, out_height, out_width, channels))
        self.mask = np.zeros_like(x)
        
        for i in range(out_height):
            for j in range(out_width):
                region = x[:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, :]
                self.output[:, i, j, :] = np.max(region, axis=(1, 2))
                # Update the mask correctly
                for c in range(channels):
                    max_vals = np.max(region[:, :, :, c], axis=(1, 2), keepdims=True)
                    self.mask[:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, c] = (region[:, :, :, c] == max_vals)
        return self.output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.input)
        batch_size, out_height, out_width, channels = grad_output.shape
        
        for i in range(out_height):
            for j in range(out_width):
                for c in range(channels):
                    grad_input[:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, c] += grad_output[:, i, j, c][:, None, None] * self.mask[:, i*self.pool_size:(i+1)*self.pool_size, j*self.pool_size:(j+1)*self.pool_size, c]
        return grad_input

# Dense Layer
class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros((1, output_size))

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, grad_output, learning_rate):
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * np.dot(self.input.T, grad_output)
        self.biases -= learning_rate * np.sum(grad_output, axis=0, keepdims=True)
        return grad_input

class Flatten:
    def forward(self, x):
        self.input_shape = x.shape
        flattened = x.reshape(x.shape[0], -1)
        print(f"Flatten: input shape {self.input_shape}, flattened shape {flattened.shape}")
        return flattened

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)

class Dropout:
    def __init__(self, rate):
        self.rate = rate

    def forward(self, x):
        self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
        return x * self.mask

    def backward(self, grad_output):
        return grad_output * self.mask

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
y_train = one_hot_encode(y_train, 10)
y_test = one_hot_encode(y_test, 10)

conv1 = Conv2D(32, 3, input_shape=(28, 28, 1), padding='same')
pool1 = MaxPooling2D(2)
conv2 = Conv2D(64, 3, input_shape=(13, 13, 32), padding='same')
pool2 = MaxPooling2D(2)
flatten = Flatten()
dense1 = Dense(3136, 10)

def forward(x):
    print(f"Initial input shape: {x.shape}")
    x = conv1.forward(x)
    print(f"After conv1: {x.shape}")
    x = pool1.forward(x)
    print(f"After pool1: {x.shape}")
    x = conv2.forward(x)
    print(f"After conv2: {x.shape}")
    x = pool2.forward(x)
    print(f"After pool2: {x.shape}")
    x = flatten.forward(x)
    print(f"After flatten: {x.shape}")
    x = dense1.forward(x)
    return softmax(x)

def backward(loss_grad):
    grad = dense1.backward(loss_grad, learning_rate=0.01)
    grad = flatten.backward(grad)
    grad = pool2.backward(grad)
    grad = conv2.backward(grad, learning_rate=0.01)
    grad = pool1.backward(grad)
    grad = conv1.backward(grad, learning_rate=0.01)

epochs = 5
batch_size = 128
for epoch in range(epochs):
    for i in range(0, x_train.shape[0], batch_size):
        x_batch = x_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]

        predictions = forward(x_batch)

        loss = cross_entropy_loss(y_batch, predictions)

        loss_grad = predictions - y_batch
        backward(loss_grad)

    test_predictions = forward(x_test)
    test_accuracy = accuracy(y_test, test_predictions)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
