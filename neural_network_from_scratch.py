import time
import numpy as np
import pandas as pd
import sys

import load_german_credit_data
import load_titanic_data


class Layer_Dense:
    """
    A hidden layer in the Neural Network.
    """

    def __init__(self, n_inputs, n_neurons):
        self.n_neurons = n_neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.learnable_weights = self.weights.size
        self.learnable_biases = self.biases.size
        self.learnable_parameters = self.weights.size + self.biases.size

    # Computes the output of the layer. Illustration of dot product: 
    # https://upload.wikimedia.org/wikipedia/commons/b/bf/Fully_connected_neural_network_and_it%27s_expression_as_a_tensor_product.jpg 
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    # Computes the derivatives from with dvalues from Activation Function
    def backward(self, dvalues):

        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Layer_Input:
    """
    First layer which we feed input data into.
    """

    def forward(self, inputs):
        self.output = inputs


class Activation_ReLU:
    """
    ReLU Activation class to pipe output of layers through.
    """

    def forward(self, inputs):
        self.inputs = inputs # values will be used for derivative
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):

        self.dinputs = dvalues.copy()

        self.dinputs[self.inputs <= 0] = 0 # derivative of ReLU function 

    def predictions(self, outputs):
        return outputs


class Activation_Softmax:
    """
    Softmax Activation class.
    """

    def forward(self, inputs):
        self.inputs = inputs

        exp_scaled = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        self.output = exp_scaled / (np.sum(exp_scaled, axis=1, keepdims=True))

    def backward(self, dvalues):

        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in \
                enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix,
                                         single_dvalues)

    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    
class Activation_Sigmoid:
    """
    Sigmoid Activation class.
    """


    def forward(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        self.dinputs = dvalues * (self.output * (1 - self.output))

    def predictions(self, outputs):
        return outputs


class Optimizer_SGD:
    """
    SGD Optimizer class to update weights and biases in layers.
    """

    def __init__(self, learning_rate=1.):
        self.learning_rate = learning_rate

    def update_params(self, layer):

        layer.weights += -self.learning_rate * layer.dweights
        layer.biases +=  -self.learning_rate * layer.dbiases



class Loss_CategoricalCrossentropy:

    def calculate(self, output, y):

        sample_losses = self.forward(output, y)

        return np.mean(sample_losses)
    def forward(self, y_pred, y_true):

        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        # y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        y_pred_clipped = y_pred

        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]

        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self, model_output, y_true):

        samples = len(model_output)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(model_output[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / model_output
        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Accuracy_Categorical:
    """
    Calculates the accuracy of model output for categorical values compared to ground truth values.
    """

    def __init__(self, binary=False):
        self.binary = binary


    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        return accuracy

    def precision(self, predictions, y, class_index=1):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        true_positives = np.sum((predictions == class_index) & (y == class_index))
        predicted_positives = np.sum(predictions == class_index)

        if predicted_positives == 0:
            return 0.0
        return true_positives / predicted_positives

    def recall(self, predictions, y, class_index=1):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        true_positives = np.sum((predictions == class_index) & (y == class_index))
        actual_positives = np.sum(y == class_index)

        if actual_positives == 0:
            return 0.0
        return true_positives / actual_positives

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    
class Model:
    """
    Creates Neural Network Model object. Usage examples at end of file.
    """

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
        self.real_layers = []


    def add(self, layer):

        if isinstance(layer, Layer_Dense) or isinstance(layer, Layer_Input):
            self.real_layers.append(layer)

        if isinstance(layer, Activation_ReLU) or isinstance(layer, Activation_Sigmoid):
            self.activation_class = layer.__class__.__name__

        if isinstance(layer, Activation_Softmax):
            self.output_activation_class = layer.__class__.__name__

        self.layers.append(layer)

    def set(self, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy


    def compute_vram_usage(self):
        total_bytes = 0
        for layer in self.trainable_layers:
            weight_bytes = layer.weights.size * layer.weights.dtype.itemsize
            bias_bytes = layer.biases.size * layer.biases.dtype.itemsize
            total_bytes += weight_bytes + bias_bytes

        return total_bytes


    def finalize(self):

        self.input_layer = Layer_Input() # first input layer

        layer_count = len(self.layers)

        self.trainable_layers = []

        total_trainable_parameters = 0
        total_trainable_weights = 0
        total_trainable_biases = 0

        for i in range(layer_count):

            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights') and hasattr(self.layers[i], 'biases'):
                self.trainable_layers.append(self.layers[i])
                total_trainable_biases += self.layers[i].learnable_biases
                total_trainable_weights += self.layers[i].learnable_weights
                total_trainable_parameters += self.layers[i].learnable_weights + self.layers[i].learnable_biases

        vram_usage = self.compute_vram_usage()

        
        self.log_model_info()
        print(f"Number of total trainable parameters: {total_trainable_parameters}\n\tof that weights: {total_trainable_weights}\n\tof that biases: {total_trainable_biases}")
        print(f"Memory usage of model (in bytes): {vram_usage}")

    def log_model_info(self):
        print("Model Info:")
        print(f"Layer structure (excluding input layer):")

        for i, layer in enumerate(self.real_layers):
            print(f"Layer {i}:\tNumber of Neurons: {layer.n_neurons}" )

        print(f"Model Activation Function: {self.activation_class}" )
        print(f"Model Output Activation Function: {self.output_activation_class}" )
        
    def fit(self, X, y, epochs=1, print_every=1):


        for epoch in range(1, epochs+1):

            output = self.forward(X)

            loss = self.loss.calculate(output, y)

            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backward(output, y)

            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)

            if not epoch % print_every:
                print(f'Epoch: {epoch}, ' +
                      f'Accuracy: {accuracy:.3f}, ' +
                      f'Loss: {loss:.3f}, ' +
                      f'Learning rate: {self.optimizer.learning_rate}')

    def predict(self, validation_data, output_file=None):
        X_val, y_val = validation_data

        output = self.forward(X_val)

        loss = self.loss.calculate(output, y_val)

        predictions = self.output_layer_activation.predictions(output)
        accuracy = self.accuracy.calculate(predictions, y_val)
        precision = self.accuracy.precision(predictions, y_val)
        recall = self.accuracy.recall(predictions, y_val)

        if output_file is not None:
            np.savetxt(f'{output_file}', predictions, delimiter=',', fmt='%d', header='Predicted_Survived', comments='')

        print(f'Test Set Results: ' +
                f'Accuracy: {accuracy:.3f}, ' +
                f'Precision: {precision:.3f}, ' +
                f'Recall: {recall:.3f}, ' +
                f'Loss: {loss:.3f}')
        
        return accuracy

    def forward(self, X):

        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output

    def backward(self, output, y):

        self.loss.backward(output, y) # calcualte derivative for loss
        
        for layer in reversed(self.layers): # last layer points to the loss object
            layer.backward(layer.next.dinputs)

class Grid_Search:
    """
    Performs a grid search over specified hyperparameters for a neural network model.
    """

    def run_grid_search(self,
                        X_train, y_train, X_test, y_test,
                        neurons_per_layer_options,
                        epochs_options,
                        hidden_activation_classes,
                        learning_rate_options,
                        print_every_train=100,
                        output_file_prefix="predictions"):

        input_shape = X_train.shape[1]
        output_classes = len(np.unique(y_train))

        count = 0
        highest_accuracy = 0
        highest_output = ""
        
        original_stdout = sys.stdout
        for layers_neurons in neurons_per_layer_options:
            for epochs_train in epochs_options:
                for activation_name, Activation_Class in hidden_activation_classes.items():
                    for learning_rate in learning_rate_options:

                        model = Model()
                        start_time = time.time()
                        current_input_size = input_shape

                        for i, layer_size in enumerate(layers_neurons):
                            model.add(Layer_Dense(current_input_size, layer_size))
                            model.add(Activation_Class())
                            current_input_size = layer_size

                        model.add(Layer_Dense(current_input_size, output_classes))
                        model.add(Activation_Softmax())

                        model.set(
                            loss=Loss_CategoricalCrossentropy(),
                            optimizer=Optimizer_SGD(learning_rate=learning_rate),
                            accuracy=Accuracy_Categorical()
                        )

                        output_file=f"{output_file_prefix}_{count}_L{'-'.join(map(str, layers_neurons))}_E{epochs_train}_A{activation_name}_LR{str(learning_rate).replace('.', '')}"
                        with open(f"{output_file}.txt", 'w') as f:
                            sys.stdout = f

                            model.finalize()

                            model.fit(X_train, y_train, epochs=epochs_train, print_every=print_every_train)

                            accuracy = model.predict((X_test, y_test),
                                            output_file=f"{output_file}.csv")
                            end_time = time.time()

                            print(f"Processing time: {((end_time - start_time)* 1000):.3f} ms")
                            
                            if (accuracy > highest_accuracy):
                                highest_accuracy = accuracy
                                highest_output = output_file
                            
                            count += 1
                        sys.stdout = original_stdout

        with open(f"{highest_output}.txt", "r") as f:
            print("Model with highest accuracy: ")
            print(f.read())


X_train, y_train, X_test, y_test = load_titanic_data.load_titanic_dataset()

neurons_per_layer = [
    [32],
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [64, 64, 64]
]

epochs = [100, 500, 1000]

grid_search = Grid_Search()
grid_search.run_grid_search(X_train, y_train, X_test, y_test, neurons_per_layer, epochs, 
                            {'ReLU': Activation_ReLU, 'Sigmoid': Activation_Sigmoid}, 
                            [1], output_file_prefix="./titanic-predictions/prediction")


X_train, y_train, X_test, y_test = load_german_credit_data.load_german_credit_data_dataset()

neurons_per_layer = [
    [32],
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [64, 64, 64]
]

epochs = [100, 500, 1000]

grid_search = Grid_Search()
grid_search.run_grid_search(X_train, y_train, X_test, y_test, neurons_per_layer, epochs, 
                            {'ReLU': Activation_ReLU, 'Sigmoid': Activation_Sigmoid}, 
                            [1], output_file_prefix="./german-credit-predictions/prediction")