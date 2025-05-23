import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import pandas as pd
import sys

nnfs.init()

class Layer_Dense:
    """
    A hidden layer in the Neural Network.
    """

    def __init__(self, n_inputs, n_neurons):
        # Random weights
        # Biases set to 0
        self.n_neurons = n_neurons
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.learnable_weights = self.weights.size
        self.learnable_biases = self.biases.size
        self.learnable_parameters = self.weights.size + self.biases.size

    # Computes the output of the layer. Illustration of dot product: 
    # https://upload.wikimedia.org/wikipedia/commons/b/bf/Fully_connected_neural_network_and_it%27s_expression_as_a_tensor_product.jpg 
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases


    # Performs the backward pass, calculating gradients for weights, biases, and inputs. These gradients will be used by the optimizer.
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)


# Dropout
class Layer_Dropout:
    """
    A hidden layer in the Neural Network, that randomly drops out neuron outputs to reduce overfitting.
    """

    def __init__(self, rate):
        # Store rate, we invert it as for example for dropout
        # of 0.1 we need success rate of 0.9
        self.rate = 1 - rate

    # Forward pass
    def forward(self, inputs, training):
        # Save input values
        self.inputs = inputs

        # If not in the training mode - return values
        if not training:
            self.output = inputs.copy()
            return

        # Generate and save scaled mask
        self.binary_mask = np.random.binomial(1, self.rate,
                           size=inputs.shape) / self.rate
        # Apply mask to output values
        self.output = inputs * self.binary_mask


    # Backward pass
    def backward(self, dvalues):
        # Gradient on values
        self.dinputs = dvalues * self.binary_mask


class Layer_Input:
    """
    First layer which we feed input data into.
    """

    # Forward pass
    def forward(self, inputs, training):
        self.output = inputs


class Activation_ReLU:
    """
    ReLU Activation class to pipe output of layers through.
    """

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Calculate output values from inputs
        self.output = np.maximum(0, inputs)

    # Backward pass
    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


class Activation_Softmax:
    """
    Softmax Activation class.
    """

    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)

        self.output = probabilities

    # Backward pass
    def backward(self, dvalues):

        # Create uninitialized array
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

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)
    
class Activation_Sigmoid:
    """
    Sigmoid Activation class.
    """


    # Forward pass
    def forward(self, inputs, training):
        # Remember input values
        self.inputs = inputs
        # Sigmoid formula: 1 / (1 + e^(-inputs))
        self.output = 1 / (1 + np.exp(-inputs))

    # Backward pass
    def backward(self, dvalues):
        # Derivative of sigmoid: output * (1 - output)
        # We use self.output from the forward pass
        self.dinputs = dvalues * (self.output * (1 - self.output))

    # Calculate predictions for outputs
    def predictions(self, outputs):
        return outputs


class Optimizer_SGD:
    """
    SGD Optimizer class to update weights and biases in layers.
    """

    # Initialize optimizer - set settings,
    # learning rate of 1. is default for this optimizer
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # Call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                (1. / (1. + self.decay * self.iterations))

    # Update parameters
    def update_params(self, layer):

        # If we use momentum
        if self.momentum:

            # If layer does not contain momentum arrays, create them
            # filled with zeros
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                # If there is no momentum array for weights
                # The array doesn't exist for biases yet either.
                layer.bias_momentums = np.zeros_like(layer.biases)
            # Build weight updates with momentum - take previous
            # updates multiplied by retain factor and update with
            # current gradients
            weight_updates = \
                self.momentum * layer.weight_momentums - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # Build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentums - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Vanilla SGD updates (as before momentum update)
        else:
            weight_updates = -self.current_learning_rate * \
                             layer.dweights
            bias_updates = -self.current_learning_rate * \
                           layer.dbiases

        # Update weights and biases using either
        # vanilla or momentum updates
        layer.weights += weight_updates
        layer.biases += bias_updates

    # Call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1

# Common loss class
class Loss:


    # Set/remember trainable layers
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers


    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):

        # Calculate sample losses
        sample_losses = self.forward(output, y)

        # Calculate mean loss
        data_loss = np.mean(sample_losses)

        # Return the data and regularization losses
        return data_loss


# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):

    # Forward pass
    def forward(self, y_pred, y_true):

        # Number of samples in a batch
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

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

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    # Backward pass
    def backward(self, model_output, y_true):

        # Number of samples
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


# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossentropy():

    # Backward pass
    def backward(self, dvalues, y_true):

        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.dinputs = dvalues.copy()
        # Calculate gradient
        self.dinputs[range(samples), y_true] -= 1
        # Normalize gradient
        self.dinputs = self.dinputs / samples

# Common accuracy class
class Accuracy:
    """
    Calculates the accuracy of model output compared to ground truth values.
    """

    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        return accuracy


class Accuracy_Categorical(Accuracy):
    """
    Calculates the accuracy of model output for categorical values compared to ground truth values.
    """

    def __init__(self, *, binary=False):
        # Binary mode?
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        return predictions == y
    
# Model class
class Model:
    """
    Creates Neural Network Model object. Usage examples at end of file.
    """

    def __init__(self):
        self.layers = []
        self.softmax_classifier_output = None
        self.real_layers = []
    # Add objects to the model
    def add(self, layer):

        if isinstance(layer, Layer_Dense) or isinstance(layer, Layer_Input):
            self.real_layers.append(layer)

        if isinstance(layer, Activation_ReLU) or isinstance(layer, Activation_Sigmoid):
            self.activation_class = layer.__class__.__name__

        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    def set(self, *, loss, optimizer, accuracy):
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

    # Finalize the model
    def finalize(self):

        # Create and set the input layer
        self.input_layer = Layer_Input()

        # Count all the objects
        layer_count = len(self.layers)

        # Initialize a list containing trainable layers:
        self.trainable_layers = []

        total_trainable_parameters = 0
        total_trainable_weights = 0
        total_trainable_biases = 0

        # Iterate the objects
        for i in range(layer_count):

            # If it's the first layer,
            # the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]

            # All layers except for the first and the last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]

            # The last layer - the next object is the loss
            # Also let's save aside the reference to the last object
            # whose output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights",
            # it's a trainable layer -
            # add it to the list of trainable layers
            # We don't need to check for biases -
            # checking for weights is enough
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                total_trainable_biases += self.layers[i].learnable_biases
                total_trainable_weights += self.layers[i].learnable_weights
                total_trainable_parameters += self.layers[i].learnable_weights + self.layers[i].learnable_biases

        vram_usage = self.compute_vram_usage()

        
        self.log_model_info()
        print(f"Number of total trainable parameters: {total_trainable_parameters}\n\tof that weights: {total_trainable_weights}\n\tof that biases: {total_trainable_biases}")
        print(f"Memory usage of model (in bytes): {vram_usage}")

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(
            self.trainable_layers
        )

        # If output activation is Softmax and
        # loss function is Categorical Cross-Entropy
        # create an object of combined activation
        # and loss function containing
        # faster gradient calculation
        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            # Create an object of combined activation
            # and loss functions
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()

    def log_model_info(self):
        print("Model Info:")
        print(f"Layer structure (excluding input layer):")

        for i, layer in enumerate(self.real_layers):
            print(f"Layer {i}:\tNumber of Neurons: {layer.n_neurons}" )

        print(f"Model Activation Function: {self.activation_class}" )
    # Train the model
    def train(self, X, y, *, epochs=1, print_every=1):

        # Initialize accuracy object
        self.accuracy.init(y)

        # Main training loop
        for epoch in range(1, epochs+1):

            # Perform the forward pass
            output = self.forward(X, training=True)


            # Calculate loss
            data_loss = \
                self.loss.calculate(output, y)
            loss = data_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(
                              output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Perform backward pass
            self.backward(output, y)

            # Optimize (update parameters)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            # Print a summary
            if not epoch % print_every:
                print(f'epoch: {epoch}, ' +
                      f'acc: {accuracy:.3f}, ' +
                      f'loss: {loss:.3f} (' +
                      f'data_loss: {data_loss:.3f}, ' +
                      f'lr: {self.optimizer.current_learning_rate}')

    def validate(self, validation_data, output_file=None):
        # For better readability
        X_val, y_val = validation_data

        # Perform the forward pass
        output = self.forward(X_val, training=False)

        # Calculate the loss
        loss = self.loss.calculate(output, y_val)

        # Get predictions and calculate an accuracy
        predictions = self.output_layer_activation.predictions(
                            output)
        accuracy = self.accuracy.calculate(predictions, y_val)
        if output_file is not None:
            np.savetxt(f'{output_file}', predictions, delimiter=',', fmt='%d', header='Predicted_Survived', comments='')

        # Print a summary
        print(f'validation, ' +
                f'acc: {accuracy:.3f}, ' +
                f'loss: {loss:.3f}')
        
        return accuracy

    # Performs forward passes of all layers
    def forward(self, X, training):

        # Call forward method on the input layer
        # this will set the output property that
        # the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # "layer" is now the last object from the list,
        # return its output
        return layer.output

    # Performs backward passes of all layers
    def backward(self, output, y):

        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # First call backward method
            # on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)

            # Since we'll not call backward method of the last layer
            # which is Softmax activation
            # as we used combined activation/loss
            # object, let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return


        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)

        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

def load_titanic_dataset():
    # Load the scaled training features
    X_train_df = pd.read_csv('./titanic/titanic_X_train_scaled.csv')
    X_train = X_train_df.to_numpy().astype('float32')

    # If you have the labels in a separate CSV (e.g., titanic_y_train.csv)
    y_train_df = pd.read_csv('./titanic/titanic_y_train.csv')
    y_train = y_train_df['Survived'].to_numpy().astype('int32').reshape(-1)

    # Similarly for the test set
    X_test_df = pd.read_csv('./titanic/titanic_X_test_scaled.csv')
    X_test = X_test_df.to_numpy().astype('float32')

    y_test_df = pd.read_csv('./titanic/titanic_y_test.csv')
    y_test = y_test_df['Survived'].to_numpy().astype('int32').reshape(-1)

    return X_train, y_train, X_test, y_test

def load_german_credit_data_dataset():
    X_train_df = pd.read_csv('./german_credit_data/german_X_train_scaled.csv')
    X_train = X_train_df.to_numpy().astype('float32')

    y_train_df = pd.read_csv('./german_credit_data/german_y_train.csv')
    y_train = y_train_df['credit_rating'].to_numpy().astype('int32').reshape(-1)

    X_test_df = pd.read_csv('./german_credit_data/german_X_test_scaled.csv')
    X_test = X_test_df.to_numpy().astype('float32')

    y_test_df = pd.read_csv('./german_credit_data/german_y_test.csv')
    y_test = y_test_df['credit_rating'].to_numpy().astype('int32').reshape(-1)

    return X_train, y_train, X_test, y_test

class Grid_Search:
    """
    Performs a grid search over specified hyperparameters for a neural network model.
    Assumes that Model, Layer_Dense, Activation_ReLU, Activation_Sigmoid,
    Activation_Softmax, Layer_Dropout, Loss_CategoricalCrossentropy,
    Optimizer_SGD, and Accuracy_Categorical classes are defined elsewhere.
    """

    def run_grid_search(self,
                        X_train, y_train, X_test, y_test,
                        neurons_per_layer_options,
                        epochs_options,
                        hidden_activation_classes, # Dictionary: {'Name': ActivationClass}
                        learning_rate_options,
                        print_every_train=100,
                        output_file_prefix="predictions"):

        input_shape = X_train.shape[1]
        output_classes = len(np.unique(y_train))

        count = 0
        highest_accuracy = 0
        highest_output = ""
        
        # Iterating through all combinations of hyperparameters
        original_stdout = sys.stdout
        for layers_neurons in neurons_per_layer_options:
            for epochs_train in epochs_options: # Use epochs_train to avoid conflict with outer 'epochs' list
                for activation_name, Activation_Class in hidden_activation_classes.items():
                    for learning_rate in learning_rate_options:

                        # Initialize a new model for each combination
                        model = Model()
                        current_input_size = input_shape

                        # Add hidden layers based on the current combination
                        for i, layer_size in enumerate(layers_neurons):
                            model.add(Layer_Dense(current_input_size, layer_size))
                            model.add(Activation_Class()) # Use the selected hidden activation
                            current_input_size = layer_size

                        # Add the output layer
                        model.add(Layer_Dense(current_input_size, output_classes))
                        model.add(Activation_Softmax())

                        # Set loss, optimizer, and accuracy objects for the model
                        model.set(
                            loss=Loss_CategoricalCrossentropy(),
                            optimizer=Optimizer_SGD(learning_rate=learning_rate),
                            accuracy=Accuracy_Categorical()
                        )

                        output_file=f"{output_file_prefix}_{count}_L{'-'.join(map(str, layers_neurons))}_E{epochs_train}_A{activation_name}_LR{str(learning_rate).replace('.', '')}"
                        with open(f"{output_file}.txt", 'w') as f:
                            sys.stdout = f

                            # Finalize the model (e.g., link layers for backprop)
                            model.finalize()

                            # Train the model with the current epochs setting
                            model.train(X_train, y_train, epochs=epochs_train, print_every=print_every_train)

                            # Validate model on test set
                            # The output_file name now includes combination details for uniqueness
                            accuracy = model.validate((X_test, y_test),
                                            output_file=f"{output_file}.csv")
                            
                            if (accuracy > highest_accuracy):
                                highest_accuracy = accuracy
                                highest_output = output_file
                            
                            count += 1
                        sys.stdout = original_stdout

        with open(f"{highest_output}.txt", "r") as f:
            print("Model with highest accuracy: ")
            print(f.read())

# Titanic dataset Model

X_train, y_train, X_test, y_test = load_titanic_dataset()

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
                            [1], output_file_prefix="titanic_predictions")

input_shape = X_train.shape
input_shape = input_shape[1]
output_classes = len(np.unique(y_train))

neurons_per_layer = [
    [32],
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [64, 64, 64]
]

epochs = [100, 500, 1000]

count = 0
for layers_neurons in neurons_per_layer:
    model = Model()
    layers = len(layers_neurons)
    current_input_size = input_shape
    for i, layer_size in enumerate(layers_neurons):
        model.add(Layer_Dense(current_input_size, layer_size))
        model.add(Activation_ReLU())
        model.add(Layer_Dropout(0.1))
        current_input_size = layer_size


    model.add(Layer_Dense(current_input_size, output_classes))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_SGD(),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    print("-----------------")
    model.finalize()

    # Train the model
    model.train(X_train, y_train, epochs=1000, print_every=100)

    # Validate model on test set
    model.validate((X_test, y_test), output_file=f"predictions{count}_titanic.csv")
    count += 1
    print("-----------------\n")

# German Credit Data dataset Model

X_train, y_train, X_test, y_test = load_german_credit_data_dataset()

input_shape = X_train.shape
input_shape = input_shape[1]
output_classes = len(np.unique(y_train))

layers_and_neurons_per_layer = [
    [32],
    [64, 32],
    [128, 64, 32],
    [256, 128, 64],
    [64, 64, 64]

]
count = 0
for layers_neurons in layers_and_neurons_per_layer:
    model = Model()
    layers = len(layers_neurons)
    current_input_size = input_shape
    for i, layer_size in enumerate(layers_neurons):
        model.add(Layer_Dense(current_input_size, layer_size))
        model.add(Activation_ReLU())
        current_input_size = layer_size


    model.add(Layer_Dense(current_input_size, output_classes))
    model.add(Activation_Softmax())

    # Set loss, optimizer and accuracy objects
    model.set(
        loss=Loss_CategoricalCrossentropy(),
        optimizer=Optimizer_SGD(),
        accuracy=Accuracy_Categorical()
    )

    # Finalize the model
    print("-----------------")
    model.finalize()

    # Train the model
    model.train(X_train, y_train, epochs=1000, print_every=100)

    # Validate model on test set
    model.validate((X_test, y_test), output_file=f"predictions{count}_german.csv")
    count += 1
    print("-----------------\n")