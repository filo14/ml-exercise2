# ml-exercise2

## Best results

### Titanic

#### From Scratch

```
Model with highest accuracy: 
Model Info:
Layer structure (excluding input layer):
Layer 0:        Number of Neurons: 32
Layer 1:        Number of Neurons: 2
Model Activation Function: Activation_ReLU
Model Output Activation Function: Activation_Softmax
Number of total trainable parameters: 418
        of that weights: 384
        of that biases: 34
Memory usage of model (in bytes): 3344
Epoch: 100, Accuracy: 0.843, Loss: 0.388, Learning rate: 1
Epoch: 200, Accuracy: 0.837, Loss: 0.374, Learning rate: 1
Epoch: 300, Accuracy: 0.836, Loss: 0.365, Learning rate: 1
Epoch: 400, Accuracy: 0.846, Loss: 0.355, Learning rate: 1
Epoch: 500, Accuracy: 0.853, Loss: 0.349, Learning rate: 1
Test Set Results: Accuracy: 0.849, Precision: 0.862, Recall: 0.725, Loss: 0.432
Processing time: 4477.670 ms

Params:
Layer count: [32]
Epochs: 500
Activation Function: ReLU (class 'Activation_ReLU')
Learning rate: 1
```

#### From PyTorch

#### From LLM

```
Best configuration: {'layer_sizes': [256, 128, 64], 'activation': 'relu', 'lr': 1, 'momentum': 0.0, 'epochs': 1000}

Validation accuracy: 0.7902097902097902
Test accuracy      : 0.8212290502793296
Total parameters   : 44098
VRAM (float32)     : 176392.00 bytes
Runtime:             3142.718
Validation precision / recall: 0.8378 / 0.5636
Test precision / recall      : 0.8627 / 0.6377
```

### German Credit

#### From Scratch

```
Model with highest accuracy: 
Model Info:
Layer structure (excluding input layer):
Layer 0:        Number of Neurons: 32
Layer 1:        Number of Neurons: 2
Model Activation Function: Activation_ReLU
Model Output Activation Function: Activation_Softmax
Number of total trainable parameters: 642
        of that weights: 608
        of that biases: 34
Memory usage of model (in bytes): 5136
Epoch: 100, Accuracy: 0.796, Loss: 0.434, Learning rate: 1
Test Set Results: Accuracy: 0.790, Precision: 0.705, Recall: 0.517, Loss: 0.493
Processing time: 1063.709 ms

Params:
Layer count: [32]
Epochs: 100
Activation Function: ReLU (class 'Activation_ReLU')
Learning rate: 1
```

#### From PyTorch

#### From LLM

```
New best val acc 0.6875 with cfg {'layer_sizes': [32], 'activation': 'relu', 'lr': 1, 'momentum': 0.0, 'epochs': 100}
New best val acc 0.7625 with cfg {'layer_sizes': [32], 'activation': 'relu', 'lr': 1, 'momentum': 0.0, 'epochs': 1000}
Grid‑search done in 30,022 ms
Best configuration: {'layer_sizes': [32], 'activation': 'relu', 'lr': 1, 'momentum': 0.0, 'epochs': 1000}

Validation accuracy: 0.7625
Test accuracy      : 0.765
Total parameters   : 642
VRAM (float32)     : 2568.00 bytes
Runtime:             375.125
Validation precision / recall: 0.6667 / 0.4167
Test precision / recall      : 0.6857 / 0.4000
```