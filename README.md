# Contents
This repository contains the implementation of the following layers:
- Convolution
- Recurrent Convolution
- ConvLSTM
- ConvGRU (Fully gated and Type 2)
- DenseConv
This layers are implemented for image classification tasks.
Find below a more detailed description of each layer.

## Convolution
The implementation details can be found in *baseconv.py*. The class *Conv2D* is ready to use in a convolutoinal network.
Objects of this class can be further configured through *config.py*-file. It can be adjusted, if batch normalization or dropout should be applied as well as the corresponding parameters and which activation function (following strings are possible: *"RELU"*, *"LEAKY_RELU"*, *"SIGMOID"*, *"TANH"*) should be utilized. Depending on the activation function, the weights are initialized with *kaiming_uniform* or *xavier_uniform*

The class can be used as follows:

```python
Conv2D(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    in_size=in_size
)
```

## Recurrent Convolution

## ConvLSTM

## ConvGRU

### Fully gated

### Type 2

## DenseConv
