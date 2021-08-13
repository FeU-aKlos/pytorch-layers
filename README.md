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

A Sample network containing a Conv2D-layer can be obtained from class *SampleConvNet*.

## Recurrent Convolution
The implemenation is based on [this](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf) research paper. Different from the original paper, local response normalization has been replaced by batch normalization. The implementation details can be found in *recurrentconv.py* in the class *RecurrentConv*. As above mentioned, the parameters about batch normalization as well as dropout can be configured by the *config.py*-file.

The class can be used as follows:

```python
RecurrentConv(
    in_channels=in_channels,
    out_channels=out_channels,
    kernel_size=kernel_size,
    stride=stride,
    in_size=in_size
)
```

A Sample network containing a RecurrentConv-layer can be obtained from class *SampleRecurrentConvNet*.

## ConvLSTM

## ConvGRU

### Fully gated

### Type 2

## DenseConv
