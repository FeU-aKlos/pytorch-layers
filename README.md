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

A Sample network containing a Conv2D-layer can be obtained from class *SampleConvNet*. By running the following command, certain unittest (including the training and testing of the sample network class *SampleConvNet*) can be executed

```bash
python -m unittest test_baseconv.py
```

## Recurrent Convolution
The implemenation is based on [this research paper.](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Liang_Recurrent_Convolutional_Neural_2015_CVPR_paper.pdf) Different from the original paper, local response normalization has been replaced by batch normalization. The implementation details can be found in *recurrentconv.py* in the class *RecurrentConv*. As above mentioned, the parameters about batch normalization as well as dropout can be configured by the *config.py*-file.

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

A Sample network containing a RecurrentConv-layer can be obtained from class *SampleRecurrentConvNet*. By running the following command, certain unittest (including the training and testing of the sample network class *SampleRecurrentConvNet*) can be executed

```bash
python -m unittest test_recurrentconv.py
```

## ConvLSTM
The implementation is based on [this research paper.](https://papers.nips.cc/paper/2015/file/07563a3fe3bbe7e3ba84431ad9d055af-Paper.pdf) and inspired by [this github repository](https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py), but also considers the cell state. Nevertheless, the implementation is thought for image classification tasks and is similar to the implementation of the Recurrent Convolution. Instead of feeding new samples of a time sequence into the ConvLSTM, we feed the initial sample and concatenate it with ![formula](https://render.githubusercontent.com/render/math?math=h_{t-1}). The forward input is assumed to has the following formating: B,C,W,H. In the current implementation, only kernels with uneven size are possible. Each convolution is followed by a dedicated batch normalization layer. For that reason, the bias is omitted. The weights are initialized with *xavier_uniform*. The implementation details can be obtained by the file *convlstm.py*, and class *ConvLSTM*. 

The class can be used as follows:

```python
ConvLSTM(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    kernel_size=kernel_size,
    stride=stride,
    in_size=in_size
)
```

A Sample network containing a ConvLSTM-layer can be obtained from class *SampleConvLSTMNet*. By running the following command, certain unittest (including the training and testing of the sample network class *SampleConvLSTMNet*) can be executed

```bash
python -m unittest test_convlstm.py
```

## ConvGRU
The implementaion is based on [this research paper.](https://arxiv.org/pdf/1511.06432v4.pdf). The implementation is thought for image classification tasks and is similar to the implementation of the Recurrent Convolution and ConvLSTM. Instead of feeding new samples of a time sequence into the ConvGRU, we feed the initial sample. The forward input is assumed to has the following formating: B,C,W,H. In the current implementation, only kernels with uneven size are possible. Each convolution is followed by a dedicated batch normalization layer. For that reason, the bias is omitted. The weights are initialized with *xavier_uniform*. The implementation details can be obtained by the file *convgru.py*, and class *ConvGRU*. 

The class can be used as follows:

```python
ConvGRU(
    in_channels=in_channels,
    hidden_channels=hidden_channels,
    kernel_size=kernel_size,
    stride=stride,
    in_size=in_size
)
```

A Sample network containing a ConvGRU-layer can be obtained from class *SampleConvGRUNet*. By running the following command, certain unittest (including the training and testing of the sample network class *SampleConvGRUNet*) can be executed

```bash
python -m unittest test_convgru.py
```

### Fully gated

### Type 2

## DenseConv
