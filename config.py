# Default activation function, possible values are listed in dictionary "act_fns" above
act_function_name = "RELU"


# Set use of batch normalization and momentum
employ_batch_normalization_dense = False
employ_batch_normalization_conv = True
batch_normalization_momentum = 0.99

# Set use of dropout and dropout rate. If both batch normalization and dropout are active, convolutional layers will
# only receive batch normalization while dropout will only be applied to dense layers
employ_dropout_dense = True
employ_dropout_conv = False

dropout_rate = 0.5
