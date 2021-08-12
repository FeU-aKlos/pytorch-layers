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


#ANN training and test config
batch_size=64
test_batch_size=1000
epochs=3
lr=1.0
gamma=0.7
seed=1
log_interval=10
save_model=False