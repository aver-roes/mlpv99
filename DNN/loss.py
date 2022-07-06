import numpy as np


# Common loss class
class Loss:

  # Regularization loss calculation
  def regularization_loss(self):

    # 0 by default
    regularization_loss = 0

    # Calculate regularization loss
    # iterate all trainable layers
    for layer in self.trainable_layers:

      # L1 regularization - weights
      # calculate only when factor greater than 0
      if layer.weight_regularizer_L1 > 0:
        regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))

      # L2 regularization - weights
      if layer.weight_regularizer_L2 > 0:
        regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)

      # L1 regularization - biases
      # calculate only when factor greater than 0
      if layer.bias_regularizer_L1 > 0:
        regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))

      # L2 regularization - biases
      if layer.bias_regularizer_L2 > 0:
        regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)

    return regularization_loss


  # Set/remember trainable layers
  def remember_trainable_layers(self, trainable_layers):
    self.trainable_layers = trainable_layers


  # Calculates the data and regularization losses
  # given model output and ground truth values
  def calculate(self, output, y, *, include_regularization=False):

    # Calculate sample losses
    sample_losses = self.forward(output, y)

    # Calculate mean loss
    data_loss = np.mean(sample_losses)

    # Add accumulated sum of losses and sample count
    self.accumulated_sum += np.sum(sample_losses)
    self.accumulated_count += len(sample_losses)

    # If just data loss - return it
    if not include_regularization:
      return data_loss

    # Return the data and regularization losses
    return data_loss, self.regularization_loss()
  

  # Calculates accumulated loss
  def calculate_accumulated(self, *, include_regularization=False):

    # Calculated mean loss
    data_loss = self.accumulated_sum / self.accumulated_count

    # If just data loss - return it
    if not include_regularization:
      return data_loss

    # Return the data and regularization losses
    return data_loss, self.regularization_loss()


  # Reset variables for accumulated loss
  def new_pass(self):
    self.accumulated_sum = 0
    self.accumulated_count = 0



# Categorical Cross Entropy Loss
class Loss_CategoricalCrossEntropy(Loss):
  
  # Forward pass
  def forward(self, y_pred, y_true):
    
    # Nummber of samples in the batch
    samples = len(y_pred)

    # Clip data to prevnet division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Probabilities for target values -
    # Only if categorical labels
    if len(y_true.shape) == 1:
      correct_confidences = y_pred_clipped[range(samples), y_true]

    # Mask values - only for one-hot encoded labels
    elif len(y_true) == 2:
      correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

    negative_log_likelihoods = -np.log(correct_confidences)  
    return negative_log_likelihoods

  # Backward pass
  def backward(self, dvalues, y_true):

    # Number of samples
    samples = len(dvalues)
    
    # Number of labels in every sample
    # We'll use the first sample to count them
    labels = len(dvalues[0])

    # one-hot encoded vectors
    if len(y_true.shape) == 1:
      y_true = np.eye(labels)[y_true]

    # Calculate graident
    self.dinputs = -y_true / dvalues
    # Normalize the gradient
    self.dinputs = self.dinputs / samples



# Softmax classifier - combined Softmax activation
# and cross-entropy loss for faster backward step
class Activation_Softmax_Loss_CategoricalCrossEntropy():

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



# Binary cross-entropy loss
class Loss_BinaryCrossEntropy(Loss):

  # Forward pass
  def forward(self, y_pred, y_true):

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    # Calculate sample-wise loss
    sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))

    # mean of all of losses from a single sample
    sample_losses = np.mean(sample_losses, axis=-1)

    return sample_losses


  # Backward pass
  def backward(self, dvalues, y_true):
    
    # Number of samples
    samples = len(dvalues)

    # Number of outputs in every sample
    # We'll use the first sample to count them
    outputs = len(dvalues[0])

    # Clip data to prevent division by 0
    # Clip both sides to not drag mean towards any value
    clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)

    # Calculate gradient
    self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs

    # Normalize gradient
    self.dinputs = self.dinputs / samples



# Mean Squared Error loss
class Loss_MeanSquaredError(Loss):

  # Forward pass
  def forward(self, y_pred, y_true):

    # Calculate loss
    sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

    return sample_losses


  # backward pass
  def backward(self, dvalues, y_true):

    # Number of samples
    samples = len(dvalues)

    # Number of outputs in every sample
    # We'll use the first sample to count them
    outputs = len(dvalues[0])

    # Gradient on values
    self.dinputs = -2 * (y_true - dvalues) / outputs

    # Normalize the graidient
    self.dinputs = self.dinputs / samples



# Mean Absolute Error Loss
class Loss_MeanAbsoluteError(Loss):

  # Forward pass
  def forward(self, y_pred, y_true):

    # Calculate loss
    sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

    return sample_losses

  
  # Backward pass
  def backward(self, dvalues, y_true):

    # Number of samples
    samples = len(dvalues)

    # Number of outputs in every sample
    # We'll use the first sample to count them
    outputs = len(dvalues[0])

    # Calculate gradient
    self.dinputs = np.sign(y_true - dvalues) / outputs

    # Normalize gradient
    self.dinputs = self.dinputs / samples