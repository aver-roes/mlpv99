import numpy as np


# Dropout
class Layer_Dropout:
  
  def __init__(self, rate):

    # Store rate, we invert it as for example for dropout
    # of 0.1 we need success rate of 0.9
    self.rate = 1 - rate


  # Forward Pass
  def forward(self, inputs, training):

    # Save the inputs
    self.inputs = inputs

    #  If not in the training mode - return values
    if not training:
      self.output = inputs.copy()
      return

    # Generate and save scaled mask
    self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

    # Apply mask to output values
    self.output = inputs * self.binary_mask

  
  # Backword Pass
  def backward(self, dvalues):

    # Gradient on values
    self.dinputs = dvalues * self.binary_mask
