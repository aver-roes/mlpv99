
# 🤖 MLPV99

A simple (Keras like) library for building Multi layer perceptron (for the purpose of learning).

The libray is built from scratch using Nerual network math with only vanilla Python &  a little bit of Numpy.

Anyone can use to learning more about how neural network work on a lower level (forward pass and backward pass using backpropagation, with SGD optimizers, activation functions...etc, everything from scratch just math and python)


Currently it can be used to build only dense neural networks, but you can add other architectures(CNN, RNN..etc) if you wish after all it's an open source project! anyone is welcome to contribute to the project.


## 🧐 Features
### Dense layer
- Dense layer with random initialization for weights.
- Forward pass method & backward pass method.
- get & set parameters(for transfer learning).

### Dropout layer
- Dropout layer that drops neurons from the neural network during training which helps prevent overfitting.
- Forward pass method & backward pass method.

### Activation Functions
- RelU: for hidden layers.
- Sigmoid: for binary classification.
- Softmax: for multi-class classification.
- Linear: for regression.
- All activation functions contain forward, backward & predictions methods.

### Loss Functions & Regularizations
- L1 & L2 Regularizations.
- Binary cross entropy loss.
- Categorical cross entropy loss.
- Mean squared error loss.
- Mean absolute error loss.
- All Loss functions contain forward, backward methods.

### Accuracy Functions
- Accuracy Categorical: for binary & multi-classification.
- Accuracy Regression: for regression.

### Optimizers
- Stochastic gradient descent (SGD).
- SGD with learning decay.
- SGD with momentum.
- Adagrad.
- RMSprop.
- Adam.

### Model layer
#### The Model Object: is used to do things like save and load this object to use for future prediction tasks. also the object can cut down on some of the more common lines of code, making it easier to work with the code base and when building new models.
- add method is used to stack layers together.
- set method is used to set the type of: (loss, optimizer, accuracy) being used.
- finalize method is used to wire the layers together.
- trian method is used to trian the model with the specified: (batch size, number of epochs, and validation data).
- evaluate method for evaluating the model with test data.
- forward method for forward propagation & backward method for backpropagation.
- getand set parameters method(from Dense layer)
- load parameters method
- save method to save the entire model 
- load method to load a model
- predict method for making predictions and inferences

