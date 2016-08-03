import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  features = W.shape[0]
  for i in xrange(num_train):
    scores = X[i, :].dot(W)
    #numerical stability
    scores -= np.max(scores)
    probabilities = np.exp(scores) / np.sum(np.exp(scores))
    loss += -np.log(probabilities[y[i]]) 
    for class_num in xrange(num_classes):
      #http://ufldl.stanford.edu/wiki/index.php/Softmax_Regression
      if class_num == y[i]:
        dW[:, class_num] -=  X[i, :] * (1 - probabilities[class_num])
      else:
        dW[:, class_num] -= X[i, :] * (-probabilities[class_num])
  loss /= num_train
  dW   /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  features = W.shape[0]
  scores = X.dot(W)
  scores -= np.max(scores, axis = 1).reshape(-1, 1)
  probabilities = np.exp(scores) / np.sum(np.exp(scores), axis = 1).reshape(-1, 1)
  idx_train = np.array(range(num_train))
  correct_class_log_probabilties = -np.log(probabilities[idx_train, y].reshape(num_train, 1))
  loss = np.sum(correct_class_log_probabilties)

  probabilities[[idx_train, y]] -= 1
  dW = X.T.dot(probabilities)

  loss /= num_train
  dW   /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW   += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

