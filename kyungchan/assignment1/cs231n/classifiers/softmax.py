from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


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
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    D = W.shape[0]
    C = W.shape[1]
    eps = 10 ** -10

    for i in range(N):
      a = np.dot(X[i, :], W)
      da = np.zeros_like(a)
      for k in range(C):
        if y[i] == k:
          loss += -np.log(np.exp(a[k]) / np.exp(a).sum() + eps)
          da[k] += np.exp(a[k]) / np.exp(a).sum() - 1
        else:
          da[k] += np.exp(a[k]) / np.exp(a).sum()
      dW += np.dot(X[i, :].reshape(-1, 1), da.reshape(1, -1))
    
    loss /= N
    dW /= N

    loss += reg * (W * W).sum()
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    eps = 10 ** -10

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    C = W.shape[1]

    a = np.dot(X, W)
    y_hot = np.eye(C)[y]
    loss = (-np.log(np.exp(a) / np.exp(a).sum(axis = 1, keepdims = True) + eps) * y_hot).sum()
    da = np.exp(a) / np.exp(a).sum(axis = 1, keepdims = True) - y_hot
    dW = np.dot(X.T, da)

    loss /= N
    dW /= N

    loss += reg * (W * W).sum()
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
