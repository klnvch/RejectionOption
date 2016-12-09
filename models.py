'''
Created on Dec 9, 2016

@author: anton
'''

import tensorflow as tf

def model_fn(features, targets, mode, params):
    """Model function for Estimator."""

    hidden_layer = tf.contrib.layers.sigmoid(features, 10)


    # Connect the output layer to second hidden layer (no activation fn)
    output_layer = tf.contrib.layers.softmax(hidden_layer, 3)

    # Reshape output layer to 1-dim Tensor to return predictions
    predictions = tf.reshape(output_layer, [-1])
    predictions_dict = {"ages": predictions}

    # Calculate loss using mean squared error
    loss = tf.contrib.losses.mean_squared_error(predictions, targets)

    train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

    return predictions_dict, loss, train_op
