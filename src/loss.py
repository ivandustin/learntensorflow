import os

# Turn off debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper function
def show(fn):
    fn()
    print()

# Import TensorFlow
import tensorflow as tf

# Import loss function
from functions import loss_function

@show
def block():
    print('Basic')
    actual   = tf.constant([0.0])
    expected = tf.constant([1.0])
    assert loss_function(actual, expected) == tf.square(actual - expected)

@show
def block():
    print('Gradient')
    actual   = tf.constant([0.0])
    expected = tf.constant([1.0])
    
    with tf.GradientTape() as tape:
        tape.watch(actual)
        loss = loss_function(actual, expected)

    gradient = tape.gradient(loss, actual)
    
    assert gradient == 2 * (actual - expected)
