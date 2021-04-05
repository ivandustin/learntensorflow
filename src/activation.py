import os

# Turn off debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper function
def show(fn):
    fn()
    print()

# Import TensorFlow
import tensorflow as tf

# Import custom activation function
from functions import activation_function

c = tf.constant([-100.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 100.0])
x = tf.Variable(c)

@show
def block():
    print('Input')
    print(c)

@show
def block():
    print('ReLU')    
    with tf.GradientTape() as tape:
        output = tf.keras.activations.relu(x)

    print(output)

    gradient = tape.gradient(output, x)

    print(gradient)

@show
def block():
    print('ReLU with Max Value')    
    with tf.GradientTape() as tape:
        output = tf.keras.activations.relu(x, max_value=1.0)

    print(output)

    gradient = tape.gradient(output, x)

    print(gradient)

@show
def block():
    print('ReLU with Threshold')
    with tf.GradientTape() as tape:
        output = tf.keras.activations.relu(x, threshold=1.0)

    print(output)

    gradient = tape.gradient(output, x)

    print(gradient)

@show
def block():
    print('ReLU with Max Value and Threshold')
    with tf.GradientTape() as tape:
        output = tf.keras.activations.relu(x, max_value=1.0, threshold=1.0)

    print(output)

    gradient = tape.gradient(output, x)

    print(gradient)

@show
def block():
    print('Custom')
    with tf.GradientTape() as tape:
        output = activation_function(x)

    gradient = tape.gradient(output, x)

    print(output)
    print(gradient)

    assert output[0] == 0
    assert output[1] == 0
    assert output[2] == 0
    assert output[3] == 0
    assert output[4] == 0
    assert output[5] == 0
    assert output[6] == 1
    assert output[7] == 1
    assert output[8] == 1

    assert gradient[0] == 1
    assert gradient[1] == 1
    assert gradient[2] == 1
    assert gradient[3] == 1
    assert gradient[4] == 1
    assert gradient[5] == 1
    assert gradient[6] == 1
    assert gradient[7] == 1
    assert gradient[8] == 1
