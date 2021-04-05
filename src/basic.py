import os

# Turn off debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper function
def show(fn):
    fn()
    print()

# Import TensorFlow
import tensorflow as tf

@show
def block():
    print('Basic tensor')
    tensor = tf.constant([1,2,3])
    print(tensor)

@show
def block():
    print('Access a value inside a tensor')
    tensor = tf.constant([1,2,3])
    print(tensor[0])
    print(tensor[1])
    print(tensor[2])

@show
def block():
    print('Generate ones and zeros')
    print(tf.ones(shape=(3, 2)))
    print(tf.zeros(shape=(3, 2)))

@show
def block():
    print('Generate random numbers')
    normal  = tf.random.normal(shape=(2, 2), mean=0.0, stddev=1.0)
    uniform = tf.random.uniform(shape=(2, 2), minval=0, maxval=10, dtype="int32")
    print(normal)
    print(uniform)

@show
def block():
    print('Mutable tensor')
    constant = tf.constant([1,2,3])
    variable = tf.Variable(constant)
    print(constant)
    print(variable)

@show
def block():
    print('Access a value inside a variable')
    constant = tf.constant([1,2,3])
    variable = tf.Variable(constant)
    print(variable[0])
    print(variable[1])
    print(variable[2])

@show
def block():
    print('Change the values inside a variable')
    constant = tf.constant([1,2,3])
    variable = tf.Variable(constant)
    variable.assign(tf.constant([3,2,1]))
    print(variable)

@show
def block():
    print('Do addition with the variable in-place')
    constant = tf.constant([1,2,3])
    variable = tf.Variable(constant)
    variable.assign_add(tf.constant([1,1,1]))
    print(variable)

@show
def block():
    print('Do subtraction with the variable in-place')
    constant = tf.constant([1,2,3])
    variable = tf.Variable(constant)
    variable.assign_sub(tf.constant([1,1,1]))
    print(variable)

@show
def block():
    print('Doing math with the tensor')
    a = tf.constant([1,1,1])
    b = tf.constant([1,1,1])
    c = a + b
    print(c)
