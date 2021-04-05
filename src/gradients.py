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
    print('Introduction')
    a = tf.random.normal(shape=(2, 2))
    b = tf.random.normal(shape=(2, 2))

    with tf.GradientTape() as tape:
        tape.watch(a)  # Start recording the history of operations applied to `a`
        c = tf.sqrt(tf.square(a) + tf.square(b))  # Do some math using `a`
        # What's the gradient of `c` with respect to `a`?
        dc_da = tape.gradient(c, a)
        print(dc_da)

@show
def block():
    print('Automatic watch for variable')
    a = tf.random.normal(shape=(2, 2))
    b = tf.random.normal(shape=(2, 2))
    a = tf.Variable(a)

    with tf.GradientTape() as tape:
        c = tf.sqrt(tf.square(a) + tf.square(b))
        dc_da = tape.gradient(c, a)
        print(dc_da)

@show
def block():
    print('Basic')
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    b = tf.Variable(b)

    with tf.GradientTape() as tape:
        c = a + b

    print(c)

    gradient = tape.gradient(c, b)
    print(gradient)

@show
def block():
    print('Multiple')
    a = tf.constant(2.0)
    b = tf.Variable(tf.constant(3.0))
    c = tf.Variable(tf.constant(5.0))

    with tf.GradientTape() as tape:
        d = a + b + c

    print(d)

    gradient = tape.gradient(d, [b, c])
    print(gradient)
