import os

# Turn off debugging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Helper function
def show(fn):
    fn()
    print()

# Import TensorFlow
import tensorflow as tf

# Import functions
from functions import activation_function, loss_function

@show
def block():
    print('Basic')
    a = tf.constant([1.0])
    b = tf.Variable(tf.constant([2.0]))
    c = tf.Variable(tf.constant([3.0]))
    E = tf.constant([[5.0]])

    with tf.GradientTape() as tape:
        d = a * b * c
        tape.watch(d)
        e = activation_function(d)
        tape.watch(e)
        f = loss_function(e, E)

    gradients = tape.gradient(f, [b, c, d, e])

    db = gradients[0]
    dc = gradients[1]
    dd = gradients[2]
    de = gradients[3]

    assert f == tf.square(e - E)
    assert de == 2 * (e - E)
    assert dd == 1 * de
    assert dc == a * b * dd
    assert db == a * c * dd

    assert db == -24.0
    assert dc == -16.0

    print(db)
    print(dc)

@show
def block():
    print('Advance')
    A = tf.constant([1.0])
    a = tf.constant([2.0])
    b = tf.constant([3.0])
    c = tf.constant([4.0])
    d = tf.constant([5.0])
    e = tf.constant([6.0])
    f = tf.constant([7.0])
    E = tf.constant([2.0])

    with tf.GradientTape() as tape:
        tape.watch(A)
        B0   = A * a * b
        B    = activation_function(B0)
        C0   = (B * c * d) + (A * e * f)
        C    = activation_function(C0)
        loss = loss_function(C, E)

    gradient = tape.gradient(loss, A)

    assert gradient == -324.0
    
    print(gradient)
