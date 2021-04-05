import tensorflow as tf

@tf.custom_gradient
def activation_function(x):
    def gradient(dy):
        return dy
    return tf.where(x >= 1.0, 1.0, 0.0), gradient

loss_function = tf.keras.losses.MeanSquaredError()
