import tensorflow as tf

def stack_indexes_with_predictions(indexes, predictions):
    return tf.stack([tf.cast(indexes, tf.float32), predictions], axis=-1)