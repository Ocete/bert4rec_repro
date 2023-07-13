import tensorflow as tf

def stack_indexes_with_predictions(indexes, predictions):

    tf.print('indexes')
    tf.print(tf.shape(indexes))
    tf.print(indexes)
    tf.print('predictions')
    tf.print(tf.shape(predictions))
    tf.print(predictions)
    return predictions
    # return tf.stack([tf.cast(indexes, tf.float32), predictions], axis=-1)