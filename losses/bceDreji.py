
from tensorflow.keras.losses import BinaryCrossentropy
import tensorflow as tf
import tensorflow.keras.backend as K

from aprec.losses.loss import Loss
import aprec.recommenders.dnn_sequential_recommender.models.sasrecDreji.sasrecDreji as sasrecDreji


class BCELossDreji(Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__name__ = "BCEDreji"
        self.less_is_better = True
        self.eps = tf.constant(1e-16, 'float32')

        if 'model_arc' not in kwargs:
            raise ValueError('BCELossDreji must be initialized with a reference to the model_arc.')
        self.model_arc = kwargs['model_arc']
        self.model = None

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 1
        self.alpha = kwargs['alpha']

    def __call__(self, y_true_raw_indexed, y_pred_indexed):

        return 1
        '''
            Example call for this loss:
                x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
                y_indexed = stack_indexes_with_predictions(indexes=x[1], predictions=y)
                y_pred_indexed = stack_indexes_with_predictions(indexes=x[1], predictions=y_pred)
                loss = self.compiled_loss(y_indexed, y_pred_indexed, sample_weight,
                        regularization_losses=self.losses)
        '''
        # This method asumes that position i-th of vector y_pred corresponds to the i-th item
        # as seen by the model (it has associated item id `i` as seen by the model).
        item_indexes, y_true_raw = self.slice_input(y_true_raw_indexed)
        _, y_pred = self.slice_input(y_pred_indexed)

        similarities = self.compute_similarities(item_indexes, y_true_raw)
        
        # y_true contains 1 if the position is the objective item and 0 if it isn't. It contains -1
        # if the data is invalid. is_target is a masked true where y_true is either 0 or 1.
        y_true = tf.cast(y_true_raw, 'float32')
        is_target = tf.cast((y_true >= -self.eps), 'float32')

        # pos = -tf.math.log(tf.sigmoid(y_pred) + self.eps) * is_target
        pos = -tf.math.log(tf.sigmoid(y_pred * similarities) + self.eps) * is_target
        num_targets = tf.reduce_sum(is_target)
        ce_sum = tf.reduce_sum(pos)
        res_sum = tf.math.divide_no_nan(ce_sum, num_targets)

        sim_sum = tf.math.reduce_sum(similarities * is_target)
        sim_mean = tf.math.divide_no_nan(sim_sum, num_targets)
        return res_sum + sim_mean
    

    def slice_input(self, tensor):
        """
            tensor: A tensor with shape [..., 2], where tensor[..., 0] being the x's and
            tensor[..., 1], the y's.

            Returns:
                A tuple of shape [...] (same shape as input without the last dimension), where
                the first element is tensor[..., 0] and the second, tensor[..., 1] (x and y respectively).
        """
        shape = tf.shape(tensor)[:-1]
        slicing_offset_x = tf.zeros(tf.size(shape) + 1, dtype=tf.int32)
        slicing_offset_y = tf.concat([tf.zeros(tf.size(shape), dtype=tf.int32), tf.constant([1])], axis=0)
        slicing_size = tf.concat([tf.repeat(-1, tf.size(shape)), tf.constant([1])], axis=0)
        
        x = tf.reshape(tf.slice(tensor, begin=slicing_offset_x, size=slicing_size), shape) 
        y = tf.reshape(tf.slice(tensor, begin=slicing_offset_y, size=slicing_size), shape)
        return tf.cast(x, tf.int32), y
    


    def compute_similarities(self, item_indexes, y_true_raw):
        """
            item_indexes: A tensor of shape
                (batch_size, guesses per batch, elements per guess) = (B, GB, I).
            y_true: A tensor of shape (B, GB, I). Contains 1 for the target item, 0 for non-target
                item and -1 for invalid data.

            Returns:
                A tensor of shape (B, GB, I) contanining in each position the similarity between
                the corresponding the target item of item_indexes (where y_true is 1) and the every
                item of item_indexes in that same bath/guess.
        """
        if self.model == None:
            self.model = self.model_arc.latest_model
        y_true = tf.cast(y_true_raw, dtype=tf.int32)

        # Substitute the invalid data rows in y_true (full of -1s) with [1, 0, ..., 0]. Since is_target
        # is used afterwards in the loss computation, this change becomes irrelevant and let us use 
        # vectorization when computing the similarities.
        new_row = tf.concat([[1], tf.zeros(tf.shape(item_indexes)[2] - 1, dtype=tf.int32)], axis=0)
        y_true_updated = tf.where(y_true != tf.repeat(-1, tf.shape(item_indexes)[2]), y_true, new_row)

        # Build the objective items shape: (B, GB, 1).
        objective_shape = tf.shape(item_indexes)
        objective_shape = tf.tensor_scatter_nd_update(objective_shape, [[2]], [1])

        # Objetive items contains only the index of the target to compared (where y_true is 1).
        # Comparative items contains the indexes of all the other items (where y_true is 0).
        objective_items = tf.reshape(tf.boolean_mask(item_indexes, y_true_updated), objective_shape)
        comparative_items =  item_indexes 

        objectives_embeddings = self.model.get_target_embeddings(objective_items)
        comparative_embeddings = self.model.get_target_embeddings(comparative_items)
        
        objectives_embeddings, _ = tf.linalg.normalize(objectives_embeddings, axis=-1)
        comparative_embeddings, _ = tf.linalg.normalize(comparative_embeddings, axis=-1)

        similarities = tf.linalg.matmul(objectives_embeddings, comparative_embeddings, transpose_b=True)
        similarities = tf.squeeze(similarities, axis=2)
        return similarities

