import tensorflow.keras
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.engine import data_adapter

from .utils import stack_indexes_with_predictions
from aprec.recommenders.dnn_sequential_recommender.models.sequential_recsys_model import SequentialRecsysModel
from .sasrec_multihead_attention import multihead_attention
import tensorflow as tf

#https://ieeexplore.ieee.org/abstract/document/8594844
#the code is ported from original code
#https://github.com/kang205/SASRec
class SASRecDreji(SequentialRecsysModel):
    def __init__(self, output_layer_activation='linear', embedding_size=64,
                 max_history_len=64, 
                 dropout_rate=0.2,
                 num_blocks=3,
                 num_heads=1,
                 reuse_item_embeddings=True,        # use same item embeddings for
                                                    # sequence embedding and for the embedding matrix
                 encode_output_embeddings=False,    #encode item embeddings with a dense layer
                                                    # may be useful if we reuse item embeddings
                 vanilla=False, #vanilla sasrec model uses shifted sequence prediction at the training time,
                                # used with NegativePerPositiveTargetBuilder.
                 sampled_target=None, # Used with FullMatrixTargetBuilder or SampledTargetMatrixBuilder.
                 negative_sampling=None, # Used with NegativeSamplingTargetBuilder.
                 use_indexed_y = False,
                 ):
        super().__init__(output_layer_activation, embedding_size, max_history_len)
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.reuse_item_embeddings=reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.sampled_target = sampled_target
        self.vanilla = vanilla
        self.negative_sampling = negative_sampling
        self.latest_model = None
        self.use_indexed_y = use_indexed_y


    encode_embedding_with_dense_layer = False,

    def get_model(self):
        model = OwnSasrecModelDreji(
            num_items=self.num_items,
            batch_size=self.batch_size,
            output_layer_activation=self.output_layer_activation,
            embedding_size=self.embedding_size,
            max_history_length=self.max_history_length,
            dropout_rate=self.dropout_rate,
            num_blocks=self.num_blocks,
            num_heads=self.num_heads,
            reuse_item_embeddings=self.reuse_item_embeddings,
            encode_output_embeddings=self.encode_output_embeddings,
            sampled_target=self.sampled_target,
            negative_sampling=self.negative_sampling,
            vanilla=self.vanilla
        )
        self.latest_model = model
        return model



class OwnSasrecModelDreji(tensorflow.keras.Model):
    def __init__(self, num_items, batch_size, output_layer_activation='linear', embedding_size=64,
                 max_history_length=64, dropout_rate=0.5, num_blocks=2, num_heads=1,
                 reuse_item_embeddings=False,
                 encode_output_embeddings=False,
                 sampled_target=None,
                 negative_sampling=None,
                 vanilla = False, #vanilla implementation; uses negative sampling and shifted sequence prediction
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert(not (vanilla and sampled_target), "only vanilla or sampled targetd strategy can be used at once")
        self.output_layer_activation = output_layer_activation
        self.embedding_size = embedding_size
        self.max_history_length = max_history_length
        self.dropout_rate = dropout_rate
        self.num_blocks = num_blocks
        self.num_heads = num_heads
        self.num_items = num_items
        self.batch_size = batch_size
        self.sampled_target = sampled_target
        self.negative_sampling = negative_sampling
        self.reuse_item_embeddings=reuse_item_embeddings
        self.encode_output_embeddings = encode_output_embeddings
        self.vanilla = vanilla

        self.positions = tf.constant(tf.tile(tf.expand_dims(tf.range(self.max_history_length), 0), [self.batch_size, 1]))

        self.item_embeddings_layer = layers.Embedding(self.num_items + 1, output_dim=self.embedding_size,
                                                       dtype='float32')
        self.postion_embedding_layer = layers.Embedding(self.max_history_length,
                                                        self.embedding_size,
                                                         dtype='float32')

        self.embedding_dropout = layers.Dropout(self.dropout_rate)

        self.attention_blocks = []
        for i in range(self.num_blocks):
            block_layers = {
                "first_norm": layers.LayerNormalization(),
                "attention_layers": {
                    "query_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "key_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "val_proj": layers.Dense(self.embedding_size, activation='linear'),
                    "dropout": layers.Dropout(self.dropout_rate),
                },
                "second_norm": layers.LayerNormalization(),
                "dense1": layers.Dense(self.embedding_size, activation='relu'),
                "dense2": layers.Dense(self.embedding_size),
                "dropout": layers.Dropout(self.dropout_rate)
            }
            self.attention_blocks.append(block_layers)
        self.output_activation = activations.get(self.output_layer_activation)
        self.seq_norm = layers.LayerNormalization()
        self.all_items = tf.range(0, self.num_items)
        if not self.reuse_item_embeddings:
            self.output_item_embeddings = layers.Embedding(self.num_items, self.embedding_size)

        if self.encode_output_embeddings:
            self.output_item_embeddings_encode = layers.Dense(self.embedding_size, activation='gelu')


    def block(self, seq, mask, i):
        x = self.attention_blocks[i]["first_norm"](seq)
        queries = x
        keys = seq
        x = multihead_attention(queries, keys, self.num_heads, self.attention_blocks[i]["attention_layers"],
                                     causality=True)
        x =x + queries
        x = self.attention_blocks[i]["second_norm"](x)
        residual = x
        x = self.attention_blocks[i]["dense1"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x = self.attention_blocks[i]["dense2"](x)
        x = self.attention_blocks[i]["dropout"](x)
        x += residual
        x *= mask
        return x

    def call(self, inputs,  **kwargs):
        input_ids = inputs[0]
        training = kwargs['training']
        seq_emb = self.get_seq_embedding(input_ids, training)
        target_ids = self.get_target_ids(inputs)
        target_embeddings = self.get_target_embeddings(target_ids)
        if self.vanilla:
            positive_embeddings = target_embeddings[:,:,0,:]
            negative_embeddings = target_embeddings[:,:,1,:]
            positive_results = tf.reduce_sum(seq_emb*positive_embeddings, axis=-1)
            negative_results = tf.reduce_sum(seq_emb*negative_embeddings, axis=-1)
            output = tf.stack([positive_results, negative_results], axis=-1)
        elif self.sampled_target:
            seq_emb = seq_emb[:, -1, :]
            output = tf.einsum("ij,ikj ->ik", seq_emb, target_embeddings)
        elif self.negative_sampling:
            seq_emb = tf.expand_dims(seq_emb, axis=-2)
            # Dimension i=1 (we just expanded in the previous line), so we squeeze it
            # in the same operation. Equivalent to "...ij,...kj ->...ik" followed by
            # squeeze.
            output = tf.einsum("...ij,...kj ->...k", seq_emb, target_embeddings)
        else:
            seq_emb = tf.einsum("...ij,...kj ->...ik", seq_emb, target_embeddings)

        output = self.output_activation(output)
        return output
    
    def score_all_items(self, inputs):
        input_ids = inputs[0]
        seq_emb = self.get_seq_embedding(input_ids)
        seq_emb = seq_emb[:, -1, :]
        target_ids = self.all_items
        target_embeddings = self.get_target_embeddings(target_ids)
        output = seq_emb @ tf.transpose(target_embeddings)
        output = self.output_activation(output)
        return output

    def get_target_embeddings(self, target_ids):
        if self.reuse_item_embeddings:
            target_embeddings = self.item_embeddings_layer(target_ids)
        else:
            target_embeddings = self.output_item_embeddings(target_ids)
        if self.encode_output_embeddings:
            target_embeddings = self.output_item_embeddings_encode(target_embeddings)
        return target_embeddings

    def get_seq_embedding(self, input_ids, training=None):
        seq = self.item_embeddings_layer(input_ids)
        mask = tf.expand_dims(tf.cast(tf.not_equal(input_ids, self.num_items), dtype=tf.float32), -1)
        pos_embeddings = self.postion_embedding_layer(self.positions)
        seq += pos_embeddings
        seq = self.embedding_dropout(seq)
        seq *= mask
        for i in range(self.num_blocks):
            seq = self.block(seq, mask, i)
        seq_emb = self.seq_norm(seq)
        return seq_emb

    def get_target_ids(self, x):
        if self.vanilla \
                or (self.sampled_target is not None) \
                or (self.negative_sampling is not None):
            target_ids = x[1]
        else:
            target_ids = self.all_items
        return target_ids

    def set_config(self, new_config):
        self.vanilla = new_config["vanilla"]
        self.sampled_target = new_config["sampled_target"]
        self.negative_sampling = new_config["negative_sampling"]
        self.use_indexed_y = new_config["use_indexed_y"]

    def prepare_y_for_loss(self, x, y):
        if self.use_indexed_y:
            target_ids = self.get_target_ids(x)
            y = stack_indexes_with_predictions(indexes=target_ids, predictions=y)
        return y

# This method is a copy of the original keras.model.train_step method found at
# https://github.com/keras-team/keras/blob/3a33d53ea4aca312c5ad650b4883d9bac608a32e/keras/engine/training.py#L755
# Needs to be customize to use a loss function that takes additional parameters.
# This is the keras code from version 2.6.0!!!! Not the newest one.
    def train_step(self, data):
        """The logic for one training step.

        This method can be overridden to support custom training logic.
        For concrete examples of how to override this method see
        [Customizing what happends in fit](https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit).
        This method is called by `Model.make_train_function`.

        This method should contain the mathematical logic for one step of training.
        This typically includes the forward pass, loss calculation, backpropagation,
        and metric updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_train_function`, which can also be overridden.

        Args:
        data: A nested structure of `Tensor`s.

        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned. Example:
        `{'loss': 0.2, 'accuracy': 0.7}`.

        """
        # These are the only transformations `Model.fit` applies to user-input
        # data when a `tf.data.Dataset` is provided.
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y = self.prepare_y_for_loss(x, y)
        
        # Run forward pass.
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = self.prepare_y_for_loss(x, y_pred)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.losses)
        # Run backwards pass.
        self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)

        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics

# This method is a copy of the original keras.model.test_step method found at
# https://github.com/keras-team/keras/blob/3a33d53ea4aca312c5ad650b4883d9bac608a32e/keras/engine/training.py#L1241
# Needs to be customize to use a loss function that takes additional parameters.
# This is the keras code from version 2.6.0!!!! Not the newest one.
    def test_step(self, data):
        """The logic for one evaluation step.

        This method can be overridden to support custom evaluation logic.
        This method is called by `Model.make_test_function`.

        This function should contain the mathematical logic for one step of
        evaluation.
        This typically includes the forward pass, loss calculation, and metrics
        updates.

        Configuration details for *how* this logic is run (e.g. `tf.function` and
        `tf.distribute.Strategy` settings), should be left to
        `Model.make_test_function`, which can also be overridden.

        Args:
        data: A nested structure of `Tensor`s.

        Returns:
        A `dict` containing values that will be passed to
        `tf.keras.callbacks.CallbackList.on_train_batch_end`. Typically, the
        values of the `Model`'s metrics are returned.
        """
        data = data_adapter.expand_1d(data)
        x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)
        y_pred = self(x, training=False)
        
        y = self.prepare_y_for_loss(x, y)
        y_pred = self.prepare_y_for_loss(x, y_pred)
        
        # Updates stateful loss metrics.
        self.compiled_loss(
            y, y_pred, sample_weight, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
        # Collect metrics to return
        return_metrics = {}
        for metric in self.metrics:
            result = metric.result()
            if isinstance(result, dict):
                return_metrics.update(result)
            else:
                return_metrics[metric.name] = result
        return return_metrics