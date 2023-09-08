from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_repro_paper.common_benchmark_config import *
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_sampling_target_builder import NegativeSamplingTargetBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.negative_per_positive_target import NegativePerPositiveTargetBuilder

from collections import defaultdict
import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def dreji_sasrec_vanilla(HISTORY_LEN=50):
    sasrec_arc = SASRecDreji(
        max_history_len=HISTORY_LEN, 
        dropout_rate=0.2,
        num_heads=1,
        num_blocks=2,
        vanilla=True,
        sampled_target=False,
        embedding_size=50,
    )
    return dnn(
        model_arch=sasrec_arc,
        loss=BCELossDreji(model_arc=sasrec_arc),
        sequence_splitter=ShiftedSequenceSplitter,
        optimizer=Adam(beta_2=0.98),
        target_builder= lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN),
        metric=BCELossDreji(model_arc=sasrec_arc),
    )

LIMIT_HOURS = 10
TRAINING_TIME_LIMIT = 3600 * LIMIT_HOURS

def dreji_sasrec(
        HISTORY_LEN=50,
        n_samples=501,
        alpha=7.5,
        training_time_limit=TRAINING_TIME_LIMIT,
        early_stop_epochs=10,
    ):
    sasrec_arc = SASRecDreji(
        max_history_len=HISTORY_LEN, 
        dropout_rate=0.2,
        num_heads=1,
        num_blocks=2,
        vanilla=False,
        sampled_target=False,
        negative_sampling=True,
        embedding_size=50,
    )
    return dnn(
        sasrec_arc,
        BCELossDreji(model_arc=sasrec_arc, alpha=alpha),
        ShiftedSequenceSplitter,
        optimizer=Adam(beta_2=0.98),
        target_builder=lambda: NegativeSamplingTargetBuilder(n_samples=n_samples),
        metric=BCELossDreji(model_arc=sasrec_arc, alpha=alpha),
        training_time_limit=training_time_limit,
        early_stop_epochs=early_stop_epochs,
    )

def dreji_sasrec_two_steps_training(
        HISTORY_LEN=50,
        n_samples=101,
        alpha=7.5,
        training_time_limit=TRAINING_TIME_LIMIT,
        early_stop_epochs=5,
    ):
    sasrec_arc = SASRecDreji(
        max_history_len=HISTORY_LEN, 
        dropout_rate=0.2,
        num_heads=1,
        num_blocks=2,
        vanilla=True,
        sampled_target=False,
        negative_sampling=False,
        embedding_size=50,
    )
    
    first_step_config = defaultdict(bool, {
        "vanilla": True,
    })
    second_step_config =  defaultdict(bool, {
        "negative_sampling": True,
        "use_indexed_y": True,
        "freeze_item_embeddings": True,
    })

    return dnn(
        sasrec_arc,
        BCELoss(),
        ShiftedSequenceSplitter,
        optimizer=Adam(beta_2=0.98),
        target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN),
        metric=BCELoss(),
        training_time_limit=training_time_limit,
        early_stop_epochs=early_stop_epochs,
        second_step_loss=BCELossDreji(model_arc=sasrec_arc, alpha=alpha, use_rmse=True),
        second_step_metric=BCELossDreji(model_arc=sasrec_arc, alpha=alpha, use_rmse=True),
        second_step_targets_builder=lambda: NegativeSamplingTargetBuilder(n_samples=n_samples),
        second_step_optimizer=Adam(beta_2=0.98, learning_rate=0.00001),
        first_step_config = first_step_config,
        second_step_config = second_step_config,
    )


DATASET = "BERT4rec.ml-1m"
USERS_FRACTIONS = [1]
N_VAL_USERS=2048
MAX_TEST_USERS=6040
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
TARGET_ITEMS_SAMPLER = PopTargetItemsSampler(101)
RECOMMENDATIONS_LIMIT = 100

METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), MAP(10)]

RECOMMENDERS = {
    #"top_recommender": top_recommender,
    #"MF-BPR": lambda: lightfm_recommender(30, 'bpr'),
    #"dreji_sasrec_vanilla": dreji_sasrec_vanilla,
    #"dreji_sasrec": dreji_sasrec,
    "dreji_sasrec_two_steps_training": dreji_sasrec_two_steps_training
}