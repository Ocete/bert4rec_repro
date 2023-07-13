from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.bert4rec_repro_paper.common_benchmark_config import *
from aprec.recommenders.dnn_sequential_recommender.target_builders.full_matrix_targets_builder import FullMatrixTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.target_builders.sampled_matrix_target_builder import SampledMatrixBuilder

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

def top_recommender():
    return FilterSeenRecommender(TopRecommender())

def lightfm_recommender(k, loss):
    return FilterSeenRecommender(LightFMRecommender(k, loss))

def dreji_sasrec_sampled_targets(HISTORY_LEN=50, n_samples=101):
    sasrec_arc = SASRecDreji(
        max_history_len=HISTORY_LEN, 
        dropout_rate=0.2,
        num_heads=1,
        num_blocks=2,
        vanilla=False,
        sampled_targets=True,
        embedding_size=50,
    )
    return dnn(
        sasrec_arc,
        BCELossDreji(model_arc=sasrec_arc),
        ShiftedSequenceSplitter,
        optimizer=Adam(beta_2=0.98),
        target_builder= lambda: SampledMatrixBuilder(n_samples=n_samples),
        metric=BCELossDreji(model_arc=sasrec_arc),
    )

def dreji_sasrec(HISTORY_LEN=50, n_samples=101):
    sasrec_arc = SASRecDreji(
        max_history_len=HISTORY_LEN, 
        dropout_rate=0.2,
        num_heads=1,
        num_blocks=2,
        vanilla=False,
        embedding_size=50,
    )
    return dnn(
        sasrec_arc,
        BCELossDreji(model_arc=sasrec_arc),
        ShiftedSequenceSplitter,
        optimizer=Adam(beta_2=0.98),
        target_builder= lambda: FullMatrixTargetsBuilder(),
        metric=BCELossDreji(model_arc=sasrec_arc),
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
    #"vanilla_sasrec": vanilla_sasrec,
    #"dreji_sasrec_sampled_targets": dreji_sasrec_sampled_targets,
    "dreji_sasrec": dreji_sasrec,
}