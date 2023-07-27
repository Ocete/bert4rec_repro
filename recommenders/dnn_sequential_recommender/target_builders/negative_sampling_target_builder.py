from random import Random

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder


class NegativeSamplingTargetBuilder(TargetBuilder):
    def __init__(self, sequence_len=64, random_seed=31337, n_samples=101):
        self.sequence_len = sequence_len
        self.rng = np.random.default_rng(seed=random_seed)
        self.n_samples = n_samples

    def set_n_items(self, n):
        '''This method must be called before building the data'''
        self.n_items = n
        self.invalid_element_index = n
        self.all_items = list(range(self.n_items))

    def build(self, user_targets):
        self.inputs = []
        self.targets = []
        for i in range(len(user_targets)):
            user_inputs = []
            targets_for_user = []
            seq = user_targets[i]

            # Add invalid elements when the sequence is too small
            if len(seq) < self.sequence_len:
                invalids_shape = (self.sequence_len - len(seq), self.n_samples)
                n_invalid_elements = self.n_samples * (self.sequence_len - len(seq))
                user_inputs = [ np.repeat(self.invalid_element_index, n_invalid_elements).reshape(invalids_shape) ]
                targets_for_user = [ np.repeat(-1, n_invalid_elements).reshape(invalids_shape) ]

            for target in seq[-self.sequence_len:]:
                targets = np.eye(1, self.n_samples)[0]
                target_ids = self.negative_sampling(positive_id=target[1])

                shuffle_indexes = np.arange(self.n_samples)
                self.rng.shuffle(shuffle_indexes)
                targets = targets[shuffle_indexes]
                target_ids = target_ids[shuffle_indexes]
                
                user_inputs.append(target_ids)
                targets_for_user.append(targets)
            self.inputs.append(np.vstack(user_inputs))
            self.targets.append(np.vstack(targets_for_user))
        self.inputs = np.stack(self.inputs, axis=0)
        self.targets = np.stack(self.targets, axis=0)
    
    def get_targets(self, start, end):
        return [self.inputs[start:end]], self.targets[start:end]

    def negative_sampling(self, positive_id):
        '''
            Samples self.n_samples - 1 elements from self.all_items, excluding positive_index.

            Returns:
                A numpy array containing the positive_index in the first position, and the negatives samples following.
        '''
        final_indexes = [positive_id]
        sampled = set(final_indexes)
        while(len(final_indexes) < self.n_samples):
            negatives = self.rng.choice(self.all_items, self.n_samples - len(final_indexes))
            for item_id in negatives:
                if item_id not in sampled:
                    sampled.add(item_id)
                    final_indexes.append(item_id)
        return np.array(final_indexes)



