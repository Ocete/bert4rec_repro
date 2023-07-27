import random

import numpy as np
from aprec.recommenders.dnn_sequential_recommender.target_builders.target_builders import TargetBuilder


class SampledMatrixSingleTargetBuilder(TargetBuilder):
    def __init__(self, max_target_label=1.0, n_samples=101):
        self.max_target_label = max_target_label
        self.n_samples = n_samples

    def build(self, user_targets):
        all_items = list(range(self.n_items))
        self.target_matrix = []
        self.target_ids = []
        for i in range(len(user_targets)): 
            targets = []
            target_ids =  []
            sampled = set()
            # Only use the last user action as target
            single_target_id = user_targets[i][-1][1]
            target_ids.append(single_target_id)
            sampled.add(single_target_id)
            targets.append(self.max_target_label)
            while(len(targets) < self.n_samples):
                negatives = np.random.choice(all_items, self.n_samples - len(targets))
                for item_id in negatives:
                    if item_id not in sampled:
                        sampled.add(item_id)
                        target_ids.append(item_id)
                        targets.append(0.0)
            targets_with_ids = list(zip(targets, target_ids))
            random.shuffle(targets_with_ids)
            targets, target_ids = zip(*targets_with_ids)
            self.target_matrix.append(targets)
            self.target_ids.append(target_ids)
        self.target_matrix = np.array(self.target_matrix)
        self.target_ids = np.array(self.target_ids)

    def get_targets(self, start, end):
        target_inputs = [self.target_ids[start:end]]
        target_outputs = self.target_matrix[start:end]
        '''
        print_bicho('get_targets -- target_inputs', target_inputs)
        print_bicho('get_targets -- target_outputs', target_outputs, sum=True)
        '''
        return target_inputs, target_outputs

def print_bicho(message, bicho, shape=True, sum=False):
    print('\t\t' + message)
    if shape:
        try:
            print('shape: {}'.format(np.array(bicho).shape))
        except:
            print('\tinvalid shape')
    if sum:
        print('sum: {}'.format(np.sum(bicho, axis=1)), end='')
    print(bicho)