import numpy as np
from abc import abstractmethod


class TestStatistic:
    def __init__(self, model, aggregate_function='avg', n_iterations=200):
        self.model = model
        assert aggregate_function in self.aggregate_functions
        self.aggregate = self.aggregate_functions[aggregate_function]
        self.n_iterations = n_iterations

    aggregate_functions = {
        'avg': np.average,
        'max': np.max,
        'min': np.min,
        'median': np.median,
    }

    def __call__(self, in_sample, out_sample):
        in_score = self.aggregate(self.predict(in_sample))
        out_score = self.aggregate(self.predict(out_sample))
        return out_score - in_score
    
    def two_sample_test(self, in_sample, out_sample):
        in_scores = self.predict(in_sample)
        out_scores = self.predict(out_sample)
        in_scores_len = len(in_scores)
        
        counter = 1
        t_star = self.aggregate(out_scores) - self.aggregate(in_scores)
        all_scores = np.concatenate((in_scores, out_scores))
        for _ in range(self.n_iterations):
            scores_perm = np.random.permutation(all_scores)
            t_k = self.aggregate(scores_perm[in_scores_len:]) - self.aggregate(scores_perm[:in_scores_len])
            counter += (t_star<=t_k)
        p_value = counter/(self.n_iterations+1)
        
        return p_value
    
    def train(self, train_set):
        self.model.fit(train_set)

    def predict(self, sample):
        return self.model.pred(sample)
    
    def load(self, folder_path):
        self.model.load_model(folder_path)
    
    def save(self, folder_path):
        self.model.save_model(folder_path)
    
    def reset(self):
        self.model.reset_model()
