from sklearn.ensemble import RandomForestClassifier
import pickle
import os

class Model:
    def __init__(self, silent=True):
        self.model = RandomForestClassifier()
        self.silent = silent

    def _print(self, str):
        if not self.silent:
            print(str)

    def load_model(self, folder_path):
        assert os.path.isfile(f"{folder_path}/model.sav"), "model cannot be found! abort."

        self.model = pickle.load(open(f"{folder_path}/model.sav", 'rb'))

        self._print(f"loaded from folder {folder_path}.")

    def save_model(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        pickle.dump(self.model, open(f"{folder_path}/model.sav", 'wb'))
        
        self._print(f"saved in folder {folder_path}.")
    
    def reset_model(self):
        self.model = RandomForestClassifier()

    def fit(self, train_set):
        # (np array: [N, d], np array: [N,])

        self.model.fit(*train_set)

    def pred(self, sample):

        probs = self.model.predict_proba(sample)
        return probs[:,1]
