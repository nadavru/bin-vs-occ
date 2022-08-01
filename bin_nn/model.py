from .model_utils import Binary_model
from .data_utils import BinaryDataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os


class Model:
    def __init__(self, d, num_epochs = 20, device="cuda", silent=True):
        ####################################################
        self.num_epochs = num_epochs
        self.lr = 0.001
        self.print_every = 1
        self.batch_size = 16

        hidden = [50,100]
        ####################################################
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else 'cpu')
        self.silent = silent
        self.model = Binary_model(d, hidden).to(self.device)
        self.lossFunc = nn.BCEWithLogitsLoss(reduction="sum")

    def _print(self, str):
        if not self.silent:
            print(str)

    def load_model(self, folder_path):
        assert os.path.isfile(f"{folder_path}/model.pt"), "model cannot be found! abort."

        self.model.load_state_dict(torch.load(f"{folder_path}/model.pt", map_location=self.device))
        self.model.to(self.device)

        self._print(f"loaded from folder {folder_path}.")

    def save_model(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.model.cpu().state_dict(), f"{folder_path}/model.pt")
        self.model.to(self.device)
        
        self._print(f"saved in folder {folder_path}.")
    
    def reset_model(self):
        #TODO check
        def weight_reset(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        self.model.apply(weight_reset)

    def fit(self, train_set):
        # (np array: [N, d], np array: [N,])

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        train_dataset = BinaryDataset(*train_set)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            total_loss = 0
            for X, Y in dataloader:
                if X.shape[0]==1:
                    continue
                # X: [b,d]
                # Y: [b]
                X, Y = X.to(self.device).float(), Y.to(self.device).float()
                logits = self.model(X)

                loss = self.lossFunc(logits, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()

            # print statistics
            avg_loss = total_loss / len(train_dataset)
            if epoch % self.print_every == (self.print_every-1):
                self._print(f'{epoch + 1} loss: {avg_loss}')

    def pred(self, sample):

        self.model.eval()

        dataloader = DataLoader(sample, batch_size=self.batch_size, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for X in dataloader:
                # X: [b,d]
                if X.shape[0]==1:
                    continue
                X = X.to(self.device).float()
                logits = self.model(X)
                # probs = torch.sigmoid(logits) : no need
                
                all_scores.append(logits.detach())

        all_scores = torch.cat(all_scores).cpu().numpy()
        return all_scores
