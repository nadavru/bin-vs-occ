from .model_utils import F_model,G_model
from .data_utils import PaperDataset
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os


# By the paper
class Model:
    def __init__(self, d, num_epochs = 20, device="cuda", silent=True):
        ####################################################
        self.num_epochs = num_epochs
        self.lr = 0.001
        self.print_every = 1
        self.batch_size = 16

        self.temperature = 0.01
        f_hidden = [50,100]
        g_hidden = [25,50]
        u = 50
        self.k = max(d//4, 1) #TODO based on d
        ####################################################
        self.device = torch.device(device if torch.cuda.is_available() and device.startswith("cuda") else 'cpu')
        self.silent = silent
        self.m = d - self.k
        self.f = F_model(self.m, u, f_hidden).to(self.device)
        self.g = G_model(self.k, u, g_hidden).to(self.device)
        self.lossFunc = nn.CrossEntropyLoss()
        self.scoreFunc = nn.CrossEntropyLoss(reduction="none")

    def _print(self, str):
        if not self.silent:
            print(str)

    def load_model(self, folder_path):
        assert os.path.isfile(f"{folder_path}/f.pt") and os.path.isfile(f"{folder_path}/g.pt"), "models cannot be found! abort."

        self.f.load_state_dict(torch.load(f"{folder_path}/f.pt", map_location=self.device))
        self.g.load_state_dict(torch.load(f"{folder_path}/g.pt", map_location=self.device))
        self.f.to(self.device)
        self.g.to(self.device)

        self._print(f"loaded from folder {folder_path}.")

    def save_model(self, folder_path):
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(self.f.cpu().state_dict(), f"{folder_path}/f.pt")
        torch.save(self.g.cpu().state_dict(), f"{folder_path}/g.pt")
        self.f.to(self.device)
        self.g.to(self.device)
        
        self._print(f"saved in folder {folder_path}.")
    
    def reset_model(self):
        def weight_reset(m):
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
                m.reset_parameters()
        self.f.apply(weight_reset)
        self.g.apply(weight_reset)

    def fit(self, train_set):
        # np array: [N, d]

        self.f.train()
        self.g.train()
        opt = torch.optim.Adam(list(self.f.parameters()) + list(self.g.parameters()), lr=self.lr)
        train_dataset = PaperDataset(train_set, self.k)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in range(self.num_epochs):
            total_loss = 0
            total_rows = 0
            for a, b in dataloader:
                if a.shape[0]==1:
                    continue
                # a: [b,m+1,k] -> G
                # b: [b,m+1,m] -> F
                a, b = a.to(self.device).float(), b.to(self.device).float()
                batch_len = a.shape[0]
                anchor = self.f(b)  # [b,m+1,u]
                pos = self.g(a)  # [b,m+1,u]
                pos = torch.permute(pos, (0, 2, 1))  # [b,u,m+1]
                scores = torch.bmm(anchor, pos) / self.temperature  # [b,m+1,m+1]
                labels = torch.arange(self.m + 1).repeat(batch_len, 1).to(self.device)  # [b,m+1]

                loss = self.lossFunc(scores, labels)
                opt.zero_grad()
                loss.backward()
                opt.step()

                total_loss += loss.item()
                total_rows += batch_len

            # print statistics
            avg_loss = total_loss / total_rows
            if epoch % self.print_every == (self.print_every-1):
                self._print(f'{epoch + 1} loss: {avg_loss}')

    def pred(self, sample):

        self.f.eval()
        self.g.eval()

        train_dataset = PaperDataset(sample, self.k)
        dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for a, b in dataloader:
                if a.shape[0]==1:
                    continue
                # a: [b,m+1,k] -> G
                # b: [b,m+1,m] -> F
                a, b = a.to(self.device).float(), b.to(self.device).float()
                batch = a.shape[0]
                anchor = self.f(b)  # [b,m+1,u]
                pos = self.g(a)  # [b,m+1,u]
                pos = torch.permute(pos, (0, 2, 1))  # [b,u,m+1]
                scores = torch.bmm(anchor, pos) / self.temperature  # [b,m+1,m+1]
                labels = torch.arange(self.m + 1).repeat(batch, 1).to(self.device)  # [b,m+1]

                score = self.scoreFunc(scores, labels).sum(1)
                all_scores.append(score.detach())

        all_scores = torch.cat(all_scores).cpu().numpy()
        return all_scores
