from torch import nn
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR


class BiGRU(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # Convert to PyTorch tensors
        self.X_train = torch.from_numpy(self.X_train).float().to(self.device, non_blocking=True)
        self.X_test = torch.from_numpy(self.X_test).float().to(self.device, non_blocking=True)
        self.y_train = torch.from_numpy(self.y_train).float().to(self.device, non_blocking=True)
        self.y_test = torch.from_numpy(self.y_test).float().to(self.device, non_blocking=True)

    def Build(self, hiddendim, learningRate=0.001, weight_decay=0.0001):
        inputdim = self.X_train.shape[2]
        outputsize = self.y_train.shape[2]

        self.input_dim = inputdim
        self.hidden_dim = hiddendim
        self.out_dim = outputsize
        self.gru = nn.GRU(self.input_dim, self.hidden_dim,bidirectional=True,
                          batch_first=True).to(self.device,non_blocking=True)
        self.fc_out = nn.Linear(self.hidden_dim * 2, self.out_dim).to(self.device, non_blocking=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate, weight_decay=weight_decay)

        # Set up the cosine learning rate scheduler
        lr_min = learningRate * 0.1
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=lr_min)
    def forward(self, xt):
        xt = xt.to(self.device, non_blocking=True)
        gruoutput, h = self.gru(xt)
        velocity = self.fc_out(gruoutput)
        return velocity

    def train_fit(self, N_epoches):
        '''

        :param N_epoches:
        :param x_latent: train X
        :param V_train: train Y
        :param x_latent_cv: test X
        :param V_cv: test Y
        :return:
        '''
        x_latent = self.X_train
        V_train = self.y_train
        x_latent_cv = self.X_test
        V_cv = self.y_test

        MSE_cv_linear_epoch = torch.empty([N_epoches])
        MSE_train_linear_epoch = torch.empty([N_epoches])

        for i in tqdm(range(N_epoches), 'epochs'):
            v_predict_cv = self(x_latent_cv)
            MSE_cv_linear_epoch[i] = self.loss_fn(v_predict_cv, V_cv)

            self.train()

            output = self(x_latent)
            loss = self.loss_fn(output, V_train)

            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            MSE_train_linear_epoch[i] = loss

        return MSE_cv_linear_epoch, MSE_train_linear_epoch

    def predict_velocity(self, x_latent):
        self.eval()
        with torch.no_grad():
            v_predict = self(x_latent).detach().cpu().numpy()
        return v_predict
