from torch import nn
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from regularizers import compute_regularizer_term

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

    def Build(self, hiddendim, learningRate=0.001, weight_decay=0.0001,hidden_prior='Uniform',hidden_prior2='False',hyperatio=1.,lambda_=0.001,c=1):


        inputdim = self.X_train.shape[2]
        outputsize = self.y_train.shape[2]

        self.input_dim = inputdim
        self.hidden_dim = hiddendim
        self.out_dim = outputsize
        self.gru = nn.GRU(self.input_dim, self.hidden_dim, num_layers=1, bidirectional=True,
                          batch_first=True).to(self.device, non_blocking=True)
        self.fc_out = nn.Linear(self.hidden_dim * 2, self.out_dim).to(self.device, non_blocking=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate, weight_decay=weight_decay)

        # Set up the cosine learning rate scheduler
        lr_min = learningRate * 0.1
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=lr_min)
        # regularizer
        self.lambda_=lambda_
        self.hidden_prior=hidden_prior
        self.hidden_prior2=hidden_prior2
        self.hyperatio=hyperatio
        self.c = c


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
            self.optimizer.zero_grad()
            output = self(x_latent)
            loss = self.loss_fn(output, V_train)
            reg_term = self.hidden_layer_regularizer()
            loss = loss + reg_term
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            MSE_train_linear_epoch[i] = loss

        return MSE_cv_linear_epoch, MSE_train_linear_epoch

    def predict_velocity(self, x_latent):

        self.eval()
        with torch.no_grad():
            v_predict, hidden_states = self.gru(x_latent)
            v_predict = self.fc_out(v_predict)

        v_predict = v_predict.cpu().detach().numpy()
        hidden_states = hidden_states.cpu().detach().numpy()
        return v_predict, hidden_states

    def hidden_layer_regularizer(self):
        """
            Compute the regularization loss of the hidden layers.
        """

        assert self.hidden_prior in ['Uniform', 'Cauchy', 'Gaussian', 'Laplace','Sinc_squared', 'negcos', 'SinFouthPower'], "Change the data name to 'uniform', 'Cauchy', 'Gaussian', 'Laplace', or 'Sinc_squared','Sinc_squared', 'negcos', 'SinFouthPower'."
        reg_loss = torch.tensor([0.0], device=self.device)
        
        
        if self.hidden_prior != "Uniform":    
            for name, param in self.named_parameters():

                if (name[0:8] not in ['gru.bias'])&(name[-4:]!='bias'):
                #if (name[0:3] == 'fc_') &(name[-4:]!='bias'):
                    reg_loss = reg_loss + compute_regularizer_term(wgts=param,
                                                                    lambda_=self.lambda_,
                                                                    hidden_prior=self.hidden_prior,
                                                                    hidden_prior2=self.hidden_prior2,
                                                                    hyperatio=self.hyperatio,
                                                                    c=self.c)
        return reg_loss
