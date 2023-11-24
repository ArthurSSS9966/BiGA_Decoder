from torch import nn
import torch
from tqdm import tqdm
import math
from torch.optim.lr_scheduler import CosineAnnealingLR


class PositionalEncoding(torch.nn.Module):
    def __init__(self,hiddendim,lens, device):
        super(PositionalEncoding,self).__init__()
        self.hiddendim = hiddendim
        self.positional_encoding = self.generate_positional_encoding(hiddendim,lens).to(device,non_blocking=True)
        
    def generate_positional_encoding(self,hiddendim,lens):
        pe = torch.zeros(lens,hiddendim)
        position = torch.arange(0,lens).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,hiddendim,2)*-(math.log(10000.0) / hiddendim))

        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)

        return pe.unsqueeze(0)
    
    def forward(self,x):
        x = x + self.positional_encoding[:,:x.size(1)]
        return x
    

class TransformerModel(torch.nn.Module):

    def __init__(self, device):
        super().__init__()
        self.device = device
        self.loss_fn = torch.nn.MSELoss()

    def load_data(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.timestep = X_train.shape[1]
        # Convert to PyTorch tensors
        self.X_train = torch.from_numpy(self.X_train).float().to(self.device, non_blocking=True)
        self.X_test = torch.from_numpy(self.X_test).float().to(self.device, non_blocking=True)
        self.y_train = torch.from_numpy(self.y_train).float().to(self.device, non_blocking=True)
        self.y_test = torch.from_numpy(self.y_test).float().to(self.device, non_blocking=True)

    def Build(self,hiddendim, middle_dim, nhead,num_layers, learningRate=0.001, weight_decay=0.0001):
        '''
        Transformer parameters: 
        nhead =                             (default)
        num_encoder_layer =                 (default)
        dropout = 0.1                       (default)
        activation = relu                   (default)
        norm_first                          (default)
        '''
        inputdim = self.X_train.shape[2]
        outputsize = self.y_train.shape[2]
        self.input_dim = inputdim
        self.hiddendim = hiddendim
        self.nhead=nhead
        self.num_layers = num_layers
        self.output_dim = outputsize
        #self.embedding = nn.Linear(self.input_dim,self.hiddendim).to(self.device,non_blocking=True)
        self.middle_dim = middle_dim
        self.embeddingv = nn.Linear(self.output_dim,self.hiddendim).to(self.device,non_blocking=True)
        self.position_encode = PositionalEncoding(self.hiddendim,self.timestep, self.device)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hiddendim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer,
                                                         num_layers=self.num_layers).to(self.device, non_blocking=True)

        self.fc_out = nn.Linear(self.hiddendim, self.middle_dim).to(self.device, non_blocking=True)
        self.fc_out1 = nn.Linear(self.middle_dim, self.middle_dim).to(self.device, non_blocking=True)
        self.fc_out2 = nn.Linear(self.middle_dim, self.middle_dim).to(self.device, non_blocking=True)
        self.fc_out3 = nn.Linear(self.middle_dim, self.middle_dim).to(self.device, non_blocking=True)
        self.fc_out4 = nn.Linear(self.middle_dim, self.output_dim).to(self.device, non_blocking=True)
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learningRate, weight_decay=weight_decay)
        # Set up the cosine learning rate scheduler
        lr_min = learningRate * 0.5
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=100, eta_min=lr_min)

    def forward(self,xt):
        xt = xt.to(self.device, non_blocking=True)
        #xt = self.embedding(xt)
        trans_out = self.position_encode(xt)

        trans_out = self.transformer_encoder(trans_out)
        trans_out = self.fc_out(trans_out)
        trans_out = self.fc_out1(trans_out)
        trans_out = self.fc_out2(trans_out)
        trans_out = self.fc_out3(trans_out)
        velocity = self.fc_out4(trans_out)

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