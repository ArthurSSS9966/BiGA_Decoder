import torch
import torch.nn as nn
import torch.nn.functional as func
import random

class InitGRU:
    def __init__(self,F,H,K0,m,n,T):
        '''
        yt和xtt1目前这里都是仅考虑一个sample, 后续需要扩展为多个sample
        '''
        self.F = F
        self.H = H
        self.K0 = K0
        self.m = m
        self.n = n
        self.T = T

class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H, ssModel.K0)   

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * 10 * 8        # 第一层的neuron 个数 (GRU前) 原(m + n) * (10) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 4          # 第二层的neuron 个数 (GRU后) 原 (m * n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)         # Init网络结构, 两个hidden layer neuro个数需要自己定义

    def InitSystemDynamics(self, F, H, K0):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)     # 有啥用?
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1)     # 有啥用?
        self.n = self.H.size()[0]

        # Set KalmanGain at t0
        self.KGain = K0.to(self.device,non_blocking = True)
        
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)   #在这里增加KG的维度(mn) self.m + self.n + self.m*self.n

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m ** 2 + self.n ** 2) * 10           # GRU 的hidden dimension; 原来是(m^2+n^2)*10, out of memory! 
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()


        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0):      #传入 xtt1_0这个在pipeline里trian的时候用到

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)

        self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

        self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)
    
    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):

        # Compute the 1-st moment of x based on model knowledge and without process noise
        self.state_process_prior_0 = torch.matmul(self.F, self.state_process_posterior_0)

        # Compute the 1-st moment of y based on model knowledge and without noise
        self.obs_process_0 = torch.matmul(self.H, self.state_process_prior_0)

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior
        self.m1x_prior = torch.matmul(self.F, self.m1x_posterior)

        # Predict the 1-st moment of y
        self.m1y = torch.matmul(self.H, self.m1x_prior)    

    ##############################
    ### Kalman Gain Estimation ###      这步第二步调用, 被knetstep调用
    ##############################
    def step_KGain_est(self, y):  #传入KG

        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)       #这个方法查下, 这里传入gru做了norm

        # Feature 2: yt - y_t+1|t
        dm1y = y - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)               #这个方法查下, 这里传入gru做了norm

        '''
        # Feature 3: Kalman Gain t-1
        !上一步的kg
        ! 这里用 self.KGain, 需要reshape为一维, 底下拼起来, 可能还需要normalize
        !
        '''

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=0)          #这里可能需要拼KGain

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)

        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (self.m, self.n))

    #######################
    ### Kalman Net Step ###   这步最先被forward函数调用, 
    #######################
    def KNet_step(self, y):     #传入真y (spikes)

        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = y
        dy = y_obs - self.m1y

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###    这步最后被kgain est步调用
    ########################
    def KGain_step(self, KGainNet_in):      #FC+GRU+FC模块, 算出来ht

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in)
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, 0, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out_reshape)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out
    
    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
