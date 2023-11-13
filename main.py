import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import random
from tqdm import tqdm

from util import import_dataset, get_spikes_and_velocity, pre_process_spike, \
    get_surrogate_data, plot_hand_trajectory_conditions
from EM import em_core
from GRUnet import InitGRU, KalmanNetNN

if __name__ == '__main__':
    ##############################Parameter Initialization##############################################
    state_dimensions = 60
    # training params
    N_E = 200  # total samples
    N_Epochs = 20  # epochs
    train_split = 0.8  # train_cv_split
    train_samples = int(train_split * N_E)  # number of training samples
    N_B = train_samples  # batch size, default full batch
    loss_fn = nn.MSELoss()  # loss function
    N_CV = N_E - train_samples  # number of cv samples
    learningRate = 1e-5  # learning rate
    weightDecay = 1e-5  # regularizer, for optimizer

    ##############################Data Initialization##############################################


    train_dataset = import_dataset('Jenkins_train.nwb')

    train_spikes, train_velocity = get_spikes_and_velocity(train_dataset, resample_size=5, smooth=True)


    train_spikes, train_velocity = pre_process_spike(train_spikes, train_velocity, train_dataset,
                                                     window_step=5, overlap=True, window_size=15, smooth=False)

    trial_type = train_dataset.trial_info.set_index('trial_type').index.tolist()

    # TODO: Examine the PSTHs

    # TODO: Plot PSTHs

    # Calculate EM #TODO: TEMPORARY CODE
    # 1) Generate surrogate data
    # 2) Run EM
    # 3) Predict latent states
    # 4) Predict hand velocity

    X, X_test, Y, Y_test, X_label, X_test_label = get_surrogate_data(train_spikes,
                                                                     train_velocity,
                                                                     trial_type,
                                                                     trials=N_E,
                                                                     split=train_split)

    ##############################EM Initialization##############################################
    mses = []

    EM_class = em_core(X, n_dim=state_dimensions, n_iters=10)
    EM_class.get_parameters(plot=True)

    # TODO: Calculate Latent States for training data and plot to see grouping

    # Predict latent states
    latent_states = EM_class.cal_latent_states(X_test, current=False)

    # One step ahead prediction for spike data
    back_predict = np.array([EM_class.get_one_step_ahead_prediction(latent_states[i])
                             for i in range(len(latent_states))])

    # Combine all trials for back_predict and X_test
    back_predict_con = np.array([back_predict[i] for i in range(len(back_predict))])
    back_predict_con = back_predict_con.reshape(-1, back_predict_con.shape[-1])
    X_test_con = np.array([X_test[i] for i in range(len(X_test))])
    X_test_con = X_test_con.reshape(-1, X_test_con.shape[-1])

    # Plot back_predict and True value
    plt.figure()
    plt.plot(back_predict_con[:300, 0:5], label='Predicted', color='red')
    plt.plot(X_test_con[:300, 0:5], label='True', color='black')
    plt.title('Predicted vs True Spike Data, with dimension ' + str(EM_class.n_dim))
    plt.legend(['Predicted', 'True'])
    plt.show()

    EM_class.fit(X, Y)

    # Predict the hand velocity
    hand_velocity = np.array([EM_class.predict_move(X_test[i]) for i in range(X_test.shape[0])])
    train_hand_velocity = np.array([EM_class.predict_move(X[i]) for i in range(X.shape[0])])

    # Calculate RMSE
    rmse = np.sqrt(np.mean((X_test - back_predict) ** 2))/np.sqrt(np.var(X_test))
    print('NRMSE for spike data:', rmse)

    # Calculate NRMSE for training velocity
    rmse_train_vel = np.sqrt(np.mean((train_hand_velocity - Y) ** 2))/np.sqrt(np.var(Y))
    print('NRMSE for training velocity:', rmse_train_vel)

    # Calculate RMSE for velocity
    rmse_vel = np.sqrt(np.mean((hand_velocity - Y_test) ** 2))/np.sqrt(np.var(Y_test))
    print('NRMSE for test velocity:', rmse_vel)

    # Calculate MSE for a randomized shuffled trial version of hand_velocity
    rmse_vel_shuffled = np.sqrt(np.mean((np.random.permutation(hand_velocity) - Y_test) ** 2))/np.sqrt(np.var(Y_test))
    print('NRMSE for shuffled velocity:', rmse_vel_shuffled)

    # Combine all trials for hand_velocity and Y_test
    hand_velocity_con = np.array([hand_velocity[i] for i in range(len(hand_velocity) // 4)])
    hand_velocity_con = hand_velocity_con.reshape(-1, hand_velocity_con.shape[-1])
    Y_test_con = np.array([Y_test[i] for i in range(len(Y_test) // 4)])
    Y_test_con = Y_test_con.reshape(-1, Y_test_con.shape[-1])

    # Plot hand_velocity and True value
    plt.plot(hand_velocity_con[:, 0], label='Predicted')
    plt.plot(Y_test_con[:, 0], label='True')
    plt.title('Predicted vs True Hand Velocity, with dimension ' + str(EM_class.n_dim))
    plt.legend()
    plt.show()

    # TODO: Make the plot prettier
    plot_hand_trajectory_conditions(hand_velocity, Y_test, X_test_label, trial_number=4)

    # Plot training hand velocity and True value
    plot_hand_trajectory_conditions(train_hand_velocity, Y, X_label, trial_number=10)


    ##############################GRU Initialization##############################################

    # data input
    A = EM_class.A_values[-1]  # latest A
    C = EM_class.C_values[-1]  # latest C
    latent_states = EM_class.cal_latent_states(X, current=False)
    latent_states_test = EM_class.cal_latent_states(X_test, current=False)

    # initialize model
    m = state_dimensions  # states dimensions
    n = X.shape[2]  # spikes dimensions
    T = X.shape[1]  # time steps
    F = torch.tensor(A).float()  # A_values
    H = torch.tensor(C).float()  # C_values
    K0 = torch.tensor(np.random.rand(m, n)).float()  # KalmanGain at t0, not used for now

    initGRU = InitGRU(F, H, K0, m, n, T)

    # data
    y_training = torch.tensor(X).float()
    y_training_cv = torch.tensor(X_test).float()
    train_target = torch.tensor(latent_states).float()
    cv_target = torch.tensor(latent_states_test).float()
    latent_states_t0_train = torch.tensor(latent_states[:, 0, :]).float()
    latent_states_t0_cv = torch.tensor(latent_states_test[:, 0, :]).float()

    kn = KalmanNetNN()
    kn.Build(initGRU)

    optimizer = torch.optim.Adam(kn.parameters(), lr=learningRate, weight_decay=weightDecay)

    # save MSE
    MSE_cv_linear_batch = torch.empty([N_CV])
    MSE_cv_linear_epoch = torch.empty([N_Epochs])
    MSE_cv_dB_epoch = torch.empty([N_Epochs])
    MSE_train_linear_batch = torch.empty([N_B])
    MSE_train_linear_epoch = torch.empty([N_Epochs])
    MSE_train_dB_epoch = torch.empty([N_Epochs])

    x_out_cv = torch.empty(N_CV, initGRU.m, initGRU.T)

    for i in range(N_Epochs):
        # validation
        for c in tqdm(range(0, N_CV), desc='validation'):
            y_cv = y_training_cv[c, :, :]
            kn.InitSequence(latent_states_t0_cv[c])

            for t in range(0, initGRU.T):
                x_out_cv[c, :, t] = kn(y_cv[t, :])

            # Compute validation Loss
            MSE_cv_linear_batch[c] = loss_fn(x_out_cv[c], cv_target[c, :].T).item()
        # Average
        MSE_cv_linear_epoch[i] = torch.mean(MSE_cv_linear_batch)
        MSE_cv_dB_epoch[i] = 10 * torch.log10(MSE_cv_linear_epoch[i])
        print(i, "MSE cv :", MSE_cv_linear_epoch[i])

        # training
        kn.train()
        kn.init_hidden()
        Batch_Optimizing_LOSS_sum = 0
        spike_loss = torch.empty([train_samples, initGRU.T, initGRU.n])
        for j in tqdm(range(0, N_B), desc='training_samples'):
            if N_B == train_samples:
                n_e = j
            else:
                n_e = random.randint(0, N_E - 1)
            kn.InitSequence(latent_states_t0_train[n_e])

            x_out_training = torch.empty(initGRU.m, initGRU.T)

            for t in range(0, T):
                x_out_training[:, t] = kn(y_training[n_e, t, :])
                spike_loss[n_e, t, :] = kn.m1y
            # Compute Training Loss
            LOSS = loss_fn(x_out_training, train_target[n_e].T)
            MSE_train_linear_batch[n_e] = LOSS.item()

            Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS

        MSE_train_linear_epoch[i] = torch.mean(MSE_train_linear_batch)
        MSE_train_dB_epoch[i] = 10 * torch.log10(MSE_train_linear_epoch[i])

        optimizer.zero_grad()
        Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / N_B
        Batch_Optimizing_LOSS_mean.backward()
        optimizer.step()
        print(i, "MSE Training :", MSE_train_linear_epoch[i])





