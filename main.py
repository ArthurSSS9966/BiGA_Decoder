import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch
import random
from tqdm import tqdm
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from util import import_dataset, get_spikes_and_velocity, pre_process_spike, \
    get_surrogate_data, plot_hand_trajectory_conditions, plot_hand_velocity, convert_angle_mag_to_velocity
from EM import em_core
from GRUnet import InitGRU, KalmanNetNN

if __name__ == '__main__':
    ##############################Parameter Initialization##############################################
    state_dimensions = 60
    # training params
    N_E = 600  # total samples
    N_Epochs = 10  # epochs
    train_split = 0.8  # train_cv_split
    train_samples = int(train_split * N_E)  # number of training samples
    N_B = train_samples  # batch size, default full batch
    loss_fn = nn.MSELoss()  # loss function
    N_CV = N_E - train_samples  # number of cv samples
    learningRate = 1e-6  # learning rate
    weightDecay = 1e-5  # regularizer, for optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    X_test_con = np.array([X_test[i] for i in range(len(X_test))])
    X_test_con = X_test_con.reshape(-1, X_test_con.shape[-1])
    Y_test_con = np.array([Y_test[i] for i in range(len(Y_test))])
    Y_test_con = Y_test_con.reshape(-1, Y_test_con.shape[-1])

    # Y_test = np.array([convert_angle_mag_to_velocity(Y_test[i, :, 0], Y_test[i, :, 1])
    #                    for i in range(len(Y_test))])

    ##############################Baseline case for LLS##############################################
    # Calculate the baseline case for LLS
    baseline_X = np.array([X[i] for i in range(len(X))])
    baseline_X = baseline_X.reshape(-1, baseline_X.shape[-1])
    baseline_Y = np.array([Y[i] for i in range(len(Y))])
    baseline_Y = baseline_Y.reshape(-1, baseline_Y.shape[-1])

    alpha = np.logspace(-4, -1, 3)
    baseline_model = GridSearchCV(Ridge(), {'alpha': alpha})
    baseline_model.fit(baseline_X, baseline_Y)
    baseline_predict = np.array([baseline_model.predict(X_test[i]) for i in range(len(X_test))])

    # baseline_predict = np.array([convert_angle_mag_to_velocity(baseline_predict[i, :, 0], baseline_predict[i, :, 1])
    #                              for i in range(len(baseline_predict))])

    # Plot hand velocity
    plot_hand_velocity(baseline_predict, Y_test, trial_num=10)

    # Plot testing hand trajectory and True value
    plot_hand_trajectory_conditions(baseline_predict, Y_test, X_test_label, trial_number=4)

    # Calculate NRMSE for baseline
    baseline_rmse = np.sqrt(np.mean((Y_test - baseline_predict) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for baseline:', baseline_rmse)

    # Calculate R square for baseline training
    baseline_r2_train = baseline_model.best_score_
    print('R square for baseline training:', baseline_r2_train)

    # Calculate R square for baseline testing
    baseline_r2 = baseline_model.score(X_test_con, Y_test_con)
    print('R square for baseline testing:', baseline_r2)

    ##############################EM Initialization##################################################
    EM_class = em_core(X, n_dim=state_dimensions)
    EM_class.get_parameters(plot=True, n_iters=N_Epochs)

    # TODO: Calculate Latent States for training data and plot to see grouping

    # Predict latent states
    latent_states = EM_class.cal_latent_states(X_test, current=False)

    # One step ahead prediction for spike data
    back_predict = np.array([EM_class.get_one_step_ahead_prediction(latent_states[i])
                             for i in range(len(latent_states))])

    # Combine all trials for back_predict and X_test
    back_predict_con = np.array([back_predict[i] for i in range(len(back_predict))])
    back_predict_con = back_predict_con.reshape(-1, back_predict_con.shape[-1])

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

    # Convert angle and magnitude to velocity

    # hand_velocity = np.array([convert_angle_mag_to_velocity(hand_velocity[i, :, 0],
    #                                                         hand_velocity[i, :, 1])
    #                           for i in range(hand_velocity.shape[0])])
    #
    # train_hand_velocity = np.array([convert_angle_mag_to_velocity(train_hand_velocity[i, :, 0],
    #                                                               train_hand_velocity[i, :, 1])
    #                                 for i in range(train_hand_velocity.shape[0])])
    #
    # Y = np.array([convert_angle_mag_to_velocity(Y[i, :, 0],
    #                                             Y[i, :, 1])
    #               for i in range(Y.shape[0])])

    # Calculate NRMSE
    rmse = np.sqrt(np.mean((X_test - back_predict) ** 2)) / np.sqrt(np.var(X_test))
    print('NRMSE for spike data:', rmse)

    # Calculate NRMSE for training velocity
    rmse_train_vel = np.sqrt(np.mean((train_hand_velocity - Y) ** 2)) / np.sqrt(np.var(Y))
    print('NRMSE for training velocity:', rmse_train_vel)

    # Calculate R square for training velocity
    r2_train_vel = EM_class.model.best_score_
    print('R square for training velocity:', r2_train_vel)

    # Calculate NRMSE for velocity
    rmse_vel = np.sqrt(np.mean((hand_velocity - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for test velocity:', rmse_vel)

    # Calculate R square for velocity
    r2_vel = EM_class.model.score(X_test_con, Y_test_con)
    print('R square for testing velocity:', r2_vel)

    # Calculate NRMSE for a randomized shuffled trial version of hand_velocity
    rmse_vel_shuffled = np.sqrt(np.mean((np.random.permutation(hand_velocity) - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for shuffled velocity:', rmse_vel_shuffled)

    # Plot hand velocity
    plot_hand_velocity(hand_velocity, Y_test, trial_num=10)

    # Plot testing hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity, Y_test, X_test_label, trial_number=4)

    # Plot training hand trajectory and True value
    plot_hand_trajectory_conditions(train_hand_velocity, Y, X_label, trial_number=10)

    # ##############################GRU Initialization##############################################
    #
    # # data input
    # A = EM_class.A_values[-1]  # latest A
    # C = EM_class.C_values[-1]  # latest C
    # latent_states = EM_class.cal_latent_states(X, current=False)
    # latent_states_test = EM_class.cal_latent_states(X_test, current=False)
    #
    # # initialize model
    # m = state_dimensions  # states dimensions
    # n = X.shape[2]  # spikes dimensions
    # T = X.shape[1]  # time steps
    # F = torch.tensor(A).float()  # A_values
    # H = torch.tensor(C).float()  # C_values
    # K0 = torch.tensor(np.random.rand(m, n)).float()  # KalmanGain at t0, not used for now
    #
    # initGRU = InitGRU(F, H, K0, m, n, T)
    #
    # # data
    # y_training = torch.tensor(X).float()
    # y_training_cv = torch.tensor(X_test).float()
    # train_target = torch.tensor(latent_states).float()
    # cv_target = torch.tensor(latent_states_test).float()
    # latent_states_t0_train = torch.tensor(latent_states[:, 0, :]).float()
    # latent_states_t0_cv = torch.tensor(latent_states_test[:, 0, :]).float()
    #
    # kn = KalmanNetNN()
    # kn.Build(initGRU)
    #
    # optimizer = torch.optim.Adam(kn.parameters(), lr=learningRate, weight_decay=weightDecay)
    #
    # # save MSE
    # MSE_cv_linear_batch = torch.empty([N_CV])
    # MSE_cv_linear_epoch = torch.empty([N_Epochs])
    # MSE_cv_dB_epoch = torch.empty([N_Epochs])
    # MSE_train_linear_batch = torch.empty([N_B])
    # MSE_train_linear_epoch = torch.empty([N_Epochs])
    # MSE_train_dB_epoch = torch.empty([N_Epochs])
    #
    # x_out_cv = torch.empty(N_CV, initGRU.m, initGRU.T)
    #
    # for i in tqdm(range(N_Epochs), desc='Epochs: '):
    #     # validation
    #     for c in range(0, N_CV):
    #         y_cv = y_training_cv[c, :, :]
    #         kn.InitSequence(latent_states_t0_cv[c])
    #
    #         for t in range(0, initGRU.T):
    #             x_out_cv[c, :, t] = kn(y_cv[t, :])
    #
    #         # Compute validation Loss
    #         MSE_cv_linear_batch[c] = loss_fn(x_out_cv[c], cv_target[c, :].T).item()
    #
    #     # Average
    #     MSE_cv_linear_epoch[i] = torch.mean(MSE_cv_linear_batch)
    #     MSE_cv_dB_epoch[i] = 10 * torch.log10(MSE_cv_linear_epoch[i])
    #     print(i, "MSE cv :", MSE_cv_linear_epoch[i])
    #
    #     # training
    #     kn.train()
    #     kn.init_hidden()
    #     Batch_Optimizing_LOSS_sum = 0
    #     spike_loss = torch.empty([train_samples, initGRU.T, initGRU.n])
    #     for j in range(0, N_B):
    #         if N_B == train_samples:
    #             n_e = j
    #         else:
    #             n_e = random.randint(0, N_E - 1)
    #         kn.InitSequence(latent_states_t0_train[n_e])
    #
    #         x_out_training = torch.empty(initGRU.m, initGRU.T)
    #
    #         for t in range(0, T):
    #             x_out_training[:, t] = kn(y_training[n_e, t, :])
    #             spike_loss[n_e, t, :] = kn.m1y
    #         # Compute Training Loss
    #         LOSS = loss_fn(x_out_training, train_target[n_e].T)
    #         MSE_train_linear_batch[n_e] = LOSS.item()
    #
    #         Batch_Optimizing_LOSS_sum = Batch_Optimizing_LOSS_sum + LOSS
    #
    #     MSE_train_linear_epoch[i] = torch.mean(MSE_train_linear_batch)
    #     MSE_train_dB_epoch[i] = 10 * torch.log10(MSE_train_linear_epoch[i])
    #
    #     optimizer.zero_grad()
    #     Batch_Optimizing_LOSS_mean = Batch_Optimizing_LOSS_sum / N_B
    #     Batch_Optimizing_LOSS_mean.backward()
    #     optimizer.step()
    #     print(i, "MSE Training :", MSE_train_linear_epoch[i])
    #
    # # NRMSE for training spikes : lastest epoch
    # nrmse_training = np.sqrt(np.mean((spike_loss.detach() - y_training).numpy() ** 2))
    # print('NRMSE for training spikes:', nrmse_training)
