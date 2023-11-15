import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from util import import_dataset, get_spikes_and_velocity, pre_process_spike, \
    get_surrogate_data, plot_hand_trajectory_conditions, plot_hand_velocity, cal_R_square
from EM import em_core
from GRUcore import BiGRU

if __name__ == '__main__':
    ##############################Parameter Initialization##############################################
    state_dimensions = 60
    # training params
    N_E = 600  # total samples
    N_Epochs = 10  # epochs
    GRU_Epochs = 600  # epochs
    train_split = 0.8  # train_cv_split
    train_samples = int(train_split * N_E)  # number of training samples
    learningRate = 1e-2  # learning rate
    weightDecay = 1  # regularizer, for optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    concatenate = False  # concatenate latent states and spike data

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

    alpha = np.logspace(-4, 4, 5)
    baseline_model = GridSearchCV(Ridge(), {'alpha': alpha})
    baseline_model.fit(baseline_X, baseline_Y)
    baseline_predict = np.array([baseline_model.predict(X_test[i]) for i in range(len(X_test))])
    baseline_predict_con = baseline_predict.reshape(-1, baseline_predict.shape[-1])

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
    baseline_r2_train = baseline_model.score(baseline_X, baseline_Y)
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

    EM_class.fit(X, Y, concatenate=concatenate)

    # Predict the hand velocity
    hand_velocity = np.array([EM_class.predict_move(X_test[i], concatenate) for i in range(X_test.shape[0])])
    train_hand_velocity = np.array([EM_class.predict_move(X[i], concatenate) for i in range(X.shape[0])])

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
    # Plot back_predict and True value

    plt.figure()
    plt.plot(back_predict_con[:300, 0:5], label='Predicted', color='red')
    plt.plot(X_test_con[:300, 0:5], label='True', color='black')
    plt.title('Predicted vs True Spike Data, with dimension ' + str(EM_class.n_dim))
    plt.legend(['Predicted', 'True'])
    plt.show()

    # Calculate NRMSE
    rmse = np.sqrt(np.mean((X_test - back_predict) ** 2)) / np.sqrt(np.var(X_test))
    print('NRMSE for spike data:', rmse)

    # Calculate NRMSE for training velocity
    rmse_train_vel = np.sqrt(np.mean((train_hand_velocity - Y) ** 2)) / np.sqrt(np.var(Y))
    print('NRMSE for training velocity:', rmse_train_vel)

    # Calculate R square for training velocity
    r2_train_vel = EM_class.cal_R_square(X, Y, concatenate)
    print('R square for training velocity:', r2_train_vel)

    # Calculate NRMSE for velocity
    rmse_vel = np.sqrt(np.mean((hand_velocity - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for test velocity:', rmse_vel)

    # Calculate R square for velocity
    r2_vel = EM_class.cal_R_square(X_test, Y_test, concatenate)
    print('R square for testing velocity:', r2_vel)

    # Calculate NRMSE for a randomized shuffled trial version of hand_velocity
    rmse_vel_shuffled = np.sqrt(np.mean((np.random.permutation(hand_velocity) - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for shuffled velocity:', rmse_vel_shuffled)
    # ##############################GRU Initialization##############################################
    #
    # # data input
    X_train = EM_class.cal_latent_states(X, current=True)
    X_test_GRU = EM_class.cal_latent_states(X_test, current=True)
    gru = BiGRU(device=device)
    gru.to(device, non_blocking=True)
    gru.load_data(X_train, X_test_GRU, Y, Y_test)
    gru.Build(hiddendim=100, learningRate=learningRate, weight_decay=weightDecay)
    train_mse, test_mse = gru.train_fit(GRU_Epochs)

    train_mse = train_mse.cpu().detach().numpy()
    test_mse = test_mse.cpu().detach().numpy()

    hand_velocity_gru_train = gru.predict_velocity(gru.X_train)
    hand_velocity_gru = gru.predict_velocity(gru.X_test)

    # Evaluate GRU

    # Calculate NRMSE for GRU training velocity
    rmse_train_vel_gru = np.sqrt(np.mean((hand_velocity_gru_train - Y) ** 2)) / np.sqrt(np.var(Y))
    print('NRMSE for GRU training velocity:', rmse_train_vel_gru)

    # Calculate NRMSE for GRU velocity
    rmse_vel_gru = np.sqrt(np.mean((hand_velocity_gru - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for GRU test velocity:', rmse_vel_gru)

    # Calculate R square for GRU training velocity
    r2_train_vel_gru = cal_R_square(hand_velocity_gru_train, Y)
    print('R square for GRU training velocity:', r2_train_vel_gru)

    # Calculate R square for GRU velocity
    r2_vel_gru = cal_R_square(hand_velocity_gru, Y_test)
    print('R square for GRU testing velocity:', r2_vel_gru)

    # Plot hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity_gru, Y_test, X_test_label, trial_number=4, seed=2024)

    # Plot training hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity_gru_train, Y, X_label, trial_number=10)