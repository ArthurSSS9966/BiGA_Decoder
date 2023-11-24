import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

from util import import_dataset, get_spikes_and_velocity, pre_process_spike, \
    get_surrogate_data, plot_hand_trajectory_conditions, cal_R_square, \
    plot_latent_states, plot_latent_states_1d, plot_raw_data, encode_trial_type, noisy_bootstrapping, \
    noisy_bootstrapping_condition
from EM import em_core
from GRUcore import BiGRU

if __name__ == '__main__':
    ##############################Parameter Initialization##############################################
    state_dimensions = 60  # number of latent states
    N_E = 200  # total samples
    N_Epochs = 20  # epochs
    GRU_Epochs = 1600  # epochs
    GRU_hidden_dim = 60  # hidden dimension
    train_split = 0.8  # train_cv_split
    train_samples = int(train_split * N_E)  # number of training samples
    learningRate = 1e-2  # learning rate
    weightDecay = 0.1  # regularizer, for optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    concatenate = False  # concatenate latent states and spike data

    ##############################Data Initialization##############################################

    train_dataset = import_dataset('Jenkins_train.nwb')

    train_spikes, train_velocity = get_spikes_and_velocity(train_dataset, resample_size=5, smooth=True)

    train_spikes, train_velocity = pre_process_spike(train_spikes, train_velocity, train_dataset,
                                                     window_step=5, overlap=True, window_size=15,
                                                     smooth=False)

    trial_type = train_dataset.trial_info.set_index('trial_type').index.tolist()
    trial_var = train_dataset.trial_info.set_index('trial_version').index.tolist()

    trial_type = encode_trial_type(trial_type, trial_var)

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

    X,Y,X_label = noisy_bootstrapping_condition(X,Y,X_label, num_bootstrap_samples=3, noise_level=0.1)
    # X, Y = noisy_bootstrapping(X, Y, num_bootstrap_samples=5, noise_level=0.1, stack=False)

    X_test_con = np.array([X_test[i] for i in range(len(X_test))])
    X_test_con = X_test_con.reshape(-1, X_test_con.shape[-1])
    Y_test_con = np.array([Y_test[i] for i in range(len(Y_test))])
    Y_test_con = Y_test_con.reshape(-1, Y_test_con.shape[-1])

    plot_raw_data(X, X_test_label, con_num=6, neuron_num=5, seed=14, label='Train Raw Dataset')

    plot_raw_data(X_test, X_test_label, con_num=6, neuron_num=5, seed=14, label='Test Raw Dataset')
    plot_raw_data(Y_test, X_test_label, con_num=6, neuron_num=5, seed=14, label='Test Velocity Dataset')

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
    # plot_hand_velocity(baseline_predict, Y_test, trial_num=10)

    # Plot testing hand trajectory and True value
    plot_hand_trajectory_conditions(Y_test, baseline_predict, X_test_label, trial_number=4, seed=14)

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

    # Plot latent states
    plot_latent_states(latent_states[:, 1:, :], X_test_label, trial_num=4, seed=14)
    plot_latent_states_1d(latent_states[:, 1:, :], X_test_label, trial_num=4, seed=14)

    ##############################Method 1: LLS with EM Initialization##################################################

    EM_class.fit(X, Y)

    # One step ahead prediction for spike data
    back_predict_LLS = np.array([EM_class.predict_move(X_test[i])
                                 for i in range(len(X_test))])

    # Calculate R square for EM training
    EM_r2_train = cal_R_square(back_predict_LLS, Y_test)
    print('R square for EM training:', EM_r2_train)

    plot_raw_data(back_predict_LLS[:, 1:, :], X_test_label, con_num=6, neuron_num=5,
                  seed=14, label='Back Predicted Dataset using EM')
    plot_hand_trajectory_conditions(Y_test, back_predict_LLS, X_test_label, trial_number=4, seed=14)

    ##############################Method 2: KF preidctor Initialization##############################################

    EM_class.kalman_train(EM_class.cal_latent_states(X, current=False), Y)

    initial_velocity = np.mean(Y, axis=0)[0]
    back_pred_KF = np.array([EM_class.kalman_predict_all(X[i], initial_velocity) for i in range(len(latent_states))])

    KF_r2_train = cal_R_square(back_pred_KF, Y_test)
    print('R square for EM training:', KF_r2_train)

    plot_raw_data(back_pred_KF[:, 1:, :], X_test_label,
                  con_num=6, neuron_num=5, seed=14,
                  label='Back Predicted Dataset using Kalman Filter')
    plot_hand_trajectory_conditions(Y_test, back_pred_KF, X_test_label, trial_number=4, seed=14)

    # ##############################Save After_EM_data##############################################    
    np.savez('afterEM_dataset.npz', X=X, X_test=X_test, Y=Y, Y_test=Y_test)

    # ##############################GRU Initialization##############################################
    #
    # # data input
    X_train = EM_class.cal_latent_states(X, current=True)
    X_test_GRU = EM_class.cal_latent_states(X_test, current=True)
    gru = BiGRU(device=device)
    gru.to(device, non_blocking=True)
    gru.load_data(X_train, X_test_GRU, Y, Y_test)
    gru.Build(hiddendim=GRU_hidden_dim, learningRate=learningRate, weight_decay=weightDecay)
    train_mse, test_mse = gru.train_fit(GRU_Epochs)

    train_mse = train_mse.cpu().detach().numpy()
    test_mse = test_mse.cpu().detach().numpy()

    hand_velocity_gru_train, gru_hidden_train = gru.predict_velocity(gru.X_train)
    hand_velocity_gru, gru_hidden_test = gru.predict_velocity(gru.X_test)

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

    # Calculate NRMSE for a randomized shuffled trial version of hand_velocity
    rmse_vel_shuffled = np.sqrt(np.mean((np.random.permutation(hand_velocity_gru) - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for shuffled velocity:', rmse_vel_shuffled)

    # Plot hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity_gru, Y_test, X_test_label, trial_number=4, seed=14)

    # Plot training hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity_gru_train, Y, X_label, trial_number=10, seed=14)

    # ##############################Transformer Initialization##############################################
    #
    from TModel7 import TransformerModel

    # # data input
    # X_train = EM_class.cal_latent_states(X, current=True)
    Transformer_hidden_dim = 60
    Transformer_Epochs = 200
    X_test_Transformer = X_test_GRU
    Transformer = TransformerModel(device=device)
    Transformer.to(device, non_blocking=True)
    Transformer.load_data(X_train, X_test_Transformer, Y, Y_test)
    Transformer.Build(hiddendim=Transformer_hidden_dim, middle_dim=Transformer_hidden_dim, nhead=1, num_layers=2,
                      learningRate=0.001, weight_decay=0.0001)
    # Transformer.Build(hiddendim=Transformer_hidden_dim, nhead = 1,num_layers = 1, learningRate=0.001, weight_decay=0.0001)
    train_mse, test_mse = Transformer.train_fit(Transformer_Epochs)

    train_mse = train_mse.cpu().detach().numpy()
    test_mse = test_mse.cpu().detach().numpy()

    hand_velocity_Transformer_train = Transformer.predict_velocity(Transformer.X_train)
    hand_velocity_Transformer = Transformer.predict_velocity(Transformer.X_test)

    # Evaluate Transformer

    # Calculate NRMSE for Transformer training velocity
    rmse_train_vel_Transformer = np.sqrt(np.mean((hand_velocity_Transformer_train - Y) ** 2)) / np.sqrt(np.var(Y))
    print('NRMSE for Transformer training velocity:', rmse_train_vel_Transformer)

    # Calculate NRMSE for Transformer velocity
    rmse_vel_Transformer = np.sqrt(np.mean((hand_velocity_Transformer - Y_test) ** 2)) / np.sqrt(np.var(Y_test))
    print('NRMSE for Transformer test velocity:', rmse_vel_Transformer)

    # Calculate R square for Transformer training velocity
    r2_train_vel_Transformer = cal_R_square(hand_velocity_Transformer_train, Y)
    print('R square for Transformer training velocity:', r2_train_vel_Transformer)

    # Calculate R square for Transformer velocity
    r2_vel_Transformer = cal_R_square(hand_velocity_Transformer, Y_test)
    print('R square for Transformer testing velocity:', r2_vel_Transformer)

    # Calculate NRMSE for a randomized shuffled trial version of hand_velocity
    rmse_vel_shuffled = np.sqrt(np.mean((np.random.permutation(hand_velocity_Transformer) - Y_test) ** 2)) / np.sqrt(
        np.var(Y_test))
    print('NRMSE for shuffled velocity:', rmse_vel_shuffled)

    # Plot hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity_Transformer, Y_test, X_test_label, trial_number=4, seed=14)

    # Plot training hand trajectory and True value
    plot_hand_trajectory_conditions(hand_velocity_Transformer_train, Y, X_label, trial_number=10, seed=14)
