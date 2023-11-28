import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import itertools
from time import time

from util import import_dataset, get_spikes_and_velocity, pre_process_spike, \
    get_surrogate_data, plot_hand_trajectory_conditions, cal_R_square, \
    plot_latent_states, plot_latent_states_1d, plot_raw_data, encode_trial_type, noisy_bootstrapping, \
    noisy_bootstrapping_condition, save_EM_result, load_EM_result, save_model_result
from EM import em_core
from GRUcore import BiGRU


if __name__ == '__main__':
    ##############################Parameter Initialization##############################################
    state_dimensions = 20  # latent state dimension TODO: Loop this [20-100]
    N_E = 600  # number of trials TODO: Loop this [200-1000]
    noise_level = 0.1  # noise level TODO: Loop this [0.05-0.2]
    bootstrapping_sample = 3  # number of bootstrapping samples TODO: Loop this [1-5]

    GRU_Epochs = 1600  # epochs TODO: Fixed [1600]
    N_Epochs = 20  # epochs TODO: Fixed [20]
    train_split = 0.8  # train_cv_split TODO: Fixed [0.8]
    learningRate = 1e-2  # learning rate TODO: Fixed [1e-2]
    weightDecay = 0.1  # regularizer, for optimizer TODO: Fixed [0.1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # device TODO: Fixed [cuda:0]
    concatenate = False  # concatenate latent states and spike data TODO: Fixed [False]
    Transformer_Epochs = 200  # TODO: Fixed [200]
    seed = 12  # TODO: Fixed [12]
    EM_base_path = 'EM_result\\'

    GRU_hidden_dim = 60  # hidden dimension TODO: Loop this [10-100]
    Transformer_hidden_dim = 60  # TODO: Loop this [10-100]
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

    # Get the combination for N_E, noise_level, bootstrapping_sample
    N_E_combination = np.arange(300, 1000, 200)
    noise_level_combination = np.arange(0.05, 0.2, 0.05)
    bootstrapping_sample_combination = np.arange(0, 5, 1)
    state_dimensions_combination = np.arange(20, 100, 10)
    # state_dimensions_combination = [60]
    # N_E_combination = [600]
    # noise_level_combination = [0.1]
    # bootstrapping_sample_combination = [0]

    params_combination = itertools.product(N_E_combination, noise_level_combination, bootstrapping_sample_combination)

    # Major Loop that Needs to Run EM Starts here:
    for N_E, noise_level, bootstrapping_sample in params_combination:

        # Print the current combination
        print('Current Combination: N_E = {}, noise_level = {}, bootstrapping_sample = {}'
              .format(N_E, noise_level, bootstrapping_sample))

        X, X_test, Y, Y_test, X_label, X_test_label = get_surrogate_data(train_spikes,
                                                                         train_velocity,
                                                                         trial_type,
                                                                         trials=N_E,
                                                                         split=train_split,
                                                                         seed=seed)

        # X,Y,X_label = noisy_bootstrapping_condition(X,Y,X_label,
        #                                             num_bootstrap_samples=bootstrapping_sample,
        #                                             noise_level=noise_level)

        X, Y = noisy_bootstrapping(X, Y, num_bootstrap_samples=bootstrapping_sample,
                                   noise_level=noise_level, stack=False)

        X_test_con = np.array([X_test[i] for i in range(len(X_test))])
        X_test_con = X_test_con.reshape(-1, X_test_con.shape[-1])
        Y_test_con = np.array([Y_test[i] for i in range(len(Y_test))])
        Y_test_con = Y_test_con.reshape(-1, Y_test_con.shape[-1])

        # plot_raw_data(X, X_label, con_num=6, neuron_num=5, seed=seed, label='Train Raw Dataset')
        # plot_raw_data(Y, X_label, con_num=6, neuron_num=5, seed=seed, label='Train Velocity Dataset')
        # plot_raw_data(X_test, X_test_label, con_num=6, neuron_num=5, seed=seed, label='Test Raw Dataset')
        # plot_raw_data(Y_test, X_test_label, con_num=6, neuron_num=5, seed=seed, label='Test Velocity Dataset')

        # ##############################Baseline case for LLS##############################################
        #
        # # Calculate the baseline case for LLS
        # baseline_X = np.array([X[i] for i in range(len(X))])
        # baseline_X = baseline_X.reshape(-1, baseline_X.shape[-1])
        # baseline_Y = np.array([Y[i] for i in range(len(Y))])
        # baseline_Y = baseline_Y.reshape(-1, baseline_Y.shape[-1])
        #
        # alpha = np.logspace(-4, 4, 5)
        # baseline_model = GridSearchCV(Ridge(), {'alpha': alpha})
        # baseline_model.fit(baseline_X, baseline_Y)
        # baseline_predict = np.array([baseline_model.predict(X_test[i]) for i in range(len(X_test))])
        # baseline_predict_con = baseline_predict.reshape(-1, baseline_predict.shape[-1])

        # Plot testing hand trajectory and True value
        # plot_hand_trajectory_conditions(Y_test, baseline_predict, X_test_label,
        #                                 trial_number=4, seed=seed, label='Baseline Test ')
        #
        # # Calculate NRMSE for baseline
        # baseline_rmse = np.sqrt(np.mean((Y_test - baseline_predict) ** 2)) / np.sqrt(np.var(Y_test))
        # print('NRMSE for baseline:', baseline_rmse)
        #
        # # Calculate R square for baseline training
        # baseline_r2_train = baseline_model.score(baseline_X, baseline_Y)
        # print('R square for baseline training:', baseline_r2_train)
        #
        # # Calculate R square for baseline testing
        # baseline_r2 = baseline_model.score(X_test_con, Y_test_con)
        # print('R square for baseline testing:', baseline_r2)

        ##############################EM Initialization##################################################

        # Evaluate EM using time cost and MSE
        MSE_EM = []
        Time_EM = []

        for state_dimensions in state_dimensions_combination:

            start_time = time()

            EM_class = em_core(X, n_dim=state_dimensions)
            EM_class.get_parameters(plot=False, n_iters=N_Epochs)

            # TODO: Calculate Latent States for training data and plot to see grouping

            # Predict latent states
            latent_states_for_back_predict = EM_class.cal_latent_states(X_test, current=False)
            spike_predicted = np.array([EM_class.get_one_step_ahead_prediction(latent_states_for_back_predict[i])
                                        for i in range(len(latent_states_for_back_predict))])

            # Calcualte NRMSE
            EM_rmse = np.sqrt(np.mean((X_test - spike_predicted) ** 2)) / np.sqrt(np.var(X_test))

            # Plot latent states
            # plot_latent_states(latent_states[:, 1:, :], X_test_label, trial_num=4, seed=seed)
            # plot_latent_states_1d(latent_states[:, 1:, :], X_test_label, trial_num=4, seed=seed)
            Time_EM.append(time() - start_time)
            MSE_EM.append(EM_rmse)
            ##############################Method 1: LLS with EM Initialization#################################

            # EM_class.fit(X, Y)

            # Calculate NRMSE for EM training


            # # One step ahead prediction for spike data
            # back_predict_LLS = np.array([EM_class.predict_move(X_test[i])
            #                              for i in range(len(X_test))])

            # # Calculate R square for EM training
            # EM_r2_train = cal_R_square(back_predict_LLS, Y_test)
            # print('R square for EM training:', EM_r2_train)
            #
            # plot_raw_data(back_predict_LLS[:, 1:, :], X_test_label, con_num=6, neuron_num=5,
            #               seed=seed, label='Back Predicted Dataset using EM')
            # plot_hand_trajectory_conditions(Y_test, back_predict_LLS, X_test_label, trial_number=4, seed=seed, label='EM Test')

            # ##############################Method 2: KF preidctor Initialization##############################################
            # EM_class.kalman_train(EM_class.cal_latent_states(X, current=False), Y)
            #
            # initial_velocity = np.mean(Y, axis=0)[0]
            # back_pred_KF = np.array([EM_class.kalman_predict_all(X[i], initial_velocity) for i in range(len(latent_states))])
            #
            # KF_r2_train = cal_R_square(back_pred_KF, Y_test)
            # print('R square for EM training:', KF_r2_train)
            #
            # plot_raw_data(back_pred_KF[:, 1:, :], X_test_label,
            #               con_num=6, neuron_num=5, seed=seed,
            #               label='Back Predicted Dataset using Kalman Filter')
            # plot_hand_trajectory_conditions(Y_test, back_pred_KF, X_test_label, trial_number=4, seed=seed, label='KF Test')

            # ##############################Save After_EM_data##############################################
            params = {'state_dimensions': state_dimensions, 'N_E': N_E, 'noise_level': noise_level,
                      'bootstrapping_sample': bootstrapping_sample}
            save_EM_result(params, EM_base_path, X, X_test, Y, Y_test, X_label, X_test_label, EM_class)



        # # Plot the time cost and MSE
        # plt.figure()
        # plt.plot(state_dimensions_combination, MSE_EM, 'b', label='MSE')
        # # Set two y-axis
        # ax1 = plt.gca()
        # ax2 = ax1.twinx()
        # ax2.plot(state_dimensions_combination, Time_EM, 'r', label='Time Cost')
        # ax1.set_xlabel('State Dimensions')
        # ax1.set_ylabel('MSE')
        # ax2.set_ylabel('Time Cost')
        # ax1.legend(loc='upper left')
        # ax2.legend(loc='upper right')
        # plt.title('MSE and Time Cost for EM')
        # plt.show()

    # ##############################Load After_EM_data##############################################
    ML_model_parameter_comb = itertools.product(state_dimensions_combination, N_E_combination,
                                                noise_level_combination, bootstrapping_sample_combination)
    # GRU_hidden_dim_comb = np.arange(10, 100, 10)
    # Transformer_hidden_dim_comb = np.arange(10, 100, 10)
    GRU_hidden_dim_comb = [60]
    Transformer_hidden_dim_comb = [60]
    GRU_base_path = 'GRU_result\\'
    Transformer_base_path = 'Transformer_result\\'

    for state_dimensions, N_E, noise_level, bootstrapping_sample in ML_model_parameter_comb:

        params_load = {'state_dimensions': state_dimensions, 'N_E': N_E, 'noise_level': noise_level,
                         'bootstrapping_sample': bootstrapping_sample}

        npzfile = load_EM_result(EM_base_path, params_load)
        X = npzfile['X']
        X_test = npzfile['Y']
        Y = npzfile['X_test']
        Y_test = npzfile['Y_test']
        X_label = npzfile['X_label']
        X_test_label = npzfile['X_test_label']
        EM_class = npzfile['EM_model']
        # ##############################GRU Initialization##############################################
        # # data input

        for GRU_hidden_dim in GRU_hidden_dim_comb:

            GRU_params_load = {'state_dimensions': state_dimensions, 'N_E': N_E, 'noise_level': noise_level,
                                 'bootstrapping_sample': bootstrapping_sample, 'GRU_hidden_dim': GRU_hidden_dim}

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

            save_model_result(GRU_params_load, GRU_base_path,
                              gru, EM_class,
                              X_test, Y_test,
                              X_test_label)

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

        # # Plot hand trajectory and True value
        # plot_hand_trajectory_conditions(hand_velocity_gru, Y_test, X_test_label, trial_number=4, seed=seed, label='GRU Test')
        #
        # # Plot training hand trajectory and True value
        # plot_hand_trajectory_conditions(hand_velocity_gru_train, Y, X_label, trial_number=10, seed=seed, label='GRU Train')

        # ##############################Transformer Initialization##############################################
        #
        from TModel7 import TransformerModel

        for Transformer_hidden_dim in Transformer_hidden_dim_comb:

            Transformer_params_load = {'state_dimensions': state_dimensions, 'N_E': N_E, 'noise_level': noise_level,
                                 'bootstrapping_sample': bootstrapping_sample, 'Transformer_hidden_dim': Transformer_hidden_dim}

            # # data input
            # X_train = EM_class.cal_latent_states(X, current=True)
            X_train = EM_class.cal_latent_states(X, current=True)
            X_test_Transformer = EM_class.cal_latent_states(X_test, current=True)
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

            save_model_result(Transformer_params_load, Transformer_base_path,
                              Transformer, EM_class,
                              X_test, Y_test,
                              X_test_label)

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

        # # Plot hand trajectory and True value
        # plot_hand_trajectory_conditions(hand_velocity_Transformer, Y_test, X_test_label,
        #                                 trial_number=4, seed=seed,label='Transformer Test')
        #
        # # Plot training hand trajectory and True value
        # plot_hand_trajectory_conditions(hand_velocity_Transformer_train, Y, X_label,
        #                                     trial_number=10, seed=seed, label='Transformer Train')
