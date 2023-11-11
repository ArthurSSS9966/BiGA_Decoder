import numpy as np
import matplotlib.pyplot as plt

from util import import_dataset, get_spikes_and_velocity, pre_process_spike, get_surrogate_data, plot_hand_trajectory
from EM import em_core

if __name__ == '__main__':
    train_dataset = import_dataset('Jenkins_train.nwb')

    train_spikes, train_velocity = get_spikes_and_velocity(train_dataset, resample_size=5, smooth=True)

    train_spikes, train_velocity = pre_process_spike(train_spikes, train_velocity, train_dataset,
                                                     window_step=5, overlap=True, window_size=15, smooth=False)

    # TODO: Examine the PSTHs

    # TODO: Plot PSTHs

    # Calculate EM #TODO: TEMPORARY CODE
    # 1) Generate surrogate data
    # 2) Run EM
    # 3) Predict latent states
    # 4) Predict hand velocity

    X, X_test, Y, Y_test = get_surrogate_data(train_spikes, train_velocity, trials=50)

    # Run EM
    mses = []

    EM_class = em_core(X, n_dim=60, n_iters=10)
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
    plt.plot(back_predict_con[:300, 0], label='Predicted')
    plt.plot(X_test_con[:300, 0], label='True')
    plt.title('Predicted vs True Spike Data, with dimension ' + str(EM_class.n_dim))
    plt.legend()
    plt.show()

    EM_class.fit(X, Y)

    # Predict the hand velocity
    hand_velocity = np.array([EM_class.predict_move(X_test[i]) for i in range(X_test.shape[0])])

    # Calculate RMSE
    rmse = np.sqrt(np.mean((X_test - back_predict) ** 2))/np.sqrt(np.var(X_test))
    print('NRMSE for spike data:', rmse)

    # Calculate RMSE for velocity
    rmse_vel = np.sqrt(np.mean((hand_velocity - Y_test) ** 2))/np.sqrt(np.var(Y_test))
    print('NRMSE for velocity:', rmse_vel)

    # Calculate MSE for a randomized shuffled trial version of hand_velocity
    rmse_vel_shuffled = np.sqrt(np.mean((np.random.permutation(hand_velocity) - Y_test) ** 2))/np.sqrt(np.var(Y_test))
    print('NRMSE for shuffled velocity:', rmse_vel_shuffled)

    # Combine all trials for hand_velocity and Y_test
    hand_velocity_con = np.array([hand_velocity[i] for i in range(len(hand_velocity) // 2)])
    hand_velocity_con = hand_velocity_con.reshape(-1, hand_velocity_con.shape[-1])
    Y_test_con = np.array([Y_test[i] for i in range(len(Y_test) // 2)])
    Y_test_con = Y_test_con.reshape(-1, Y_test_con.shape[-1])

    # Plot hand_velocity and True value
    plt.plot(hand_velocity_con[:, 0], label='Predicted')
    plt.plot(Y_test_con[:, 0], label='True')
    plt.title('Predicted vs True Hand Velocity, with dimension ' + str(EM_class.n_dim))
    plt.legend()
    plt.show()

    # TODO: Make the plot prettier
    fig, ax = plt.subplots()
    for i in range(50):
        plot = plot_hand_trajectory(Y_test[i], hand_velocity[i], ax)
    ax.set_title('Predicted vs True Hand Trajectory, with dimension ' + str(EM_class.n_dim))
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    ax.legend(['True', 'Predicted'])
    plt.show()

    # mses.append(mse)
    #
    # # Plot MSE vs dimension
    # plt.plot(np.arange(10, 100, 10), mses)
    # plt.title('MSE vs dimension')
    # plt.xlabel('Dimension')
    # plt.ylabel('MSE')
    # plt.show()
