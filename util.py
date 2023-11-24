from nlb_tools.nwb_interface import NWBDataset
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm


def import_dataset(filepath, filetype='nwb'):
    '''
    Import dataset from NWB file
    :param filepath:
    :param filetype:
    :return:
    '''
    start = time.time()
    if filetype == 'nwb':
        dataset = NWBDataset(filepath, "*train", split_heldout=False)
    else:
        raise NotImplementedError
    print(f'Importing dataset took {time.time() - start} seconds')
    return dataset


def encode_trial_type(trial_type, trial_version):
    """
    Encodes trial type and trial version into a unique trial type identifier.

    :param trial_type: List or array of trial types.
    :param trial_version: List or array of trial versions.
    :return: List of new trial type identifiers.
    """
    if len(trial_type) != len(trial_version):
        raise ValueError("trial_type and trial_version must be of the same length")

    # Find the maximum trial type value to create a unique encoding
    max_trial_type = max(trial_type)

    # Encode trial type and version into a single value
    # This assumes that trial_version is relatively small compared to max_trial_type
    encoded_trial_type = [t_type + (t_version * (max_trial_type + 1)) for t_type, t_version in zip(trial_type, trial_version)]

    return encoded_trial_type


def get_spikes_and_velocity(dataset, resample_size=1, smooth=False):
    '''
    Extract spikes and velocity from dataset
    :param dataset:
    :param resample_size:
    :return:
    '''

    if smooth:
        assert resample_size >= 3, 'Resample size must be greater than 3 for smoothing'
        dataset.resample(resample_size)
        dataset.smooth_spk(50, name='smth_spk')
    else:
        dataset.resample(resample_size)

    # Extract neural data and lagged hand velocity
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-70, 430))  # Default -130,370
    lagged_trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-20, 480))

    # Extract spikes and velocity
    if smooth:
        spikes = _seg_data_by_trial(trial_data, data_type='spikes_smth_spk')
    else:
        spikes = _seg_data_by_trial(trial_data, data_type='spikes')
    vel = _seg_data_by_trial(lagged_trial_data, data_type='hand_vel')

    # pos = _seg_data_by_trial(lagged_trial_data, data_type='hand_pos')
    #
    # # Combine vel and pos
    # vel = [np.concatenate((vel[i], pos[i]), axis=1) for i in range(len(vel))]

    # # Convert velocity to angle and magnitude
    # vel = [calculate_angles_and_velocities(vel[i]) for i in range(len(vel))]

    return spikes, vel


def calculate_angles_and_velocities(velocities):
    """
    Calculate the angles (in degrees) and magnitudes of velocity vectors in a 2D array.

    :param velocities: A 2D numpy array where each row is a time point and columns are [x_velocity, y_velocity]
    :return: A 2D numpy array where each row is [angle, magnitude] for each time point
    """
    x_velocities = velocities[:, 0]
    y_velocities = velocities[:, 1]

    angles_rad = np.arctan2(y_velocities, x_velocities)  # Angles in radians
    angles_deg = np.degrees(angles_rad)  # Convert to degrees

    magnitudes = np.sqrt(x_velocities ** 2 + y_velocities ** 2)  # Magnitudes of the velocities

    return np.column_stack((angles_deg, magnitudes))


def pre_process_spike(spikes, vel, dataset, window_step=50, overlap=False, smooth=True, demean=False, **kwargs):
    '''

    :param spikes:
    :param vel:
    :param dataset:
    :param window_step:
    :param overlap:
    :param smooth:
    :param kwargs:
    :return:
    '''

    # Create a list the same size as spikes
    down_spikes = [None] * len(spikes)
    down_vel = [None] * len(spikes)
    rate = [None] * len(spikes)

    for i in range(len(spikes)):

        down_spikes[i] = _downsample_data(spikes[i], downsample_rate=window_step, overlap=overlap,
                                          **kwargs)  # Smooth with 250ms moving window
        # Downsample velocity the same rate as spikes
        bin_num = down_spikes[i].shape[0]
        down_ind = np.linspace(0, vel[i].shape[0] - 1, bin_num).astype(int)
        down_vel[i] = vel[i][down_ind, :]

        # Convert spikes from Poisson to Gaussian
        down_spikes[i] = np.power(down_spikes[i], 3 / 4)  # TODO: change to 3/4

        if smooth:
            # Apply Gaussian smoothing to spikes
            down_spikes[i] = apply_gaussian_smoothing_2d(down_spikes[i],
                                                         window_length=window_step // 5 * 2,
                                                         window_step=window_step // 5)

        if not overlap:
            rate[i] = down_spikes[i] / dataset.bin_width * 1000 / window_step  # Convert to Hz
        else:
            rate[i] = down_spikes[i] / dataset.bin_width * 1000 / kwargs.get('window_size',
                                                                             window_step)  # Convert to Hz


    return rate, down_vel


def _seg_data_by_trial(df, data_type='spikes'):
    '''
    Segment data by trial
    :param df:
    :param data_type:
    :return:
    '''
    # Assuming `df` is your DataFrame and `timestamp` is your column with timestamps as strings
    # First, convert the timestamp strings to timedeltas
    df['align_time'] = pd.to_timedelta(df['align_time'])

    # Find where the new segment starts by finding the rows where the timestamp
    # is less than the previous timestamp (which means it wrapped around)
    segment_starts = df['align_time'] < df['align_time'].shift(1)

    # Create a group ID that increments each time a new segment starts
    df['group_id'] = segment_starts.cumsum()

    # Initialize an empty list to hold each segment's data
    segmented_data = []

    # Group by the 'group_id' and extract each segment
    for _, group_df in df.groupby('group_id'):
        # Drop the 'group_id' column as it's no longer needed
        group_df = group_df.drop(columns=['group_id', 'align_time'])  # Drop 'timestamp' if it's no longer needed
        # Convert the DataFrame to a 2D array (if needed) and append to the list
        segmented_data.append(group_df[[data_type]].to_numpy())

    # Now `segmented_data` is a list containing the data for each segment
    return segmented_data


def _downsample_data(data, downsample_rate=5, overlap=False, **kwargs):
    """
    Downsample data by summing over a window of bins
    :param data:
    :param downsample_rate:
    :param overlap:
    :param kwargs:
    :return:
    """
    if not overlap:
        # Down sample by calculating the sum firing rate over a window of every downsample_rate bins
        re_sampled_data = data.reshape(-1, downsample_rate, data.shape[1]).sum(axis=1)
    else:
        window_size = kwargs.get('window_size', downsample_rate)
        n_rows, n_cols = data.shape
        n_windows = 1 + (n_rows - window_size) // downsample_rate
        summed_windows = np.zeros((n_windows, n_cols))

        for i in range(n_windows):
            start_idx = i * downsample_rate
            end_idx = start_idx + window_size
            summed_windows[i, :] = data[start_idx:end_idx, :].sum(axis=0)

        re_sampled_data = summed_windows

    return re_sampled_data


def get_surrogate_data(train_spikes, train_velocity, trial_type, trials=50, split=0.8, seed=2023):
    '''
    :param train_spikes: Spike data for each trial.
    :param train_velocity: Velocity data for each trial.
    :param trial_type: Type of each trial.
    :param trials: Total number of trials to select.
    :return: Surrogate data for training and testing.
    '''
    # Set seed
    np.random.seed(seed)
    n_trial_types = np.random.choice(len(np.unique(trial_type)), min(len(np.unique(trial_type)), trials // 10),
                                     replace=False)

    # Initialize lists to store indices
    train_data_idx = []
    test_data_idx = []

    # Calculate the number of trials per condition for train and test
    trials_per_condition_train = int(np.floor(split * trials / len(n_trial_types)))
    trials_per_condition_test = int(np.ceil((1 - split) * trials / len(n_trial_types)))

    # Select indices for train and test data
    for i in n_trial_types:
        idx = np.where(np.array(trial_type) == trial_type[i])[0]
        np.random.shuffle(idx)
        train_data_idx.extend(idx[:trials_per_condition_train])
        test_data_idx.extend(idx[trials_per_condition_train:trials_per_condition_train + trials_per_condition_test])

    # Extracting training and testing data
    surrogate_data_train = np.array([train_spikes[i] for i in train_data_idx])
    surrogate_data_test = np.array([train_spikes[i] for i in test_data_idx])
    surrogate_data_movement_train = np.array([train_velocity[i] for i in train_data_idx])
    surrogate_data_movement_test = np.array([train_velocity[i] for i in test_data_idx])

    # Examine if any column is all 0
    surrogate_data_comb = surrogate_data_train.reshape(-1, surrogate_data_train.shape[-1])
    print('Number of all 0 columns:', np.sum(np.sum(surrogate_data_comb, axis=0) == 0))
    all_0_col_idx = np.where(np.sum(surrogate_data_comb, axis=0) == 0)[0]

    # Delete all 0 columns
    surrogate_data_train = np.delete(surrogate_data_train, all_0_col_idx, axis=2)
    surrogate_data_test = np.delete(surrogate_data_test, all_0_col_idx, axis=2)

    trial_type_train = [trial_type[i] for i in train_data_idx]
    trial_type_test = [trial_type[i] for i in test_data_idx]

    return surrogate_data_train, surrogate_data_test, surrogate_data_movement_train, surrogate_data_movement_test, trial_type_train, trial_type_test


def apply_gaussian_smoothing_2d(data, window_length, window_step):
    """
    Apply Gaussian smoothing to 2D time series data.

    Parameters:
    data (2D array-like): The input time series data where rows are time steps and columns are neurons.
    window_length (int): The length of the Gaussian smoothing window.
    step_size (float): The standard deviation for Gaussian kernel, related to the window length.

    Returns:
    smoothed_data (2D array-like): The smoothed time series data.
    """

    # Calculate the sigma for the Gaussian kernel based on window length and step size
    sigma = window_length / (2 * window_step)

    # Initialize an array to hold the smoothed data
    smoothed_data = np.zeros_like(data)

    # Apply the Gaussian filter to each column (neuron) individually
    for i in range(data.shape[1]):  # Iterate over columns
        smoothed_data[:, i] = gaussian_filter1d(data[:, i], sigma=sigma, mode='reflect',
                                                truncate=window_length / (2 * sigma))

    return smoothed_data


def lowess_smoothing_2d(matrix, window_length, window_step):
    """
    Apply LOWESS smoothing to each column of a 2D matrix.

    :param matrix: The input signal data as a 2D NumPy array.
    :param window_length: The length of the window, which determines the amount of smoothing.
    :param window_step: The step size for the window to move for each calculation.
    :return: The smoothed matrix as a 2D NumPy array.
    """
    # Normalize window length relative to the data length
    n_rows = matrix.shape[0]
    frac = window_length / n_rows

    # Generate an array of indices for the data, to be used as the 'x' values in the LOWESS function
    x = np.arange(n_rows)

    # Initialize the smoothed matrix
    smoothed_matrix = np.zeros_like(matrix)

    # Apply LOWESS smoothing to each column
    for i in range(matrix.shape[1]):
        column_data = matrix[:, i]
        smoothed = sm.nonparametric.lowess(column_data, x, frac=frac, it=0, delta=window_step)
        smoothed_matrix[:, i] = smoothed[:, 1]

    return smoothed_matrix


def _plot_hand_trajectory(true_vel, pred_vel, plot, **kwargs):
    '''
    Plot hand trajectory
    :param true_vel:
    :param pred_vel:
    :return:
    '''
    plot_color = kwargs.get('plot_color', 'red')

    # Calculate hand position with initial position (0, 0)
    true_pos = np.cumsum(true_vel, axis=0)
    pred_pos = np.cumsum(pred_vel, axis=0)
    true_pos = true_pos - true_pos[0, :]
    pred_pos = pred_pos - pred_pos[0, :]

    plot.plot(true_pos[:, 0], true_pos[:, 1], label='True', color=plot_color, linewidth=1, linestyle='--')
    plot.plot(pred_pos[:, 0], pred_pos[:, 1], label='Predicted', color=plot_color, linewidth=1, linestyle='-')


def _get_color_for_condition(condition, min_condition, max_condition):
    """
    Maps a condition number to a color.

    :param condition: The condition number.
    :param min_condition: The minimum condition number in the range.
    :param max_condition: The maximum condition number in the range.
    :return: A color corresponding to the condition number.
    """
    # Normalize the condition number
    norm = mcolors.Normalize(vmin=min_condition, vmax=max_condition)

    # Choose a colormap
    colormap = plt.cm.viridis

    # Map the normalized condition number to a color
    return colormap(norm(condition))


def plot_hand_trajectory_conditions(true_vel, pred_vel, labels, trial_number=5, con_num=4, seed=2023):
    '''
    Select 5 conditions and plot at most 10 trials within that condition
    :param true_vel:
    :param pred_vel:
    :param conditions:
    :param trial_number:
    :return:
    '''

    # Set seed
    np.random.seed(seed)

    fig, ax = plt.subplots()
    unique_condition = np.unique(labels)
    # Choose 4 conditions and plot all trials in that condition
    condition_index = np.random.choice(unique_condition, con_num, replace=False)
    plot_index = []

    for i in condition_index:
        plot_index.extend(np.where(labels == i)[0][:trial_number])

    for i in plot_index:
        # Choose a random trial
        color = _get_color_for_condition(labels[i], np.min(labels), np.max(labels))
        _plot_hand_trajectory(pred_vel[i], true_vel[i], ax, plot_color=color)
    ax.set_title('Predicted vs True Hand Trajectory')
    ax.set_xlabel('X position (mm)')
    ax.set_ylabel('Y position (mm)')
    ax.legend(['Condition ' + str(i) for i in condition_index])
    # Change color of legend
    for i in range(len(condition_index)):
        ax.get_legend().get_texts()[i].set_color(_get_color_for_condition(condition_index[i],
                                                                          np.min(labels),
                                                                          np.max(labels)))

    fig.suptitle('Hand Trajectory', fontsize=20)
    plt.show()


def plot_hand_velocity(true_vel, pred_vel, trial_num=5, seed=2023):
    '''
    Plot hand velocity
    :param true_vel:
    :param pred_vel:
    :param trial_num:
    :return:
    '''
    # Set seed
    np.random.seed(seed)
    trial_index = np.random.choice(len(true_vel), trial_num)

    # Combine all trials for hand_velocity and Y_test
    hand_velocity_con = np.array([pred_vel[i] for i in trial_index])
    hand_velocity_con = hand_velocity_con.reshape(-1, hand_velocity_con.shape[-1])
    Y_test_con = np.array([true_vel[i] for i in trial_index])
    Y_test_con = Y_test_con.reshape(-1, Y_test_con.shape[-1])

    # Plot hand_velocity and True value
    plt.figure()
    plt.plot(hand_velocity_con[:, 0], label='Predicted')
    plt.plot(Y_test_con[:, 0], label='True')
    plt.title('Predicted vs True Hand Velocity in X direction')
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(hand_velocity_con[:, 1], label='Predicted')
    plt.plot(Y_test_con[:, 1], label='True')
    plt.title('Predicted vs True Hand Velocity in Y direction')
    plt.legend()
    plt.show()


def convert_angle_mag_to_velocity(angles, magnitudes):
    """
    Convert an array of angles and magnitudes to an array of velocities.

    :param angles: A 1D array of angles in degrees.
    :param magnitudes: A 1D array of magnitudes.
    :return: A 2D array of velocities where each row is [x_velocity, y_velocity].
    """
    # Convert angles to radians
    angles_rad = np.radians(angles)

    # Calculate the x and y velocities
    x_velocities = magnitudes * np.cos(angles_rad)
    y_velocities = magnitudes * np.sin(angles_rad)

    # Combine the velocities into a 2D array
    velocities = np.column_stack((x_velocities, y_velocities))

    return velocities


def plot_latent_states(latent_states, trial_type, trial_num=5, con_num=4, seed=2023):
    '''
    Plot latent states in 3D and choose con_num conditions randomly and plot trial_num trials within that condition
    :param latent_states:
    :param trial_type:
    :param trial_num:
    :param con_num:
    :param seed:
    :return:
    '''
    # Set seed
    np.random.seed(seed)
    fig, ax = plt.subplots()

    unique_condition = np.unique(trial_type)
    # Choose 4 conditions and plot all trials in that condition
    condition_index = np.random.choice(unique_condition, con_num, replace=False)
    plot_index = []

    for i in condition_index:
        plot_index.extend(np.where(trial_type == i)[0][:trial_num])

    ax = fig.add_subplot(111, projection='3d')
    for i in plot_index:
        color = _get_color_for_condition(trial_type[i], np.min(trial_type), np.max(trial_type))
        ax.plot(latent_states[i, :, 0], latent_states[i, :, 1], latent_states[i, :, 2],
                color=color, label='Condition ' + str(trial_type[i]))
    ax.set_title('Latent States')
    ax.set_xlabel('Latent State 1')
    ax.set_ylabel('Latent State 2')
    ax.set_zlabel('Latent State 3')
    ax.legend(['Condition ' + str(i) for i in condition_index])
    # Change color of legend
    for i in range(len(condition_index)):
        ax.get_legend().get_texts()[i].set_color(_get_color_for_condition(condition_index[i],
                                                                          np.min(trial_type),
                                                                          np.max(trial_type)))

    fig.suptitle('Latent States 3D', fontsize=20)

    plt.show()


def plot_latent_states_1d(latent_states, trial_type, trial_num=5, con_num=4, seed=2023):
    '''
    Plot first 10 latent states in 1D and choose con_num conditions randomly and plot trial_num trials within that condition
    :param latent_states:
    :param trial_type:
    :param trial_num:
    :param con_num:
    :param seed:
    :return:
    '''
    # Set seed
    np.random.seed(seed)
    fig, ax = plt.subplots()
    # Set figure larger
    fig.set_size_inches(18.5, 10.5)

    unique_condition = np.unique(trial_type)
    # Choose 4 conditions and plot all trials in that condition
    condition_index = np.random.choice(unique_condition, con_num, replace=False)
    plot_index = []

    for i in condition_index:
        plot_index.extend(np.where(trial_type == i)[0][:trial_num])

    for j in range(10):
        # Plot latent states in each subplot
        ax = fig.add_subplot(2, 5, j + 1)
        for i in plot_index:
            color = _get_color_for_condition(trial_type[i], np.min(trial_type), np.max(trial_type))
            ax.plot(latent_states[i, :, j], color=color, label='Condition ' + str(trial_type[i]))
        ax.set_title('Latent State ' + str(j + 1))
        ax.set_xlabel('Time')
        ax.set_ylabel('Latent State ' + str(j + 1))
        ax.legend(['Condition ' + str(i) for i in condition_index])
        # Change color of legend
    for i in range(len(condition_index)):
        ax.get_legend().get_texts()[i].set_color(_get_color_for_condition(condition_index[i],
                                                                          np.min(trial_type),
                                                                          np.max(trial_type)))

    fig.suptitle('Latent States 1D', fontsize=20)
    plt.show()


def plot_raw_data(data, trial_type, con_num=4, neuron_num=5, seed=2023, label = 'Train'):
    '''
    Plot con_num conditions in each subplot and within each subplot plot neuron_num neurons and trial_num trials
    :param data:
    :param trial_type:
    :param trial_num:
    :param con_num:
    :param neuron_num:
    :param seed:
    :return:
    '''
    # Set seed
    np.random.seed(seed)
    fig, ax = plt.subplots()
    # Set figure larger
    fig.set_size_inches(18.5, 10.5)

    unique_condition = np.unique(trial_type)
    # Choose con_num conditions and plot all trials in that condition
    condition_index = np.random.choice(unique_condition, con_num, replace=False)
    plot_index = [[]] * con_num
    # create empty list for each condition
    neuron_index = np.random.choice(data.shape[-1], neuron_num)

    for i, j in enumerate(condition_index):
        plot_index[i] = np.where(trial_type == j)[0]

    for j, i in enumerate(condition_index):
        # Plot raw data in each subplot
        ax = fig.add_subplot(2, con_num // 2, j + 1)

        for k in neuron_index:
            # Calculate the trial average for each neuron
            neuron_avg = np.mean(data[plot_index[j], :, k], axis=0)
            neuron_conf_int = np.std(data[plot_index[j], :, k], axis=0) / np.sqrt(len(plot_index))
            color = _get_color_for_condition(k, np.min(neuron_index), np.max(neuron_index))
            ax.plot(neuron_avg, color=color, label='Neuron ' + str(k))
            ax.fill_between(np.arange(len(neuron_avg)), neuron_avg - neuron_conf_int, neuron_avg + neuron_conf_int,
                            color=color, alpha=0.2)

        ax.set_title('Condition ' + str(i))
        ax.set_xlabel('Time')
        ax.set_ylabel('Firing Rate')
        ax.legend(['Neuron ' + str(i) for i in neuron_index])
        # Change color of legend
    for i in range(len(neuron_index)):
        ax.get_legend().get_texts()[i].set_color(_get_color_for_condition(neuron_index[i],
                                                                          np.min(neuron_index),
                                                                          np.max(neuron_index)))

    # Add big title
    fig.suptitle(label + ' Raw Data', fontsize=20)
    plt.show()


def cal_R_square(Y_train, Y_test):
    '''
    Calculate R square
    :param Y_train:
    :param Y_test:
    :return:
    '''
    # Calculate the mean of Y_train
    Y_train_mean = np.mean(Y_train, axis=0)

    # Calculate the total sum of squares
    SST = np.sum((Y_test - Y_train_mean) ** 2)

    # Calculate the residual sum of squares
    SSR = np.sum((Y_test - Y_train) ** 2)

    # Calculate R square
    R_square = 1 - SSR / SST

    return R_square


def noisy_bootstrapping(X, y, num_bootstrap_samples, noise_level, stack=True, seed = 2023):
    """
    Perform noisy bootstrapping for regression problems.

    Parameters:
    - X: NumPy array, feature variable data.
    - y: NumPy array, target variable data.
    - num_bootstrap_samples: int, number of bootstrap samples to generate.
    - noise_level: float, controls the standard deviation of the Gaussian noise.

    Returns:
    - X_bootstrapped: NumPy array, combined feature variable data.
    - y_bootstrapped: NumPy array, combined target variable data.
    """

    def add_noise(data, noise_level, seed=2023):
        """
        Add Gaussian noise to the data.

        Parameters:
        - data: NumPy array, input data to which noise will be added.
        - noise_level: float, controls the standard deviation of the Gaussian noise.

        Returns:
        - noisy_data: NumPy array, data with added noise.
        """
        np.random.seed(seed)
        noise = noise_level * np.random.randn(*data.shape)
        noisy_data = data + noise
        return noisy_data
    np.random.seed(seed)
    X_bootstrapped = []
    y_bootstrapped = []

    for _ in range(num_bootstrap_samples):
        # Randomly select a subset of the training data with replacement
        indices = np.random.choice(len(X), len(X), replace=True)

        # Add noise to the features and target variable
        X_noisy = add_noise(X[indices], noise_level * np.std(X, axis=0))
        y_noisy = add_noise(y[indices], noise_level * np.std(y, axis=0))

        X_bootstrapped.append(X_noisy)
        y_bootstrapped.append(y_noisy)

    # Flatten the lists to create NumPy arrays
    X_bootstrapped = np.concatenate(X_bootstrapped)
    y_bootstrapped = np.concatenate(y_bootstrapped)

    if stack == True:
        # Combine the noisy samples with the original data
        X_combined = np.vstack([X, X_bootstrapped])
        y_combined = np.vstack([y, y_bootstrapped])

    else:
        X_combined = X_bootstrapped
        y_combined = y_bootstrapped

    # Shuffle the combined data
    shuffle_indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_indices]
    y_combined = y_combined[shuffle_indices]

    return X_combined, y_combined


def noisy_bootstrapping_condition(X, Y, trial_type, num_bootstrap_samples, noise_level=0.1, stack = True, seed = 2023):
    '''
    Add noise to each condition and perform bootstrapping
    :param X:
    :param Y:
    :param trial_type:
    :param num_bootstrap_samples:
    :param noise_level:
    :return:
    '''
    np.random.seed(seed)
    def add_gaussian_noise(data, std_vec, noise_level, seed = 2023):
        """
        Adds Gaussian noise to the dataset.

        :param data: 3D input data with shape (N, T, D)
        :param std_vec: 2D standard deviation vector with shape (T, D)
        :return: Data with added noise
        """
        np.random.seed(seed)
        N, T, D = data.shape

        # Check if std_vec has the correct shape
        if std_vec.shape != (T, D):
            raise ValueError("std_vec must have the shape (T, D)")

        # Generate noise for each time step and dimension
        noise = np.random.normal(0, std_vec, (N, T, D))*noise_level

        # Add noise to the original data
        noisy_data = data + noise

        return noisy_data

    # Initialize lists to store indices
    unique_condition = np.unique(trial_type)
    X_bootstrapped = []
    Y_bootstrapped = []
    trial_type_bootstrapped = []
    for i,j in enumerate(unique_condition):
        # Randomly select a subset of the training data with replacement
        indices = np.where(trial_type == j)[0]
        train_data = X[indices]
        trian_velocity = Y[indices]
        for _ in range(num_bootstrap_samples):
            data_index = np.random.choice(len(train_data), len(train_data), replace=True)
            boots_strap_data = train_data[data_index]
            boots_strap_velocity = trian_velocity[data_index]
            std_data = np.std(boots_strap_data, axis=0)
            std_velocity = np.std(boots_strap_velocity, axis=0)
            X_noisy = add_gaussian_noise(boots_strap_data, std_data, noise_level)
            Y_noisy = add_gaussian_noise(boots_strap_velocity, std_velocity, noise_level)
            X_bootstrapped.append(X_noisy)
            Y_bootstrapped.append(Y_noisy)
            trial_type_bootstrapped.append(np.ones(len(X_noisy))*j)

    # Flatten the lists to create NumPy arrays
    X_bootstrapped = np.concatenate(X_bootstrapped)
    Y_bootstrapped = np.concatenate(Y_bootstrapped)
    trial_type_bootstrapped = np.concatenate(trial_type_bootstrapped)
    # Combine the noisy samples with the original data
    if stack == True:
        X_combined = np.vstack([X, X_bootstrapped])
        Y_combined = np.vstack([Y, Y_bootstrapped])
        trial_type_combined = np.hstack([trial_type, trial_type_bootstrapped])
    else:    
        X_combined = X_bootstrapped
        Y_combined = Y_bootstrapped
        trial_type_combined =  trial_type_bootstrapped
    # Shuffle the combined data
    shuffle_indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_indices]
    trial_type_combined = trial_type_combined[shuffle_indices]
    Y_combined = Y_combined[shuffle_indices]
    return X_combined, Y_combined, trial_type_combined


if __name__ == '__main__':
    ########## test function ##########
    dataset = import_dataset('Jenkins_small_train.nwb')

    spike, vel = get_spikes_and_velocity(dataset, resample_size=5, smooth=True)

    rate, vel = pre_process_spike(spike, vel, dataset, window_step=5, overlap=False, window_size=5, smooth=False)

    trial_condition = dataset.trial_info.set_index('trial_type').index.tolist()

    X, X_test, Y, Y_test, X_label, X_test_label = get_surrogate_data(rate, vel, trial_condition, trials=50)
