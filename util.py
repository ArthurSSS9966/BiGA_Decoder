from nlb_tools.nwb_interface import NWBDataset
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import time
from scipy.ndimage import gaussian_filter1d
import statsmodels.api as sm

def import_dataset(filepath,filetype='nwb'):
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


def get_spikes_and_velocity(dataset, resample_size=1, smooth=False):
    '''
    Extract spikes and velocity from dataset
    :param dataset:
    :param resample_size:
    :return:
    '''

    if smooth:
        assert resample_size >= 5, 'Resample size must be greater than 5 for smoothing'
        dataset.resample(5)
        dataset.smooth_spk(50, name='smth_spk')
    else:
        dataset.resample(resample_size)

    # Extract neural data and lagged hand velocity
    trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-130, 370))
    lagged_trial_data = dataset.make_trial_data(align_field='move_onset_time', align_range=(-50, 450))

    # Extract spikes and velocity
    if smooth:
        spikes = _seg_data_by_trial(trial_data, data_type='spikes_smth_spk')
    else:
        spikes = _seg_data_by_trial(trial_data, data_type='spikes')
    vel = _seg_data_by_trial(lagged_trial_data, data_type='hand_vel')

    return spikes, vel


def pre_process_spike(spikes, vel, dataset, window_step=50, overlap=False, smooth=True, **kwargs):
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

        down_spikes[i] = _downsample_data(spikes[i], downsample_rate=window_step, overlap=overlap, **kwargs)  # Smooth with 250ms moving window
        # Downsample velocity the same rate as spikes
        bin_num = down_spikes[i].shape[0]
        down_ind = np.linspace(0, vel[i].shape[0]-1, bin_num).astype(int)
        down_vel[i] = vel[i][down_ind, :]

        # Convert spikes from Poisson to Gaussian
        down_spikes[i] = np.power(down_spikes[i], 3/4)  #TODO: change to 3/4

        if smooth:
            # Apply Gaussian smoothing to spikes
            down_spikes[i] = apply_gaussian_smoothing_2d(down_spikes[i],
                                                         window_length=window_step//5*2,
                                                         window_step=window_step//5)

        if not overlap:
            rate[i] = down_spikes[i] / dataset.bin_width * 1000 / window_step  # Convert to Hz
        else:
            rate[i] = down_spikes[i] / dataset.bin_width * 1000 / kwargs.get('window_size', window_step)  # Convert to Hz

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
    re_sampled_data = None
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


def get_surrogate_data(train_spikes, train_velocity, trials=50):
    '''

    :param train_spikes:
    :param train_velocity:
    :param trials:
    :return:
    '''
    # Generate surrogate data Using 50 trials
    # Choose Train data index from a random sample of 50 trials
    train_data_idx = np.random.choice(len(train_spikes), trials, replace=False)
    # Choose Test data index that is different from Train data index
    test_data_idx = np.random.choice([i for i in range(len(train_spikes)) if i not in train_data_idx], trials,
                                        replace=False)

    surrogate_data = np.array([train_spikes[i] for i in range(train_data_idx.shape[0])])
    surrogate_data = surrogate_data.reshape(-1, surrogate_data.shape[-1])

    surrogate_data_test = np.array([train_spikes[i] for i in range(test_data_idx.shape[0])])

    surrogate_data_movement = np.array([train_velocity[i] for i in range(train_data_idx.shape[0])])
    surrogate_data_movement = surrogate_data_movement.reshape(-1, surrogate_data_movement.shape[-1])

    surrogate_data_movement_test = np.array([train_velocity[i] for i in range(test_data_idx.shape[0])])

    # Examine if any column is all 0
    print('Number of all 0 columns:', np.sum(np.sum(surrogate_data, axis=0) == 0))
    # Get column index of all 0 columns
    all_0_col_idx = np.where(np.sum(surrogate_data, axis=0) == 0)[0]
    # Delete all 0 columns
    surrogate_data = np.delete(surrogate_data, all_0_col_idx, axis=1)
    surrogate_data_test = np.delete(surrogate_data_test, all_0_col_idx, axis=2)

    return surrogate_data, surrogate_data_test, surrogate_data_movement, surrogate_data_movement_test


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
                                                truncate=window_length/(2*sigma))

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


def plot_hand_trajectory(true_vel, pred_vel, plot):
    '''
    Plot hand trajectory
    :param true_vel:
    :param pred_vel:
    :return:
    '''
    # Calculate hand position with initial position (0, 0)
    true_pos = np.cumsum(true_vel, axis=0)
    true_pos = true_pos - true_pos[0, :]
    pred_pos = np.cumsum(pred_vel, axis=0)
    pred_pos = pred_pos - pred_pos[0, :]
    plot.plot(true_pos[:, 0], true_pos[:, 1], label='True',color='black', linewidth=1)
    plot.plot(pred_pos[:, 0], pred_pos[:, 1], label='Predicted', color='red', linewidth=1)
    return plot

if __name__ == '__main__':

    ########## test function ##########
    dataset = import_dataset('Jenkins_small_train.nwb')

    spike, vel = get_spikes_and_velocity(dataset, resample_size=5, smooth=True)

    rate, vel = pre_process_spike(spike, vel, dataset, window_step=5, overlap=False, window_size=5, smooth=False)

    X, X_test, Y, Y_test = get_surrogate_data(rate, vel, trials=50)

