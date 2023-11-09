import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


class em_core:

    def __init__(self, data, n_dim=20, n_iters=1000):
        self.model = None
        self.data = data
        self.n_dim = n_dim
        self.n_iters = n_iters

    def _plot_log_likelihood(self, log_likelihood):
        plt.plot(log_likelihood)
        plt.xlabel('iteration')
        plt.ylabel('log-likelihood')
        plt.title('EM convergence')
        plt.show()

    def get_parameters(self, plot=False):
        '''

        :param data: 2D data with shape (timesteps, n_dim)
        :param timesteps:
        :param n_dim:
        :return:
        '''

        print('Running EM algorithm...')
        sttime = time()

        data_dim = self.data.shape[1]
        timesteps = self.data.shape[0]

        # Initialize parameters
        A = np.random.rand(self.n_dim, self.n_dim)
        C = np.random.rand(data_dim, self.n_dim)
        Q = np.eye(self.n_dim)
        R = np.eye(data_dim)
        state_initial = np.random.rand(self.n_dim)
        state_noise_initial = np.eye(self.n_dim)

        initial_state_comb, initial_noise_comb, A_values, C_values, Q_values, R_values, log_likelihood \
            = EM(A, C, Q, R, state_initial, state_noise_initial, self.data, timesteps, self.n_iters)

        self.initial_state_comb = initial_state_comb
        self.initial_noise_comb = initial_noise_comb
        self.A_values = A_values
        self.C_values = C_values
        self.Q_values = Q_values
        self.R_values = R_values

        if plot:
            self._plot_log_likelihood(log_likelihood)

        print('EM algorithm finished in', time() - sttime, 'seconds')

    def cal_latent_states(self, input_data, current=True):
        '''

        :param data:
        :return:
        '''

        print('Calculating Latent States...')
        sttime = time()

        # Check the number of dimensions of data
        dimension = len(input_data.shape)

        trials = 0

        if dimension == 2:
            trials = 1

        elif dimension == 3:
            trials = input_data.shape[0]

        x_pred_trial = []
        x_curr_trial = []

        for i in range(trials):

            if trials == 1:
                data = input_data

            else:
                data = input_data[i]

            assert data.shape[1] == self.data.shape[1], 'Data dimension does not match'
            T = data.shape[0]

            x_pred_curr, _, _, x_pred, _ = kalman_filter(self.initial_state_comb[-1], self.initial_noise_comb[-1],
                                               data, self.A_values[-1], self.C_values[-1],
                                               self.Q_values[-1], self.R_values[-1], T=T)

            x_pred_new = np.zeros_like(x_pred)
            x_pred_new[0] = self.initial_state_comb[0]/T
            x_pred_new[1:] = x_pred[:-1]

            x_pred_trial.append(x_pred_new)
            x_curr_trial.append(x_pred_curr)

        x_curr_trial = np.array(x_curr_trial)
        x_pred_trial = np.array(x_pred_trial)

        print('Prediction finished in', time() - sttime, 'seconds')

        if current:
            return x_curr_trial

        else:
            return x_pred_trial

    def get_one_step_ahead_prediction(self, x_pred):
        '''

        :param x_pred:
        :return:
        '''

        print('Running One step ahead prediction...')
        C = self.C_values[-1]
        T = x_pred.shape[0]
        y_pred = np.zeros((T, C.shape[0]))
        for i in range(T):
            y_pred[i] = np.dot(C, x_pred[i])

        return y_pred

    def fit(self, spike_data, move, **kwargs):
        alpha = kwargs.get('alpha', np.logspace(-4, 0, 5))
        self.model = GridSearchCV(Ridge(), {'alpha': alpha})
        x_latent = self.cal_latent_states(spike_data, current=True)
        x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
        x_latent = x_latent.reshape(-1, x_latent.shape[-1])
        self.model.fit(x_latent, move)

    def predict_move(self, spike_data):
        x_latent = self.cal_latent_states(spike_data, current=True)
        x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
        x_latent = x_latent.reshape(-1, x_latent.shape[-1])
        y_pred = self.model.predict(x_latent)
        return y_pred


def calculate_log_likelihood(A, C, Q, R, mu, K, X, Y, T):
    # Calculate the log likelihood for the different terms
    term1 = -0.5 * T * np.log(np.linalg.det(R))
    s1 = 0
    s2 = 0
    for t in range(T):
        y_t = Y[t]
        Cx_t = np.dot(C, X[t])
        inv_R = np.linalg.inv(R)
        diff_1 = y_t - Cx_t
        s1 += - 0.5 * np.dot(diff_1.T, np.dot(inv_R, diff_1))

    for t in range(1, T):
        x_t = X[t]
        x_prev = X[t - 1]
        diff_2 = x_t - np.dot(A, x_prev)
        inv_Q = np.linalg.inv(Q)
        s2 += -0.5 * np.dot(diff_2.T, np.dot(inv_Q, diff_2))

    term3 = -0.5 * np.log(np.linalg.det(K))
    term4 = -0.5 * (X[0] - mu).T.dot(np.linalg.inv(K)).dot(X[0] - mu)
    term5 = -0.5 * (T - 1) * np.log(np.linalg.det(Q))

    # Calculate the overall log likelihood
    log_likelihood = term1 + s1 + term3 + term4 + term5 + s2

    return log_likelihood


def kalman_filter(initial_state, initial_cov, Y, A, C, Q, R, T):
    state_predict = initial_state
    error_cov_predict = initial_cov
    p = len(initial_state)
    q = len(Y[0])

    # Lists to store the values at each time step
    K_values = np.zeros((T, p, q))
    state_values = np.zeros((T, p))
    state_predict_values = np.zeros((T, p))
    error_cov_values = np.zeros((T, p, p))
    error_cov_predict_values = np.zeros((T, p, p))

    for t in range(T):
        # Update step
        K = error_cov_predict.dot(C.T).dot(np.linalg.inv(C.dot(error_cov_predict).dot(C.T) + R))
        state = state_predict + K.dot(Y[t] - C.dot(state_predict))
        error_cov = (np.eye(len(initial_state)) - K.dot(C)).dot(error_cov_predict)

        # Store values for this time step
        K_values[t] = K
        state_values[t] = state
        error_cov_values[t] = error_cov

        # Prediction step
        state_predict = A.dot(state)
        error_cov_predict = A.dot(error_cov).dot(A.T) + Q

        state_predict_values[t] = state_predict
        error_cov_predict_values[t] = error_cov_predict

    return state_values, error_cov_values, K_values, state_predict_values, error_cov_predict_values


def kalman_smoothing(Y, A, state_values, error_cov_values, state_predict_values, error_cov_predict_values, T):
    p = len(state_values[0])
    state_smooth = state_values[-1]
    error_cov_smooth = error_cov_values[-1]

    state_smooth_values = np.zeros((T, p))
    error_cov_smooth_values = np.zeros((T, p, p))
    S_values = np.zeros((T - 1, p, p))
    P_1 = np.zeros((T, p, p))

    state_smooth_values[-1] = state_smooth
    error_cov_smooth_values[-1] = error_cov_smooth
    P_1[-1] = error_cov_smooth + np.outer(state_smooth, state_smooth)

    for t in reversed(range(T - 1)):
        S = error_cov_values[t].dot(A.T).dot(np.linalg.inv(error_cov_predict_values[t]))
        state_smooth = state_values[t] + S.dot(state_smooth - state_predict_values[t])
        error_cov_smooth = error_cov_values[t] + S.dot(error_cov_smooth - error_cov_predict_values[t]).dot(S.T)

        # Store values for this time step
        state_smooth_values[t] = state_smooth
        error_cov_smooth_values[t] = error_cov_smooth
        S_values[t] = S
        P_1[t] = error_cov_smooth + np.outer(state_smooth, state_smooth)

    return state_smooth_values, error_cov_smooth_values, P_1, S_values


def get_P_2(state_smooth_values, S_values, K_values, error_cov_values, A, C, T):
    p = len(state_smooth_values[0])
    P_2 = np.zeros((T - 1, p, p))

    # initialize
    error_cross_cov = (np.eye(p) - K_values[-1].dot(C)).dot(A).dot(error_cov_values[-2])
    P_2[-1] = error_cross_cov + np.outer(state_smooth_values[-1], state_smooth_values[-2])

    for t in reversed(range(T - 2)):
        error_cross_cov = error_cov_values[t + 1].dot(S_values[t].T) + S_values[t + 1].dot(
            error_cross_cov - A.dot(error_cov_values[t + 1])).dot(S_values[t].T)
        P_2[t] = error_cross_cov + np.outer(state_smooth_values[t + 1], state_smooth_values[t])

    return P_2


def M_step(Y, X, P_1, P_2, T):
    # T: Number of time steps
    # X: List of estimated kalman backward values x̂₁, x̂₂, ..., x̂ₜ, given T
    # P_1: List of matrices P₁, P₂, ..., Pₜ
    # P_2: List of matrices P_t,t-1
    # Y: List of observed values y₁, y₂, ..., yₜ
    mu = X[0]
    K = P_1[0] - np.outer(X[0], X[0])

    num_rows = len(Y[0])
    num_cols = len(X[0])

    C_numerator = np.zeros((num_rows, num_cols))
    C_denominator = np.zeros_like(P_1[0])
    for t in range(T):
        C_numerator += np.outer(Y[t], X[t])
        C_denominator += P_1[t]
    C = np.dot(C_numerator, np.linalg.inv(C_denominator))

    R = np.zeros((num_rows, num_rows))
    for t in range(T):
        R += np.outer(Y[t], Y[t]) - np.outer(Y[t], np.dot(C, X[t]))
    R = R / T

    A_numerator = np.zeros_like(P_1[0])
    A_denominator = np.zeros_like(P_1[0])
    for t in range(T - 1):
        A_numerator += P_2[t]
        A_denominator += P_1[t]
    A = np.dot(A_numerator, np.linalg.inv(A_denominator))

    Q = np.zeros_like(P_1[0])
    for t in range(T - 1):
        Q += P_1[t + 1] - np.dot(P_2[t], A.T)
    Q = Q / (T - 1)

    return mu, K, A, C, Q, R


def EM(A_initial, C_initial, Q_initial, R_initial, state_initial, state_noise_initial, Y, T, n_iters=1000):
    '''

    :param A_initial:
    :param C_initial:
    :param Q_initial:
    :param R_initial:
    :param state_initial:
    :param state_noise_initial:
    :param Y:
    :param T:
    :param n_iters:
    :return:
    '''
    p = len(state_initial)
    q = len(Y[0])
    tol = 1e-8

    initial_state_comb = np.zeros((n_iters + 1, p))
    initial_noise_comb = np.zeros((n_iters + 1, p, p))
    A_values = np.zeros((n_iters + 1, p, p))
    C_values = np.zeros((n_iters + 1, q, p))
    Q_values = np.zeros((n_iters + 1, p, p))
    R_values = np.zeros((n_iters + 1, q, q))
    log_likelihood = np.zeros(n_iters)

    initial_state_comb[0] = state_initial
    initial_noise_comb[0] = state_noise_initial

    A_values[0] = A_initial
    C_values[0] = C_initial
    Q_values[0] = Q_initial
    R_values[0] = R_initial

    for i in tqdm(range(n_iters), desc='EM iteration: '):
        state_values, error_cov_values, K_values, state_predict_values, error_cov_predict_values = kalman_filter(
            initial_state_comb[i], initial_noise_comb[i], Y, A_values[i], C_values[i], Q_values[i], R_values[i], T)
        state_smooth_values, error_cov_smooth_values, P_1, S_values = kalman_smoothing(Y, A_values[i], state_values,
                                                                                       error_cov_values,
                                                                                       state_predict_values,
                                                                                       error_cov_predict_values, T)
        P_2 = get_P_2(state_smooth_values, S_values, K_values, error_cov_values, A_values[i], C_values[i], T)
        initial_state_comb[i + 1], initial_noise_comb[i + 1], A_values[i + 1], C_values[i + 1], Q_values[i + 1], \
            R_values[i + 1] = M_step(Y, state_smooth_values, P_1, P_2, T)
        log_likelihood[i] = calculate_log_likelihood(A_values[i + 1], C_values[i + 1], Q_values[i + 1], R_values[i + 1],
                                                     initial_state_comb[i + 1], initial_noise_comb[i + 1],
                                                     state_values, Y, T)

        if i <= 2:
            LLbase = log_likelihood[i]
        elif log_likelihood[i] < log_likelihood[i - 1]:
            print('Error: Data likelihood has decreased from', log_likelihood[i - 1], 'to', log_likelihood[i], '\n')
        elif log_likelihood[i] - LLbase < tol * (log_likelihood[i - 1] - LLbase):
            print('Log likelihood increase stopped at iteration %d', i)
            break

    if len(log_likelihood) < n_iters:
        print('Log likelihood increase stopped at iteration %d', n_iters)

    return initial_state_comb, initial_noise_comb, A_values, C_values, Q_values, R_values, log_likelihood


if __name__ == '__main__':
    # Create artificial data with 250 columns and 150 rows
    data = np.load('observations.npz')['Y']

    Y_train = data[:10000]
    Y_test = data[10000:]

    # Run EM
    EM_class = em_core(Y_train, n_dim=4, n_iters=10)
    EM_class.get_parameters(plot=True)

    # Predict latent states
    latent_states = EM_class.cal_latent_states(Y_train, current=False)
    back_predict = EM_class.get_one_step_ahead_prediction(latent_states)

    # Plot back_predict and True value
    plt.plot(back_predict[:, 0], label='Predicted')
    plt.plot(Y_train[:, 0], label='True')
    plt.legend()
    plt.show()

    # Calculate MSE
    mse = np.mean((back_predict - Y_train) ** 2)
    print('MSE:', mse)
