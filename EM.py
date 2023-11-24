import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm
import numpy.linalg as lin
import warnings

class em_core:

    def __init__(self, data, n_dim=20):
        self.kf_parameter = None
        self.model = None
        self.data = data
        self.n_dim = n_dim

    def _plot_log_likelihood(self, log_likelihood):
        plt.figure()
        plt.plot(log_likelihood)
        plt.xlabel('iteration')
        plt.ylabel('log-likelihood')
        plt.title('EM convergence')
        plt.show()

    def get_parameters(self, n_iters=10, plot=False):
        '''

        :param data: 2D data with shape (timesteps, n_dim)
        :param timesteps:
        :param n_dim:
        :return:
        '''

        print('Running EM algorithm...')
        sttime = time()

        data_dim = self.data.shape[2]
        timesteps = self.data.shape[1]

        # Initialize parameters
        A = np.random.rand(self.n_dim, self.n_dim)
        C = np.random.rand(data_dim, self.n_dim)
        Q = np.eye(self.n_dim)
        R = np.eye(data_dim)
        state_initial = np.random.rand(self.n_dim)
        state_noise_initial = np.eye(self.n_dim)

        initial_state_comb, initial_noise_comb, A_values, C_values, Q_values, R_values, log_likelihood \
            = EM(A, C, Q, R, state_initial, state_noise_initial, self.data, timesteps, n_iters)

        self.initial_state_comb = initial_state_comb
        self.initial_noise_comb = initial_noise_comb
        self.A_values = A_values
        self.C_values = C_values
        self.Q_values = Q_values
        self.R_values = R_values

        if plot:
            self._plot_log_likelihood(log_likelihood)

        print('EM algorithm finished in', time() - sttime, 'seconds')

    def cal_latent_states(self, input_data, current=True, verbose = False):
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

            assert data.shape[1] == self.data.shape[2], 'Data dimension does not match'
            T = data.shape[0]

            x_pred_curr, _, _, x_pred, _ = kalman_filter(self.initial_state_comb[-1], self.initial_noise_comb[-1],
                                                         data, self.A_values[-1], self.C_values[-1],
                                                         self.Q_values[-1], self.R_values[-1], T=T)

            x_pred_new = np.zeros_like(x_pred)
            x_pred_new[0] = self.initial_state_comb[0] / T
            x_pred_new[1:] = x_pred[:-1]

            x_pred_trial.append(x_pred_new)
            x_curr_trial.append(x_pred_curr)

        x_curr_trial = np.array(x_curr_trial)
        x_pred_trial = np.array(x_pred_trial)

        if verbose:
            print('Latent States Calculated in', time() - sttime, 'seconds')

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

    def fit(self, spike_data, move, concatenate=False, **kwargs):
        if concatenate == False:
            alpha = kwargs.get('alpha', np.logspace(-4, 2, 5))
            self.model = GridSearchCV(Ridge(), {'alpha': alpha})

            x_latent = self.cal_latent_states(spike_data, current=True)
            x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
            x_latent = x_latent.reshape(-1, x_latent.shape[-1])

            move = np.array([move[i] for i in range(len(move))])
            move = move.reshape(-1, move.shape[-1])
            self.model.fit(x_latent, move)

        else:
            alpha = kwargs.get('alpha', np.logspace(-4, 2, 5))
            self.model = GridSearchCV(Ridge(), {'alpha': alpha})

            x_latent = self.cal_latent_states(spike_data, current=True)
            x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
            x_latent = np.concatenate((x_latent, spike_data), axis=2)
            x_latent = x_latent.reshape(-1, x_latent.shape[-1])

            move = np.array([move[i] for i in range(len(move))])
            move = move.reshape(-1, move.shape[-1])
            self.model.fit(x_latent, move)

    def kalman_train(self, letent_state, velocity):
        self.kf_parameter = kal_fit(velocity, letent_state)
        em_parameter = {'C_orth' : self.C_values[-1], 'R': self.R_values[-1]}
        self.M, self.kf = get_Mmatrix(em_parameter, self.kf_parameter)

    def kalman_predict(self, spike_data, v_0):
        velocity = decode_KF_sgtrial(self.M, spike_data, v_0)
        return velocity

    def kalman_predict_all(self, spike_data, v_0):
        '''

        :param spike_data: single trial spike data with shape (timesteps, n_dim)
        :param v_0: initial velocity (if not known can use the average of the training velocity)
        :return:
        '''
        vel = list([v_0])
        while len(vel) < len(spike_data):
            vel.append(self.kalman_predict(spike_data[len(vel)], vel[-1][:]))
        return np.array(vel)

    def cal_R_square(self, spike_data, move, concatenate=False):
        if concatenate == False:
            x_latent = self.cal_latent_states(spike_data, current=True)
            x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
            x_latent = x_latent.reshape(-1, x_latent.shape[-1])

            move = np.array([move[i] for i in range(len(move))])
            move = move.reshape(-1, move.shape[-1])
        else:
            x_latent = self.cal_latent_states(spike_data, current=True)
            x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
            x_latent = np.concatenate((x_latent, spike_data), axis=2)
            x_latent = x_latent.reshape(-1, x_latent.shape[-1])

            move = np.array([move[i] for i in range(len(move))])
            move = move.reshape(-1, move.shape[-1])
        return self.model.score(x_latent, move)

    def predict_move(self, spike_data, concatenate=False):
        if concatenate == False:
            x_latent = self.cal_latent_states(spike_data, current=True)
            x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
            x_latent = x_latent.reshape(-1, x_latent.shape[-1])
            y_pred = self.model.predict(x_latent)
        else:
            x_latent = self.cal_latent_states(spike_data, current=True)
            x_latent = np.array([x_latent[i] for i in range(len(x_latent))])
            x_latent = x_latent.reshape(-1, x_latent.shape[-1])
            x_latent = np.concatenate((x_latent, spike_data), axis=1)
            y_pred = self.model.predict(x_latent)
        return y_pred


def calculate_sum_log_likelihood(A, C, Q, R, mu, K, X_total, Y_total):
    ##X_total = N * T * D
    N = X_total.shape[0]
    T = X_total.shape[1]
    sum_log_likelihood = 0
    for i in range(N):
        sum_log_likelihood += calculate_log_likelihood(A, C, Q, R, mu, K, X_total[i], Y_total[i], T)

    return sum_log_likelihood


def calculate_log_likelihood(A, C, Q, R, mu, K, X, Y, T):
    # Precompute inverses and determinants
    inv_R = np.linalg.inv(R)
    inv_Q = np.linalg.inv(Q)
    inv_K = np.linalg.inv(K)
    det_R = np.linalg.det(R)
    det_Q = np.linalg.det(Q)
    det_K = np.linalg.det(K)

    # Calculate terms that don't require looping
    term1 = -0.5 * T * np.log(det_R)
    term3 = -0.5 * np.log(det_K)
    term4 = -0.5 * np.dot((X[0] - mu).T, np.dot(inv_K, X[0] - mu))
    term5 = -0.5 * (T - 1) * np.log(det_Q)

    # Vectorized computation for s1
    Cx = np.dot(C, X.T).T  # C times X for all t
    diff_1 = Y - Cx
    s1 = -0.5 * np.sum(np.einsum('ij,ij->i', np.dot(diff_1, inv_R), diff_1))

    # Vectorized computation for s2
    X_shifted = X[1:]  # X[t] for t from 1 to T
    X_prev = X[:-1]  # X[t-1] for t from 1 to T
    diff_2 = X_shifted - np.dot(A, X_prev.T).T
    s2 = -0.5 * np.sum(np.einsum('ij,ij->i', np.dot(diff_2, inv_Q), diff_2))

    # Calculate the overall log likelihood
    log_likelihood = term1 + s1 + term3 + term4 + term5 + s2

    return log_likelihood


def cal_old_log_likelihood(A, C, Q, R, mu, K, X, Y, T):
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


def kalman_smoothing(A, state_values, error_cov_values, state_predict_values, error_cov_predict_values, T):
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


def M_step(Y_total, X_total, P_1_total, P_2_total, T):
    # T: Number of time steps
    # X: List of estimated kalman backward values x̂₁, x̂₂, ..., x̂ₜ, given T
    # P_1: List of matrices P₁, P₂, ..., Pₜ
    # P_2: List of matrices P_t,t-1
    # Y: List of observed values y₁, y₂, ..., yₜ
    N = X_total.shape[0]

    # # Compute mu and K
    mu = np.mean(X_total[:, 0, :], axis=0)
    K = np.mean(P_1_total[:, 0, :, :], axis=0) - np.outer(mu, mu)

    # Compute C
    C_numerator = np.tensordot(Y_total, X_total, axes=([0, 1], [0, 1]))
    C_denominator = np.sum(P_1_total, axis=(0, 1))
    C = np.dot(C_numerator, np.linalg.inv(C_denominator))

    # Compute R
    YX = np.einsum('nij,nik->jk', Y_total, Y_total)
    YXC = np.einsum('nij,nik->jk', Y_total, np.dot(X_total, C.T))
    R = (YX - YXC) / (T * N)

    # Compute A
    A_numerator = np.sum(P_2_total, axis=(0, 1))
    A_denominator = np.sum(P_1_total[:, :-1, :, :], axis=(0, 1))
    A = np.dot(A_numerator, np.linalg.inv(A_denominator))

    # Compute Q
    Q = np.sum(P_1_total[:, 1:, :, :], axis=(0, 1)) - np.dot(A_numerator, A.T)
    Q = Q / ((T - 1) * N)

    return mu, K, A, C, Q, R


def EM(A_initial, C_initial, Q_initial, R_initial, state_initial, state_noise_initial, Y, T, n_iters=1000):
    '''

    :param A_initial:
    :param C_initial:
    :param Q_initial:
    :param R_initial:
    :param state_initial:
    :param state_noise_initial:
    :param Y: list of 2D arrays with shape (timesteps, n_dim)
    :param T:
    :param n_iters:
    :return:
    '''
    p = len(state_initial)
    q = Y.shape[-1]
    trial_number = len(Y)
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
        state_smooth_values = []
        P_1 = []
        P_2 = []
        state_values = []
        for k in range(trial_number):
            state_values_tep, error_cov_values, K_values, state_predict_values, error_cov_predict_values = \
                kalman_filter(initial_state_comb[i], initial_noise_comb[i], Y[k],
                              A_values[i], C_values[i], Q_values[i], R_values[i], T)

            state_values.append(state_values_tep)

            state_smooth_values_tep, error_cov_smooth_values, \
                P_1_tep, S_values = kalman_smoothing(A_values[i], state_values_tep, error_cov_values,
                                                     state_predict_values, error_cov_predict_values, T)

            P_1.append(P_1_tep)
            state_smooth_values.append(state_smooth_values_tep)

            P_2_tep = get_P_2(state_smooth_values_tep, S_values, K_values,
                              error_cov_values, A_values[i], C_values[i], T)
            P_2.append(P_2_tep)

        state_values = np.array(state_values)
        state_smooth_values = np.array(state_smooth_values)
        P_1 = np.array(P_1)
        P_2 = np.array(P_2)

        initial_state_comb[i + 1], initial_noise_comb[i + 1], A_values[i + 1], C_values[i + 1], Q_values[i + 1], \
            R_values[i + 1] = M_step(Y, state_smooth_values, P_1, P_2, T)

        log_likelihood[i] = calculate_sum_log_likelihood(A_values[i + 1], C_values[i + 1], Q_values[i + 1],
                                                         R_values[i + 1],
                                                         initial_state_comb[i + 1], initial_noise_comb[i + 1],
                                                         state_values, Y)

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


def kal_fit(x,y):
    '''
    Use latent states and kinematics data to find kalman filter parameters
    :param x: kinematics data for each trial
    :param y: latent state for each trial
    :return: kf: Kalman filter paremeter set:
        miu_1,V1 as the covariance matrix of the initial position of the training set
        A as the matrix for state transition
        C and d as encoding matrix for neural data
        x_{t+1} = Ax_t + w_t
        y_t = Cx_t + d + v_t ,
        R and Q as the covariance matrix for w_t and v_t
    Originally from William Bishop
    Author: Xiecheng Shao
    '''

    # Transpose second and third dimension of x and y
    x = np.array([x[i].T for i in range(0,len(x))])
    y = np.array([y[i].T for i in range(0,len(y))])

    #Learn the initial distribution of kinematics data
    x0 = [x[i][:,0] for i in range(0,len(x))]
    x0 = np.vstack(x0[:])
    mu_1 = np.mean(x0,axis=0)
    V_1 = np.cov(x0.T)

    #Learn covariance matrix for state transition noise
    x_first = [x[i][:,0:-1] for i in range(0,len(x))]
    x_first = np.hstack(x_first[:])
    x_second = [x[i][:,1:] for i in range(0,len(x))]
    x_second = np.hstack(x_second[:])
    A = lin.lstsq(x_first.T,x_second.T,rcond=None)[0]
    x_secpred = np.dot(A,x_first)
    x_delt = x_secpred - x_second
    Q = np.cov(x_delt)

    #Calculate C and d
    x_all = np.hstack(x[:])
    tep = np.ones((1,len(x_all.T)))
    x_all = np.vstack((x_all,tep))
    y_all = np.hstack(y[:])
    Cd = lin.lstsq(x_all.T,y_all.T,rcond=None)[0]
    Cd = Cd.T
    C = Cd[:,0:-1]
    d = Cd[:,-1]

    #Calculate R
    ypred = np.dot(Cd,x_all)
    ydelta = y_all - ypred
    R = np.cov(ydelta)

    kf = Kfclass(mu_1,V_1,A,Q,C,d,R,None)

    return kf


def get_Mmatrix(fapara,kfpara):
    '''
    :param fapara: Factor Analysis parameters, which should contain : C,d and R : see extra_traj.py
    :param kfpara: Kalman Filter parameters, which should contain: A,C,d,R,Q,V1,mu1 : see kal_fit above
    :return: Mmartrix: Decoding matrix with M2,M1,M0
    The code follows the criteria of :
        x_t = M_2y_t + M_1x_{t-1} + M_0
    Where x_t is kinematics data and y_t is spike bin count
    For the first step, x_{t-1} will be set to mu_1
    Originally from William Bishop
    Author: Xiecheng Shao
    '''

    faC = fapara['C_orth']
    faR = fapara['R']
    kfA = getattr(kfpara,'A')
    kfC = getattr(kfpara,'C')
    kfd = getattr(kfpara,'d')
    kfR = getattr(kfpara,'R')
    kfQ = getattr(kfpara,'Q')
    kfV1 = getattr(kfpara,'V1')
    kfmiu1 = getattr(kfpara,'mu_1')

    maxstep = 1e6
    tol = 1e-8

    #Calculate beta
    beta = lin.lstsq(np.dot(faC,faC.T)+faR,faC,rcond=None)[0].T

    #Find Converged Kalman Gain
    prevGain = np.ones(np.shape(kfC.T))*np.inf
    delta = np.inf
    one_setpV = kfV1
    curstep = 1

    while (curstep < maxstep + 1) & (delta > tol):
        curGain = lin.lstsq((kfR + np.dot(np.dot(kfC,one_setpV),kfC.T)).T, np.dot(one_setpV,kfC.T).T, rcond=None)[0]
        curGain = curGain.T
        curV = one_setpV - np.dot(np.dot(curGain,kfC),one_setpV)
        one_setpV = np.dot(np.dot(kfA,curV),kfA.T) + kfQ
        # one_setpV.dropna(inplace=True)
        delta = np.sum(np.square(curGain - prevGain))
        prevGain = curGain
        curstep += 1
        print('delta is ' + str(delta))

    if delta > tol:
        warnings.warn('Unable to find Kalman gain')
    else:
        print('Total loop is '+ str(curstep))
    kf = Kfclass(kfmiu1,kfV1,kfA,kfQ,kfC,kfd,kfR,curGain)

    #Calculate M Matrix
    M2 = np.dot(curGain,beta)
    M1 = np.dot(np.eye(len(kfA)) - np.dot(curGain,kfC),kfA)
    M0 = np.dot(-curGain,kfd[:,None])

    Mmatrix = Mmatclass(M0,M1,M2,kfmiu1)

    return Mmatrix, kf


def decode_KF_sgtrial(Mmatrix,data,x0):
    '''
    :param Mmatrix:
    :param data:
    :param x0: initial hand velocity
    :return: kinematics: decoded hand velocity
    This function decode neural spike count to kinematics data using the Mmatrix, see: get_Mmatrix. However, this function
    only serves for off-line decoding as the number of decoding will be determined by the input
    '''

    M2 = getattr(Mmatrix,'M2')
    M1 = getattr(Mmatrix, 'M1')
    M0 = getattr(Mmatrix, 'M0')

    kinematics = np.dot(M2,data) + np.dot(M1,x0) + M0[:,0]

    return kinematics


class Kfclass():
    def __init__(self,mu_1,V1,A,Q,C,d,R,curGain):
        self.mu_1 = mu_1
        self.V1 = V1
        self.A = A
        self.Q = Q
        self.C = C
        self.d = d
        self.R = R
        self.curGain = curGain


class Mmatclass():
    def __init__(self,M0,M1,M2,miu1):
        self.M0 = M0
        self.M1 = M1
        self.M2 = M2
        self.mu1 = miu1


if __name__ == '__main__':
    # Create artificial data with 250 columns and 150 rows
    data = np.load('observations.npz')['Y']

    # Keep the same feature but divide the data into 1000 trials
    data = data.reshape(1000, -1, data.shape[-1])

    Y_train = data[:50]
    Y_test = data[50:]

    # Run EM
    EM_class = em_core(Y_train, n_dim=2, n_iters=3)
    EM_class.get_parameters(plot=True)

    # Predict latent states
    latent_states = EM_class.cal_latent_states(Y_train, current=False)
    back_predict = np.array([EM_class.get_one_step_ahead_prediction(latent_states[i])
                             for i in range(len(latent_states))])

    # Plot back_predict and True value
    plt.plot(back_predict[0, :, 0], label='Predicted')
    plt.plot(Y_train[0, :, 0], label='True')
    plt.legend()
    plt.show()

    # Calculate MSE
    mse = np.mean((back_predict - Y_train) ** 2)
    print('MSE:', mse)
