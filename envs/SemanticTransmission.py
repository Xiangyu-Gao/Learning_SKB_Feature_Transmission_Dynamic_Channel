import inspect
import os
import sys
import math
import random
import numpy as np
from sklearn.metrics import pairwise_distances
from random import seed
from gym import spaces

# set parent directory as sys path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from config import (
    system_config,
    Visual_trans,
    Semantic_trans,
    kopt_trans,
    Visual_rec,
    Semantic_rec,
    Semantic_mec,
    te_cl_id,
    S_te_pro,
    trans_num_SKL,
    receiv_num_SKL,
    mec_num_SKL,
    beta0,
    B,
    P_U,
    N0,
    Is_AwA_dataset,
)

tau = system_config["tau"]  # transmission delay constraint
M = system_config["M"]  # number of samples in each time
interval = system_config[
    "interval"
]  # the max duration length for an actor with constant speed


class SemanticTransmission(object):
    """Custom Environment that follows gym interface"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        train_data,  # the training samples
        test_data,  # the testing samples
        train_label,
        test_label,
        train_sample_indices,
        test_sample_indices,
        train_rand_v,
        test_rand_v,
        sample_size=M,  # the number of test samples in one batch
        MAX_STEPS=2000,
    ):
        super(SemanticTransmission, self).__init__()
        self.current_step = 0  # for select the request sample in testing and get step for average loss and latency
        self.global_step = 0  # for select the request sample in training
        self.train_data = train_data
        self.test_data = test_data  # N*1024
        self.train_label = train_label  # N*1
        self.test_label = test_label
        self.train_sample_indices = (
            train_sample_indices  # samples indices following the probability transition
        )
        self.test_sample_indices = test_sample_indices
        self.train_rand_v = train_rand_v
        self.test_rand_v = test_rand_v
        self.sample_size = sample_size
        self._max_episode_steps = MAX_STEPS
        self.sum_loss = 0
        self.sum_latency = 0
        self.interval = interval
        self.kappa1 = 500  # weight coefficient for penalty of the extra time
        self.kappa2 = 500  # weight coefficient for penalty of the unfilled time

        # action: [l (all samples)], l={1, 2, 3, 4, 5}
        # sys_state: [L_1, L_2, L_3, L_4, L_5, T_1, T_2, T_3, T_4, T_5 (all samples), avg_L, avg_T, vel, timestamp]]
        self.action_low = np.asarray([0.5] * M)  # for action space
        self.action_high = np.asarray([5.5] * M)  # for action space
        self.action_space = spaces.Box(
            low=self.action_low, high=self.action_high, dtype=np.float16
        )

    def step(self, action, sys_state, cur_label, Y_hit5):
        """
        Parameters
        ----------
        :param action: [l (all samples)], l={1, 2, 3, 4, 5}
        :param sys_state: [L_1, L_2, L_3, L_4, L_5, T_1, T_2, T_3, T_4, T_5 (all samples), avg_L, avg_T, vel, timestamp]]
        :param cur_label:
        :param Y_hit5:
        Returns
        -------
        reward_, accu, done, valid, info (tuple)
        """
        self.current_step += 1
        self.global_step += 1
        done = False

        # calculate the semantic loss based on action
        loss_cost = self.calc_cost(sys_state, action)
        latency_cost = self.calc_latency(sys_state, action)

        # calculate the accuracy
        accu = self.zsl_acc(action, cur_label, Y_hit5)

        # update the sum loss and latency
        self.sum_loss += loss_cost
        self.sum_latency += latency_cost

        # get reward and penalty
        avg_latency = self.sum_latency / self.current_step
        avg_loss = self.sum_loss / self.current_step

        if avg_latency > tau:
            min_achieve_latency = min(sys_state[5:10])
            penalty = (latency_cost - min_achieve_latency) * self.kappa1
        else:
            penalty = (tau - avg_latency) * self.kappa2
            # penalty = 0
        reward_ = -loss_cost - penalty

        # if approach the max steps, stop or reset
        if self.current_step >= self._max_episode_steps:
            done = True

        info = {
            "semantic_loss": loss_cost,
            "latency_cost": latency_cost,
            "action": action,
            "avg_loss": avg_loss,
            "avg_latency": avg_latency,
            "reward": reward_,
        }

        return reward_, accu, done, info

    def reset(self):
        """
        Reset the variables
        """
        self.current_step = 0
        self.sum_loss = 0
        self.sum_latency = 0
        self.global_step -= (
            self.global_step % self.interval
        )  # make the global_epoch in new epoch divided by interval

        # # shuffle the train data and label
        # idx = np.random.permutation(len(self.train_data))
        # self.train_data, self.train_label = self.train_data[idx], self.train_label[idx]

    def render(self):
        """Render the environment to the screen"""
        print(f"Step: {self.global_step}")

    def next(self, evaluate=False):
        """
        :return: update the system state for the next step
        # sys_state: [L_1, L_2, L_3, L_4, L_5, T_1, T_2, T_3, T_4, T_5 (all samples), avg_L, avg_T, vel, timestamp]]
        """
        if evaluate:
            data, label, timestamp, dist, dist1, dist2, vel = self.fetch_data(
                self.test_data,
                self.test_label,
                self.test_sample_indices,
                self.test_rand_v,
                self.current_step,
            )
        else:
            data, label, timestamp, dist, dist1, dist2, vel = self.fetch_data(
                self.train_data,
                self.train_label,
                self.train_sample_indices,
                self.train_rand_v,
                self.global_step,
            )

        # run lossMode3 to get the semantic loss and transmission delay
        sem_loss, Y_hit5, latency = self.lossMode3(data, dist, dist1, dist2)

        # update the system states
        sys_state = []
        for i in range(sem_loss.shape[0]):
            sys_state += list(sem_loss[i, :])
            sys_state += list(latency[i, :])

        if self.current_step == 0:
            avg_loss = 0
            avg_latency = 0
        else:
            avg_loss = self.sum_loss / self.current_step
            avg_latency = self.sum_latency / self.current_step

        sys_state += [avg_loss, avg_latency, vel, timestamp]

        return sys_state, data, label, sem_loss, Y_hit5, latency

    def fetch_data(self, data_all, label_all, sample_indices_all, vel_all, step):
        """
        fetch the data, label, and timestamp from the dataset
        """
        start_idx = step % (len(sample_indices_all) // self.sample_size)
        indices = sample_indices_all[
            start_idx * self.sample_size : (start_idx + 1) * self.sample_size
        ]
        data = data_all[indices, :]
        label = label_all[indices]

        timestamp = step % self.interval + 1  # start from 1
        select_idx = step // self.interval
        vel = vel_all[select_idx % len(vel_all)]
        # consider the scenario that transmitter is moving from (0, 0) towards to receivers with speed vel,
        # receiver is static at (500, 0) and the mec server is initially have distance 500 to both transmitter and receiver
        d_E2E = max(
            1e-4, abs(500 + vel * timestamp)
        )  # constant velocity model, base distance is 500
        d_E2M = max(1e-4, math.sqrt((250 + vel * timestamp) ** 2 + 433**2))
        d_M2E = 500

        return data, label, timestamp, d_E2E, d_E2M, d_M2E, vel

    def integer_action(self, action):
        """
        make the action be integer and satisfy system requirement i.e., between [1, 5]
        """
        action_new = []
        for a in action:
            a = max(1, min(5, round(a)))
            action_new.append(a)

        return np.asarray(action_new)

    def calc_cost(self, sys_state, action):
        """
        # action: [l (all samples)], l={1, 2, 3, 4, 5}
        # sys_state: [L_1, L_2, L_3, L_4, L_5, T_1, T_2, T_3, T_4 T_5 (all samples), avg_L, avg_T, vel, timestamp]]
        """
        loss_cost = 0
        for idx, select_l in enumerate(action):
            loss_cost += sys_state[
                idx * 10 + select_l - 1
            ]  # L_l = sys_state[idx * 10 + select_l - 1]

        return loss_cost / M

    def calc_latency(self, sys_state, action):
        """
        # action: [l (all samples)], l={1, 2, 3, 4, 5}
        # sys_state: [L_1, L_2, L_3, L_4, L_5, T_1, T_2, T_3, T_4, T_5  (all samples), avg_L, avg_T, vel, timestamp]]
        """
        trans_time = 0
        for idx, select_l in enumerate(action):
            trans_time += sys_state[
                idx * 10 + select_l + 4
            ]  # T_l = sys_state[idx * 10 + select_l + 4]

        latency_cost = trans_time / M

        return latency_cost

    def lossMode3(self, data, d_E2E, d_E2M, d_M2E):
        """
        Calculate the semantic loss and transmission delay for sample at  4 modes
        :return:
        """
        # generate the R_E2Es for data where d_E2E = 500 + v*step
        R_E2Es = [self.calc_R_E2E(d_E2E)] * self.sample_size
        R_E2Ms = [self.calc_R_E2E(d_E2M)] * self.sample_size
        R_M2Es = [self.calc_R_E2E(d_M2E)] * self.sample_size

        trans_cl_id = te_cl_id[:trans_num_SKL]
        trans_SKL = S_te_pro[:trans_num_SKL, :]
        receiv_cl_id = te_cl_id[:receiv_num_SKL]
        receiv_SKL = S_te_pro[:receiv_num_SKL, :]
        mec_cl_id = te_cl_id[:mec_num_SKL]
        mec_SKL = S_te_pro[:mec_num_SKL, :]

        # get the size sizes
        num_te, dim_te = data.shape
        _, dim_se = trans_SKL.shape

        # Initialize arrays
        r = np.zeros((num_te, 5))
        Y_hit5 = np.zeros((num_te, 5))
        load = np.zeros((num_te, 5))

        # loss and inference for mode 1: visual feature transmission
        C_est1 = Visual_rec @ data.T
        S_estt1 = Semantic_rec.T @ C_est1
        S_est1 = S_estt1.T
        S_est1 = np.real(S_est1)
        dist2 = pairwise_distances(S_est1, receiv_SKL, metric="cosine")

        HITK = 1
        for i in range(dist2.shape[0]):
            R_E2E = R_E2Es[i]
            sort_dist_i2 = np.sort(dist2[i, :])
            I2 = np.argsort(dist2[i, :])
            r[i, 0] = sort_dist_i2[:HITK]  # semantic loss
            Y_hit5[i, 0] = receiv_cl_id[I2[:HITK]]
            load[i, 0] = dim_te / R_E2E  # transmission delay

        # loss and inference for mode 2: intermediate feature transmission
        C_est2 = Visual_trans @ data.T
        S_estt2 = Semantic_rec.T @ C_est2
        S_est2 = S_estt2.T
        S_est2 = np.real(S_est2)
        dist2 = pairwise_distances(S_est2, receiv_SKL, metric="cosine")

        for i in range(dist2.shape[0]):
            R_E2E = R_E2Es[i]
            sort_dist_i2 = np.sort(dist2[i, :])
            I2 = np.argsort(dist2[i, :])
            r[i, 1] = sort_dist_i2[:HITK]
            Y_hit5[i, 1] = receiv_cl_id[I2[:HITK]]
            load[i, 1] = kopt_trans / R_E2E

        # loss and inference for mode 3: semantic feature transmission
        C_est3 = Visual_trans @ data.T
        S_estt3 = Semantic_trans.T @ C_est3
        S_est3 = S_estt3.T
        S_est3 = np.real(S_est3)
        dist2 = pairwise_distances(S_est3, receiv_SKL, metric="cosine")

        for i in range(dist2.shape[0]):
            R_E2E = R_E2Es[i]
            sort_dist_i2 = np.sort(dist2[i, :])
            I2 = np.argsort(dist2[i, :])
            r[i, 2] = sort_dist_i2[:HITK]
            Y_hit5[i, 2] = receiv_cl_id[I2[:HITK]]
            load[i, 2] = dim_se / R_E2E

        # loss and inference for mode 4: estimated class knowledge transmission
        dist2 = pairwise_distances(S_est3, trans_SKL, metric="cosine")

        for i in range(dist2.shape[0]):
            R_E2E = R_E2Es[i]
            sort_dist_i2 = np.sort(dist2[i, :])
            I2 = np.argsort(dist2[i, :])
            r[i, 3] = sort_dist_i2[:HITK]
            Y_hit5[i, 3] = trans_cl_id[I2[:HITK]]
            a = np.isin(Y_hit5[i, 3], receiv_cl_id)
            load[i, 3] = (1 * a + dim_se * (1 - a)) / R_E2E
            # load[i, 3] = 1 / R_E2E

        # loss and inference for mode 5: MEC server estimated class knowledge transmission
        C_est5 = Visual_trans @ data.T
        S_estt5 = Semantic_mec.T @ C_est5
        S_est5 = S_estt5.T
        S_est5 = np.real(S_est5)
        dist2 = pairwise_distances(S_est5, mec_SKL, metric="cosine")

        for i in range(dist2.shape[0]):
            R_E2M = R_E2Ms[i]
            R_M2E = R_M2Es[i]
            sort_dist_i2 = np.sort(dist2[i, :])
            I2 = np.argsort(dist2[i, :])
            r[i, 4] = sort_dist_i2[:HITK]
            Y_hit5[i, 4] = mec_cl_id[I2[:HITK]]
            a = np.isin(Y_hit5[i, 4], receiv_cl_id)
            load[i, 4] = (kopt_trans / R_E2M + 1 / R_M2E) * (1 * a + dim_se * (1 - a))
            # print(R_E2M, R_M2E, R_E2E, kopt_trans / R_E2M + 1 / R_M2E, kopt_trans / R_E2E, 1 * a + dim_se * (1 - a))

        if Is_AwA_dataset:
            load = load * 4
        r = r * 10

        return r, Y_hit5, load

    def calc_R_E2E(self, dE2E):
        g_E2E = beta0 / (dE2E**2)
        R_E2E = B * math.log(1 + P_U * g_E2E / (B * N0))

        return R_E2E

    def zsl_acc(self, action, test_labels_cub, Y_hit5):
        """
        calculate the accuracy after semantic transmission
        action: [l (all samples)], l in {1, 2, 3, 4, 5}
        """
        n = 0
        num_te = action.shape[0]
        # test_labels_cub = np.squeeze(test_labels_cub)

        for i in range(num_te):
            if action[i] == 1 and test_labels_cub[i] == Y_hit5[i, 0]:
                n += 1
            elif action[i] == 2 and test_labels_cub[i] == Y_hit5[i, 1]:
                n += 1
            elif action[i] == 3 and test_labels_cub[i] == Y_hit5[i, 2]:
                n += 1
            elif action[i] == 4 and test_labels_cub[i] == Y_hit5[i, 3]:
                n += 1
            elif action[i] == 5 and test_labels_cub[i] == Y_hit5[i, 4]:
                n += 1
        zsl_accuracy = n / num_te

        return zsl_accuracy

    @property
    def max_episode_steps(self):
        return self._max_episode_steps
