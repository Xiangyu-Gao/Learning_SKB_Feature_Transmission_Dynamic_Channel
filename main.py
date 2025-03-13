import argparse
import datetime
import numpy as np
import itertools
import torch

from sac.sac import SAC
from torch.utils.tensorboard import SummaryWriter
from sac.replay_memory import ReplayMemory
from envs.SemanticTransmission import SemanticTransmission
from sklearn.model_selection import train_test_split
from utils.greedy import greedy_loss_first_action, greedy_latency_first_action
from config import system_config, mat_contents
from utils.sample_data_label import sample_data_label


parser = argparse.ArgumentParser(description="PyTorch Soft Actor-Critic Args")
parser.add_argument(
    "--env-name",
    default="SemanticTransmission",
    help="Wireless Comm environment (default: SemanticTransmission)",
)
parser.add_argument("--exp_name", default=None, help="name this experiment")
parser.add_argument(
    "--method",
    default="drl",
    help="method name: drl, greedy_loss, greedy_latency, (default: drl)",
)
parser.add_argument(
    "--policy",
    default="Gaussian",
    help="Policy Type: Gaussian | Deterministic (default: Gaussian)",
)
parser.add_argument(
    "--eval",
    action="store_false",
    help="Evaluates a policy a policy every 10 episode (default: True)",
)
parser.add_argument(
    "--eval_interval", type=int, default=5, help="test per interval of training episode"
)
parser.add_argument(
    "--skip_training",
    action="store_true",
    help="skip the training process for heuristic algorithms (default: False)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor for reward (default: 0.99)",
)
parser.add_argument(
    "--tau",
    type=float,
    default=0.005,
    metavar="G",
    help="target smoothing coefficient(τ) (default: 0.005)",
)
parser.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    metavar="G",
    help="learning rate (default: 0.0001)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=0.2,
    metavar="G",
    help="Temperature parameter α for entropy term (default: 0.2)",
)
parser.add_argument(
    "--automatic_entropy_tuning",
    action="store_true",
    help="Automatically adjust α (default: False)",
)
parser.add_argument(
    "--entropy_decay", action="store_true", help="decaying α (default: False)"
)
parser.add_argument(
    "--seed",
    type=int,
    default=123456,
    metavar="N",
    help="random seed (default: 123456)",
)
parser.add_argument(
    "--batch_size", type=int, default=256, metavar="N", help="batch size (default: 256)"
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=5000001,
    metavar="N",
    help="maximum number of steps (default: 5000001)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    metavar="N",
    help="hidden size (default: 256)",
)
parser.add_argument(
    "--updates_per_step",
    type=int,
    default=1,
    metavar="N",
    help="model updates per simulator step (default: 1)",
)
parser.add_argument(
    "--start_steps",
    type=int,
    default=20000,
    metavar="N",
    help="Steps sampling1 random actions (default: 5000)",
)
parser.add_argument(
    "--target_update_interval",
    type=int,
    default=1000,
    metavar="N",
    help="Value target update per no. of updates per step (default: 1000)",
)
parser.add_argument(
    "--replay_size",
    type=int,
    default=1000000,
    metavar="N",
    help="size of replay buffer (default: 10000000)",
)
parser.add_argument("--cuda", action="store_true", help="run on CUDA (default: False)")
args = parser.parse_args()

# separate train and test data
X_train, X_test, y_train, y_test = train_test_split(
    mat_contents["X_te"], mat_contents["test_labels"], test_size=0.2, random_state=42
)

# sample the data based on a random Markov probability transition matrix
M = system_config["M"]  # number of samples in each time
indices_samples_train, indices_samples_test = sample_data_label(
    y_train, y_test, sample_size=M, random_seed=40
)

# generate train-test random velocity
max_speed = system_config["max_speed"]
np.random.seed(args.seed)
rand_v_train = np.random.randint(max_speed * 2, size=1000000) - max_speed
np.random.seed(args.seed)
rand_v_test = np.random.randint(max_speed * 2, size=2000) - max_speed

# Environment
env = SemanticTransmission(
    train_data=X_train,
    test_data=X_test,
    train_label=y_train,
    test_label=y_test,
    train_sample_indices=indices_samples_train,
    test_sample_indices=indices_samples_test,
    train_rand_v=rand_v_train,
    test_rand_v=rand_v_test,
    MAX_STEPS=len(X_train),
)
# env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)

# Agent
len_state = (
    10 * env.sample_size + 4
)  # 10 is num. of states per sample, 4 is extra properties
agent = SAC(len_state, env.action_space, args)

# Tensorboard
writer = SummaryWriter(
    "runs/{}_SAC_{}_{}_{}_{}".format(
        datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        args.env_name,
        args.policy,
        "autotune" if args.automatic_entropy_tuning else "",
        args.exp_name if args.exp_name else "",
    )
)
# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0
result_semantic = []  # for evaluation use
result_latency = []  # for evaluation use
result_accuracy = []  # for evaluation use
result_reward = []  # for evaluation use
best_semantic = None
best_latency = None
best_accuracy = None
best_reward = None
best_training_epoch = None

"""
# Format
# action: [l (all samples)], l={1, 2, 3, 4, 5}
# sys_state: [L_1, L_2, L_3, L_4, L_5, T_1, T_2, T_3, T_4, T_5 (all samples), avg_L, avg_T, vel, timestamp]]
"""

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_accuracy = 0
    prev_sys_state = [0] * len_state
    done = False
    env.reset()  # reset the environment

    # update the entropy temperature
    if args.entropy_decay and i_episode > 5:
        agent.alpha = 0
    # customize penalty weight
    if i_episode <= 10:
        env.kappa1 = 50
    else:
        env.kappa1 = 500

    ###############################################################################################################
    # Training
    while not args.skip_training and not done:
        # trigger the env to obtain the system state, data, label, semantic_loss, Y_hit5, trans_latency
        sys_state, cur_data, cur_label, sem_loss, Y_hit5, trans_latency = env.next()

        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(sys_state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (
                    critic_1_loss,
                    critic_2_loss,
                    policy_loss,
                    ent_loss,
                    alpha,
                ) = agent.update_parameters(memory, args.batch_size, updates)
                writer.add_scalar("loss/critic_1", critic_1_loss, updates)
                writer.add_scalar("loss/critic_2", critic_2_loss, updates)
                writer.add_scalar("loss/policy", policy_loss, updates)
                writer.add_scalar("loss/entropy_loss", ent_loss, updates)
                writer.add_scalar("entropy_temprature/alpha", alpha, updates)
                updates += 1

        action_ = env.integer_action(action)  # make action integer
        reward, accuracy, done, info = env.step(
            action_, sys_state, cur_label, Y_hit5
        )  # Env Step
        # print(sys_state, action, reward, cur_label, env.current_step)
        # input()
        total_numsteps += 1
        episode_reward += reward  # the reward
        episode_accuracy += accuracy  # the classification accuracy

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if env.current_step == env.max_episode_steps else float(not done)
        memory.push(
            prev_sys_state, action, reward, sys_state, mask
        )  # Append transition to memory
        prev_sys_state = sys_state.copy()  # update the history of system state

    if total_numsteps >= args.num_steps:
        break
    if not args.skip_training:
        episode_reward /= env.current_step
        episode_accuracy /= env.current_step
        writer.add_scalar("reward/train", episode_reward, i_episode)
        print(
            "Episode:{}, total num. steps:{}, episode steps:{}, reward:{}, avg. semantic loss:{}, avg. latency:{}, "
            "accuracy:{}".format(
                i_episode,
                total_numsteps,
                env.current_step,
                round(episode_reward, 4),
                round(info["avg_loss"], 4),
                round(info["avg_latency"], 4),
                round(episode_accuracy, 4),
            )
        )

    ##################################################################################################################
    # Evaluation
    eval_freq = args.eval_interval  # evaluation frequency: default 10 epochs
    if i_episode % eval_freq == 0 and args.eval is True:
        env.reset()  # reset the environment
        episode_reward = 0
        episode_accuracy = 0

        while env.current_step + 1 < (len(env.test_data) // env.sample_size):
            # trigger the env to obtain the system state, data, label, semantic_loss, Y_hit5, trans_latency
            sys_state, cur_data, cur_label, sem_loss, Y_hit5, trans_latency = env.next(
                evaluate=True
            )

            if args.method == "drl":
                # SAC agent generate action
                action = agent.select_action(sys_state, evaluate=True)
            elif args.method == "greedy_loss":
                # loss-first greedy baseline: input (sem_loss, trans_latency, avg_latency), output (action)
                action = greedy_loss_first_action(
                    sem_loss, trans_latency, sys_state[-3]
                )
            elif args.method == "greedy_latency":
                # latency-first greedy baseline: input (sem_loss, trans_latency), output (action)
                action = greedy_latency_first_action(sem_loss, trans_latency)

            # evaluate the action
            action_ = env.integer_action(action)  # make action integer
            reward, accuracy, done, info = env.step(
                action_, sys_state, cur_label, Y_hit5
            )  # Env Step
            episode_reward += reward
            episode_accuracy += accuracy

        episode_reward /= env.current_step
        episode_accuracy /= env.current_step
        writer.add_scalar("reward/test", episode_reward, i_episode)
        print("----------------------------------------")
        print(
            "Test Episodes: {}, Total Steps: {}, Avg. Reward: {}, avg. semantic loss: {}, avg. latency: {}, "
            "Avg. Accuracy: {}".format(
                0,
                int(env.current_step),
                round(episode_reward, 4),
                round(info["avg_loss"], 4),
                round(info["avg_latency"], 4),
                round(episode_accuracy, 4),
            )
        )
        print("----------------------------------------")

        result_semantic.append(info["avg_loss"])
        result_latency.append(info["avg_latency"])
        result_accuracy.append(episode_accuracy)
        result_reward.append(episode_reward)

        if len(result_semantic) > 10:
            print_result_semantic = np.average(np.asarray(result_semantic[-10:]))
            print_result_latency = np.average(np.asarray(result_latency[-10:]))
            print_result_accuracy = np.average(np.asarray(result_accuracy[-10:]))
            print_result_reward = np.average(np.asarray(result_reward[-10:]))
        else:
            print_result_semantic = np.average(np.asarray(result_semantic))
            print_result_latency = np.average(np.asarray(result_latency))
            print_result_accuracy = np.average(np.asarray(result_accuracy))
            print_result_reward = np.average(np.asarray(result_reward))

        if np.max(np.asarray(result_latency[-10:])) <= system_config["tau"]:
            # The past few testing epochs are all satisfy the latency constraint
            if not best_semantic or print_result_semantic < best_semantic:
                best_semantic = print_result_semantic
                best_latency = print_result_latency
                best_accuracy = print_result_accuracy
                best_reward = print_result_reward
                best_training_epoch = i_episode

        print(
            "Final Avg Results for last 10 testing epochs: Avg. Reward: {}, Avg. Semantic Loss: {}, Avg. latency: {}, "
            "Avg. Accuracy: {}".format(
                round(print_result_reward, 4),
                round(print_result_semantic, 4),
                round(print_result_latency, 4),
                round(print_result_accuracy, 4),
            )
        )
        if best_semantic:
            print(
                "Best performance so far (averaging 10 testing epochs): Training Epoch: {}, Avg. Reward: {}, "
                "Avg. Semantic Loss: {}, Avg. latency: {}, Avg. Accuracy: {}"
                "".format(
                    best_training_epoch,
                    round(best_reward, 4),
                    round(best_semantic, 4),
                    round(best_latency, 4),
                    round(best_accuracy, 4),
                )
            )
        print("----------------------------------------")

        # writer.add_scalar('avg_cost/trans_cost', round(print_avg_trans, 2), i_episode)
        # writer.add_scalar('avg_cost/comp_cost', round(print_avg_comp, 2), i_episode)

    writer.flush()
