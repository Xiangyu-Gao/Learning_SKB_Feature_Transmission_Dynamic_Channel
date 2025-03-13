import numpy as np
from config import system_config

tau = system_config["tau"]  # transmission delay constraint


def greedy_loss_first_action(sem_loss, trans_latency, avg_latency):
    """
    greedy algo that seeks for min loss with the constraint of latency
    """
    all_selection = []
    for i in range(sem_loss.shape[0]):
        select_idx = two_sorts(sem_loss[i, :], trans_latency[i, :])
        if avg_latency <= tau:
            all_selection.append(select_idx + 1)  # index start from 1
        else:
            all_selection.append(4)  # add the action for label only transmission

    return all_selection


def greedy_latency_first_action(sem_loss, trans_latency):
    """
    greedy algo that seeks for min latency
    """
    all_selection = []
    for i in range(sem_loss.shape[0]):
        select_idx = two_sorts(trans_latency[i, :], sem_loss[i, :])
        all_selection.append(select_idx + 1)  # index start from 1

    return all_selection


def two_sorts(seq1, seq2):
    """
    special functions that sort based on two seqs
    first priority: min in seq1; if exist two same mins, then go to seq2
    """
    sort_idx = np.argsort(seq1)
    if seq1[sort_idx[0]] == seq1[sort_idx[1]] and seq2[sort_idx[0]] > seq2[sort_idx[1]]:
        select_idx = sort_idx[1]
    else:
        select_idx = sort_idx[0]

    return select_idx
