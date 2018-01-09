import numpy as np
from helpers.utils import compute_metrics


def generate_vote(gt, machine):
    acc = machine[gt]  # take either acc_positive of acc_negative
    if np.random.binomial(1, acc):
        return gt
    else:
        return 1 - gt


def test_machines_accuracy(filters_num, machines_params):
    # test machines
    tests_num = 20  # 20 test items
    machines_accuracy = [[] for _ in range(filters_num)]
    for filter_index in range(filters_num):
        for machine in machines_params[filter_index]:
            # Indeed,  we need to reject the machine if acc < 0.5
            while True:
                correct_votes_num = sum(np.random.binomial(1, machine[0], tests_num // 2)) + \
                                    sum(np.random.binomial(1, machine[1], tests_num // 2))
                acc = correct_votes_num / tests_num
                if acc > 0.5 and acc != 1.:
                    break
            machines_accuracy[filter_index].append(acc)
    return machines_accuracy


# fuse votes via weighted majority voting
# output: probabilities to be negatives for each filter and item
def weighted_mv(votes_list, filters_num, items_num, machines_accuracy):
    probs_list = [None]*filters_num*items_num
    for filter_index in range(filters_num):
        filter_machines_acc = machines_accuracy[filter_index]
        for item_index in range(items_num):
            like_true_val = 1  # assume true value is positive
            a, b = 1., 1.  # constituents of baysian formula, prior is uniform dist.
            # a responds for positives, b - for negatives
            for vote, acc in zip(votes_list[item_index*filters_num + filter_index], filter_machines_acc):
                if vote == like_true_val:
                    a *= acc
                    b *= 1 - acc
                else:
                    a *= 1 - acc
                    b *= acc
            probs_list[item_index*filters_num + filter_index] = b / (a + b)
    return probs_list


def classify_items(ensembled_vote, lr, filters_num, items_num):
    prob_in_list = []
    items_labels = []
    pos_thr = lr / (1. + lr)  # threshold to classify as a positive
    for item_index in range(items_num):
        prob_all_neg = 1.
        for filter_index in range(filters_num):
            prob_all_neg *= ensembled_vote[item_index*filters_num + filter_index]
        prob_item_pos = 1. - prob_all_neg
        prob_in_list.append(prob_item_pos)

        # classify item
        if prob_item_pos > pos_thr:
            items_labels.append(0)
        else:
            items_labels.append(1)
    return items_labels, prob_in_list


def machine_ensemble(filters_num, items_num, gt_values, lr):
    # parameters for machine-based classifiers (accuracy for positives, accuracy for negatives)
    # positive vote - out of scope
    # negative vote - in scope
    machines_params = [[(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)],  # machines for criteria 0
                       [(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)],  # machines for criteria 1
                       [(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)],  # machines for criteria 2
                       [(0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8), (0.9, 0.8)]]  # machines for criteria 3

    votes_list = [[] for _ in range(items_num*filters_num)]
    for item_index in range(items_num):
        for filter_index in range(filters_num):
            gt = gt_values[item_index*filters_num + filter_index]  # can be either 0 or 1
            for machine in machines_params[filter_index]:
                vote = generate_vote(gt, machine)
                votes_list[item_index*filters_num + filter_index].append(vote)

    # estimate machines accuracies
    machines_accuracy = test_machines_accuracy(filters_num, machines_params)
    # ensemble votes for each filter and item
    ensembled_votes = weighted_mv(votes_list, filters_num, items_num, machines_accuracy)
    items_labels, prob_in_list = classify_items(ensembled_votes, lr, filters_num, items_num)
    loss, fp_rate, fn_rate, recall, precision, f_beta = compute_metrics(items_labels, gt_values, lr, filters_num)
    return loss, fp_rate, fn_rate, recall, precision, f_beta

