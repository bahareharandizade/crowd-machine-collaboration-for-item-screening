import numpy as np
from scipy.special import binom


def assign_criteria(papers_ids, criteria_num, values_count, power_cr_list, acc_cr_list, GT, prior_prob_in=None):
    cr_assigned = []
    papers_ids_new = []
    cr_list = range(criteria_num)
    expert_papers = []
    for p_id in papers_ids:
        p_classify = []
        n_min_list = []
        joint_prob_votes_out = [1., 1., 1., 1.]
        for cr in cr_list:
            acc_cr = acc_cr_list[cr]
            if prior_prob_in != None:
                p_paper_out = 1 - prior_prob_in[p_id * criteria_num + cr]
            else:
                p_paper_out = power_cr_list[cr]
            cr_count = values_count[p_id * criteria_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]
            for n in range(1, 11):
                # new value is out
                p_vote_out = acc_cr * p_paper_out + (1 - acc_cr) * (1 - p_paper_out)
                joint_prob_votes_out[cr] *= p_vote_out
                term1_p_out = binom(in_c+out_c+n, out_c+n)*acc_cr**(out_c+n)*(1-acc_cr)**in_c*p_paper_out
                term1_p_in = binom(in_c+out_c+n, in_c)*acc_cr**in_c*(1-acc_cr)**(out_c+n)*(1-p_paper_out)
                p_paper_in_vote_out = term1_p_in * p_vote_out / (term1_p_out + term1_p_in)
                p_paper_out = 1 - p_paper_in_vote_out
                if p_paper_out >= 0.99:
                    p_classify.append(joint_prob_votes_out[cr]/n)
                    n_min_list.append(n)
                    break
                elif n == 10:
                    p_classify.append(joint_prob_votes_out[cr]/n)
                    n_min_list.append(n)
        cr_assign = p_classify.index(max(p_classify))
        n_min = n_min_list[cr_assign]
        joint_prob = joint_prob_votes_out[cr_assign]

        # check stopping condition
        if n_min / joint_prob >= 20:
            if sum([GT[p_id * criteria_num + e_paper_id] for e_paper_id in range(criteria_num)]):
                expert_papers.append((p_id, 0))
            else:
                expert_papers.append((p_id, 1))
        else:
            cr_assigned.append(cr_assign)
            papers_ids_new.append(p_id)
    return cr_assigned, expert_papers, papers_ids_new


def classify_papers_baseline(papers_ids, criteria_num, values_prob, lr):
    classified_papers = []
    classified_papers_ids = []
    rest_papers_ids = []
    # trsh = lr / (lr + 1.)
    # TO DO !!!
    trsh = 0.99
    for paper_id in papers_ids:
        p_inclusion = 1.
        for e_paper_id in range(criteria_num):
            p_inclusion *= values_prob[paper_id * criteria_num + e_paper_id][0]
        p_exclusion = 1 - p_inclusion

        if p_exclusion > trsh:
            classified_papers.append(0)
            classified_papers_ids.append(paper_id)
        elif p_inclusion > trsh:
            classified_papers.append(1)
            classified_papers_ids.append(paper_id)
        else:
            rest_papers_ids.append(paper_id)
    return dict(zip(classified_papers_ids, classified_papers)), rest_papers_ids


def classify_papers(papers_ids, criteria_num, values_count, p_thrs, acc_cr_list, power_cr_list, prior_prob_in=None):
    classified_papers = []
    classified_papers_ids = []
    rest_papers_ids = []

    for p_id in papers_ids:
        p_inclusion = 1.
        for cr in range(criteria_num):
            acc_cr = acc_cr_list[cr]
            # power_cr = power_cr_list[cr]
            cr_count = values_count[p_id * criteria_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]
            if prior_prob_in != None:
                p_paper_out = 1 - prior_prob_in[p_id * criteria_num + cr]
            else:
                p_paper_out = power_cr_list[cr]

            if in_c == 0 and out_c == 0:
                prob_cr_in = 1 - p_paper_out
            else:
                prop_p_in = binom(in_c+out_c, in_c)*acc_cr**in_c*(1-acc_cr)**out_c*(1-p_paper_out)
                prop_p_out = binom(in_c+out_c, out_c)*acc_cr**out_c*(1-acc_cr)**in_c*p_paper_out
                prob_cr_in = prop_p_in / (prop_p_in + prop_p_out)
            p_inclusion *= prob_cr_in
        p_exclusion = 1 - p_inclusion

        if p_exclusion > p_thrs:
            classified_papers.append(0)
            classified_papers_ids.append(p_id)
        elif p_inclusion > p_thrs:
            classified_papers.append(1)
            classified_papers_ids.append(p_id)
        else:
            rest_papers_ids.append(p_id)
    return dict(zip(classified_papers_ids, classified_papers)), rest_papers_ids


def generate_responses(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
    responses = []
    n = len(papers_ids)
    workers_n = 1 if n < papers_worker else n // papers_worker
    for w_ind in range(workers_n):
        worker_acc_in = acc[1].pop()
        acc[1].insert(0, worker_acc_in)
        worker_acc_out = acc[0].pop()
        acc[0].insert(0, worker_acc_out)
        for cr, p_id in zip(cr_assigned[w_ind*papers_worker: w_ind*papers_worker+papers_worker],
                            papers_ids[w_ind*papers_worker: w_ind*papers_worker+papers_worker]):
            cr_vals_id = range(p_id * criteria_num, p_id * criteria_num + criteria_num, 1)
            isPaperIN = sum([GT[i] for i in cr_vals_id]) == 0
            if isPaperIN:
                worker_acc = worker_acc_in
            else:
                worker_acc = worker_acc_out

            GT_cr = GT[p_id * criteria_num + cr]
            cr_dif = criteria_difficulty[cr]
            if np.random.binomial(1, worker_acc * cr_dif if worker_acc * cr_dif <= 1. else 1.):
                vote = GT_cr
            else:
                vote = 1 - GT_cr
            responses.append(vote)
    return responses


# # a low accurate criteria
# def generate_responses(GT, papers_ids, criteria_num, papers_worker, acc, criteria_difficulty, cr_assigned):
#     responses = []
#     n = len(papers_ids)
#     workers_n = 1 if n < papers_worker else n // papers_worker
#     for w_ind in range(workers_n):
#         worker_acc_in = acc[1].pop()
#         acc[1].insert(0, worker_acc_in)
#         worker_acc_out = acc[0].pop()
#         acc[0].insert(0, worker_acc_out)
#         for cr, p_id in zip(cr_assigned[w_ind*papers_worker: w_ind*papers_worker+papers_worker],
#                             papers_ids[w_ind*papers_worker: w_ind*papers_worker+papers_worker]):
#             cr_vals_id = range(p_id * criteria_num, p_id * criteria_num + criteria_num, 1)
#             isPaperIN = sum([GT[i] for i in cr_vals_id]) == 0
#             if isPaperIN:
#                 worker_acc = worker_acc_in
#             else:
#                 worker_acc = worker_acc_out
#
#             GT_cr = GT[p_id * criteria_num + cr]
#             cr_dif = criteria_difficulty[cr]
#             if cr_dif < 1.:
#                 if np.random.binomial(1, 0.55):
#                     vote = GT_cr
#                 else:
#                     vote = 1 - GT_cr
#             else:
#                 if np.random.binomial(1, worker_acc * cr_dif if worker_acc * cr_dif <= 1. else 1.):
#                     vote = GT_cr
#                 else:
#                     vote = 1 - GT_cr
#             responses.append(vote)
#     return responses


def update_v_count(values_count, criteria_num, cr_assigned, responses, p_ids):
    for cr, vote, p_id in zip(cr_assigned, responses, p_ids):
        if vote:
            values_count[p_id * criteria_num + cr][1] += 1
        else:
            values_count[p_id * criteria_num + cr][0] += 1


def update_cr_power(n_papers, criteria_num, acc_cr_list, power_cr_list, values_count):
    power_cr_new = []
    apply_criteria_list = [[] for _ in range(criteria_num)]
    for p_id in range(n_papers):
        for cr in range(criteria_num):
            acc_cr = acc_cr_list[cr]
            power_cr = power_cr_list[cr]
            cr_count = values_count[p_id * criteria_num + cr]
            in_c = cr_count[0]
            out_c = cr_count[1]
            prop_p_in = binom(in_c+out_c, in_c)*acc_cr**in_c*(1-acc_cr)**out_c*(1-power_cr)
            prop_p_out = binom(in_c+out_c, out_c)*acc_cr**out_c*(1-acc_cr)**in_c*power_cr
            prob_cr_out = prop_p_out / (prop_p_in + prop_p_out)
            apply_criteria_list[cr].append(prob_cr_out)
    for probs in apply_criteria_list:
        power_cr_new.append(np.mean(probs))
    return power_cr_new
