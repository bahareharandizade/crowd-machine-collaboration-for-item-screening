import itertools
import pandas as pd
'''
find best criteria order
'''

def get_loss(order, criteria_power, criteria_acc, CR):
    pfi = 0.
    criteria_num = len(criteria_power)
    vals_combinations = list(itertools.product([0, 1], repeat=criteria_num))
    for vals in vals_combinations:
        m = 1.
        for i in order:
            pow_i = criteria_power[i]
            val_i = vals[i]
            acc_i = criteria_acc[i]
            m *= pow_i**val_i * (1-pow_i)**(1-val_i) * acc_i**(1-val_i) * (1-acc_i)**val_i
        pfi += m

    pfe_0 = (1 - criteria_power[order[0]]) * (1 - criteria_acc[order[0]])
    pfe = pfe_0
    for i_ind, i in enumerate(order[1:], 1):
        m = 1.
        pfe_i = (1 - criteria_power[i]) * (1 - criteria_acc[i])
        for j in order[:i_ind]:
            pfi_j = criteria_power[j] * (1 - criteria_acc[j])
            pti_j = (1 - criteria_power[j]) * criteria_acc[j]
            pin_j = pfi_j + pti_j
            m *= pin_j
        pfe += pfe_i * m

    loss = CR * pfe + pfi
    return loss


def get_cost(order, criteria_power, criteria_acc):
    cost = 1.
    for i_ind, i in enumerate(order[:-1]):
        m = 1.
        for j in order[:i_ind+1]:
            pfi_j = criteria_power[j] * (1 - criteria_acc[j])
            pti_j = (1 - criteria_power[j]) * criteria_acc[j]
            pin_j = pfi_j + pti_j
            m *= pin_j
        cost += m
    return cost


def estimate_cr_order(criteria_power, criteria_acc):
    CR = 5
    # criteria_power = [0.14, 0.14, 0.28, 0.42]
    # criteria_acc = [0.6, 0.7, 0.8, 0.9]
    criteria_num = len(criteria_power)

    print '----------------------------------'
    for cr_id in range(criteria_num):
        print 'cr_id: {} | power: {} | cr_acc: {}'.format(cr_id, criteria_power[cr_id], criteria_acc[cr_id])
    print '----------------------------------'

    orders = itertools.permutations(range(criteria_num))
    data = []
    for order in orders:
        loss = get_loss(order, criteria_power, criteria_acc, CR)
        cost = get_cost(order, criteria_power, criteria_acc)
        print '{} | loss: {} | cost: {}'.format(order, loss, cost)
        data.append([order, loss, cost])
    pd.DataFrame(data, columns=['order', 'loss', 'cost']). \
        to_csv('output/data/loss_cost_cr.csv', index=False)
