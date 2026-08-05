import numpy as np

import single_pool as single_pool

# probability seq. structure: [P(0), P(1), P(2) ... P(batch)]

def calc_plan_sequence(batch_count, pulls_per_batch, cost_per_pull, single_pull_prob, plan_sequence, predict_cost_sequence, mode, total_cost=0):
    prob_base_map = dict()
    for n in set(plan_sequence):
        if type(n) != int:
            print("Something in plan_sequence is not int.")
        prob_base_map[n] = single_pool.multi_character(n, batch_count, pulls_per_batch, single_pull_prob)
    if len(plan_sequence) != len(predict_cost_sequence):
        print("Length of plan and cost is not the same.")

    # using plan cost seq. to discard impossible pulls
    exact_prob = np.ones(1)
    cost_per_batch = pulls_per_batch * cost_per_pull
    for i in range(len(plan_sequence)):
        exact_prob = np.convolve(exact_prob, prob_base_map[plan_sequence[i]])
        max_cost = predict_cost_sequence[i]
        max_batch_num = int(max_cost / cost_per_batch)
        max_batch_num = min(max_batch_num, len(exact_prob) - 1)
        exact_prob[max_batch_num + 1:] = 0

    cumu_prob = np.cumsum(exact_prob)

    if mode == "spend":
        cost = np.linspace(0, (len(exact_prob) - 1) * pulls_per_batch * cost_per_pull, len(exact_prob))
        x_title = "Spend"
    elif mode == "remain":
        cost = total_cost - np.linspace(0, (len(exact_prob) - 1) * pulls_per_batch * cost_per_pull, len(exact_prob))
        x_title = "Remain"
    elif mode == "pull":
        cost = np.linspace(0, (len(exact_prob) - 1) * pulls_per_batch, len(exact_prob))
        x_title = "Pull Count"

    return cost, exact_prob, cumu_prob, x_title