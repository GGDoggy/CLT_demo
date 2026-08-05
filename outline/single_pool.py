import numpy as np

# pull -> batch -> character
def one_character(batch_count, pulls_per_batch, single_pull_prob):
    fail_prob = 1 - single_pull_prob
    tobatch_prob = np.zeros(batch_count + 1)
    for i in range(1, batch_count + 1):
        pull_count = i * pulls_per_batch
        tobatch_prob[i] = 1 - fail_prob ** pull_count
    for i in range(batch_count):
        tobatch_prob[batch_count - i] -= tobatch_prob[batch_count - i - 1]

    return tobatch_prob

def one_character_loop(free_batch, total_batch, current_prob, character_remain, result_prob, base_prob, batch_count):
    new_character_remain = character_remain - 1
    if free_batch == batch_count:  # having exactly enough pulls to exchange character
        if new_character_remain == 0:
            result_prob[total_batch] += current_prob
        else:
            one_character_loop(0, total_batch, current_prob, new_character_remain, result_prob, base_prob, batch_count)
        return

    remain_prob = 1
    for i in range(1, batch_count + 1):
        new_free_batch = free_batch + i
        new_total_batch = total_batch + i
        new_prob = current_prob * base_prob[i]  # pick the character
        remain_prob -= base_prob[i]
        if new_character_remain == 0:
            result_prob[new_total_batch] += new_prob
        else:
            one_character_loop(new_free_batch, new_total_batch, new_prob, new_character_remain, result_prob, base_prob, batch_count)
        if new_free_batch == batch_count:  # having enough pulls during one character
            new_prob = current_prob * remain_prob
            if new_character_remain == 0:
                result_prob[new_total_batch] += new_prob
            else:
                one_character_loop(0, new_total_batch, new_prob, new_character_remain, result_prob, base_prob, batch_count)
            break

    return


def multi_character(character_count, batch_count, pulls_per_batch, single_pull_prob):
    result_prob = np.zeros(character_count * batch_count + 1)
    base_prob = one_character(batch_count,pulls_per_batch, single_pull_prob)
    one_character_loop(0, 0, 1, character_count, result_prob, base_prob, batch_count)
    return result_prob