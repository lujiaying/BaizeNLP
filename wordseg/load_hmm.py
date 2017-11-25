# -*- encoding: utf-8 -*-

from __future__ import division
from collections import defaultdict

STATES = ('B', 'E', 'M', 'S')
UNK = '<UNK>'
PROB_MIN = 3.14e-100

def get_word_states(word):
    """
    Returns:
        word_states: list, ['B', 'E']
    """
    #TODO: support python2.x
    word_states = []
    if len(word) == 1:
        word_states.append('S')
    else:
        for idx, c in enumerate(word):
            if idx == 0:
                word_states.append('B')
            elif idx == len(word) - 1:
                word_states.append('E')
            else:
                word_states.append('M')
    return word_states

def load_segment_corpus(corpus_file, sep):
    """
    Args:
        corpus_file: string, file name
        sep: string, separator for words in corpus

    Returns:
        start_prob: dict of float, {'B': 0.5, 'E': 0.0, 'M': 0.0, 'S': 0.5}
        transition_prob: dict of dict, {'B': {'B': 0.1, 'E':0.9}, 'E':{...}, ...}
        emission_prob: dict of dict, {'B': {'中国': 0.002, ...}, ...}
    """
    global STATES

    start_counter = defaultdict(int)
    transition_counter = defaultdict(lambda: defaultdict(int))
    emission_counter = defaultdict(lambda: defaultdict(int))
    with open(corpus_file) as fopen:
        for line in fopen:
            word_states = []
            line_list = line.strip().split(sep)
            line_str = ''.join(line_list)
            for word in line_list:
                word_states += get_word_states(word)
            if len(word_states) <= 0:
                continue
            for idx in range(len(word_states)):
                # compute start counter
                if idx == 0:
                    start_counter[word_states[0]] += 1
                # compute transition counter
                if idx >= 1:
                    transition_counter[word_states[idx-1]][word_states[idx]] += 1
                # compute emission counter
                emission_counter[word_states[idx]][line_str[idx]] += 1

    start_prob = {}
    start_state_sum = sum(start_counter.values())
    for k, v in start_counter.items():
        start_prob[k] = v / start_state_sum
    for s in STATES:
        if s not in start_prob:
            start_prob[s] = PROB_MIN

    transition_prob = {}
    for k, d in transition_counter.items():
        k_sum = sum(d.values())
        transition_prob[k] = {}
        for _k, v in d.items():
            transition_prob[k][_k] = v / k_sum
        for s in STATES:
            if s not in transition_prob[k]:
                transition_prob[k][s] = PROB_MIN

    emission_prob = {}
    # Good-Turing: reallocate the probability mass of n-grams that were seen once to 
    # the n-grams that were never seen
    for k, d in emission_counter.items():
        raw_k_sum = sum(d.values())
        n_1 = len([_ for _ in d.values() if _ == 1])  # num of events with freq 1
        k_sum = raw_k_sum + 1 * n_1
        emission_prob[k] = {}
        for _k, v in d.items():
            emission_prob[k][_k] = v / k_sum
        emission_prob[k][UNK] = n_1 / k_sum
    return start_prob, transition_prob, emission_prob

if __name__ == '__main__':
    start_prob, transition_prob, emission_prob = load_segment_corpus('./icwb2-data/training/pku_training.utf8', '  ')
    print('start_prob: ', start_prob)
    print('trans_prob: ', transition_prob)
    #print(emission_prob['B'][UNK], emission_prob['S'][UNK])
