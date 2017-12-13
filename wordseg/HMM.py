# -*- encoding: utf-8 -*-

from __future__ import division
import math
from collections import defaultdict
import common_conf
import common_util

class HMM(object):

    PROB_MIN = 3.14e-100

    def __init__(self):
        self._start_prob = None
        self._transition_prob = None
        self._emission_prob = None

    def train(self, corpus_file, sep):
        """
        train model by segmentation corpus_file

        Args:
            corpus_file: string, file name
            sep: string, separator for words in corpus
        """
        self._start_prob, self._transition_prob, self._emission_prob = self._load_segment_corpus(corpus_file, sep)
        print('train corpus_file:%s done' % (corpus_file))

    def infer(self, sentence):
        """
        do inference for sentence

        Args:
            sentence: string
        Returns:
            prob: float
            hidden_states: list, ["B", "E", "S"]
        """
        return self._do_viterbi(sentence)

    def cut(self, sentence):
        """
        cut sentence

        Args:
            sentence: string
        Returns:
            segments: list of strings, elements are tokenized terms
        """
        prob, path = self._do_viterbi(sentence)
        return common_util.convert_path2segments(sentence, path)

    def _load_segment_corpus(self, corpus_file, sep):
        """
        Args:
            corpus_file: string, file name
            sep: string, separator for words in corpus

        Returns:
            start_prob: dict of float, {'B': 0.5, 'E': 0.0, 'M': 0.0, 'S': 0.5}
            transition_prob: dict of dict, {'B': {'B': 0.1, 'E':0.9}, 'E':{...}, ...}
            emission_prob: dict of dict, {'B': {'中国': 0.002, ...}, ...}
        """
        start_counter = defaultdict(int)
        transition_counter = defaultdict(lambda: defaultdict(int))
        emission_counter = defaultdict(lambda: defaultdict(int))
        with open(corpus_file) as fopen:
            for line in fopen:
                word_states = []
                line_list = line.strip().split(sep)
                line_str = ''.join(line_list)
                for word in line_list:
                    word_states += common_util.get_word_states(word)
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
        for s in common_conf.STATES:
            if s not in start_prob:
                start_prob[s] = HMM.PROB_MIN

        transition_prob = {}
        for k, d in transition_counter.items():
            k_sum = sum(d.values())
            transition_prob[k] = {}
            for _k, v in d.items():
                transition_prob[k][_k] = v / k_sum
            for s in common_conf.STATES:
                if s not in transition_prob[k]:
                    transition_prob[k][s] = HMM.PROB_MIN

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
            emission_prob[k][common_conf.UNK_TAG] = n_1 / k_sum
        return start_prob, transition_prob, emission_prob

    def _do_viterbi(self, sentence):
        """
        V_{1,k} = P(y_1|k) * pi_k
        V_{t,k} = P(y_t|k) * max(a_{x,k}, * V_{t-1,x})

        Args:
            sentence: string

        Returns:
            prob: float
            hidden_states: list, ["B", "E", "S"]
        """
        start_p = self._start_prob
        trans_p = self._transition_prob
        emission_p = self._emission_prob
        V = [{}]
        path = {}

        for s in common_conf.STATES:  # t == 0
            obs = sentence[0] if sentence[0] in emission_p[s] else common_conf.UNK_TAG
            V[0][s] = math.log(emission_p[s][obs] * start_p[s])
            path[s] = [s]

        for t in range(1, len(sentence)):
            V.append({})
            new_path = {}

            for s in common_conf.STATES:
                obs = sentence[t] if sentence[t] in emission_p[s] else common_conf.UNK_TAG
                (prob, state) = max([(math.log(emission_p[s][obs]) + math.log(trans_p[s_][s]) + V[t-1][s_], s_) for s_ in common_conf.STATES])
                V[t][s] = prob
                new_path[s] = path[state] + [s]
            path = new_path

        (prob, state) = max([(V[len(sentence)-1][s], s) for s in common_conf.STATES])
        return prob, path[state]

if __name__ == '__main__':
    sentence = '南京市长江大桥'
    hmm_ins = HMM()
    hmm_ins.train('./icwb2-data/training/pku_training.utf8', '  ')
    prob, path = hmm_ins.infer(sentence)
    print('sentence: %s, prob:%s, h-state:%s' % (sentence, prob, '->'.join(path)))
    print(hmm_ins.cut(sentence))
