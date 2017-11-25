import math
from load_hmm import UNK

def do_viterbi(query, states, start_p, trans_p, emission_p):
    """
    V_{1,k} = P(y_1|k) * pi_k
    V_{t,k} = P(y_t|k) * max(a_{x,k}, * V_{t-1,x})

    Returns:
        prob: float
        hidden_states: list, ["B", "E", "S"]
    """
    V = [{}]
    path = {}

    for s in states:  # t == 0
        obs = query[0] if query[0] in emission_p[s] else UNK
        V[0][s] = math.log(emission_p[s][obs] * start_p[s])
        path[s] = [s]

    for t in range(1, len(query)):
        V.append({})
        new_path = {}

        for s in states:
            obs = query[t] if query[t] in emission_p[s] else UNK
            (prob, state) = max([(math.log(emission_p[s][obs]) + math.log(trans_p[s_][s]) + V[t-1][s_], s_) for s_ in states])
            V[t][s] = prob
            new_path[s] = path[state] + [s]
        path = new_path

    (prob, state) = max([(V[len(query)-1][s], s) for s in states])
    return prob, path[state]


if __name__ == '__main__':
    from load_hmm import load_segment_corpus, STATES
    start_prob, transition_prob, emission_prob = load_segment_corpus('./icwb2-data/training/pku_training.utf8', '  ')

    query = '南京市长江大桥'
    prob, path = do_viterbi(query, STATES, start_prob, transition_prob, emission_prob)
    print('query: %s, prob:%s, h-state:%s' % (query, prob, '->'.join(path)))
    query = '今天是个晴天'
    prob, path = do_viterbi(query, STATES, start_prob, transition_prob, emission_prob)
    print('query: %s, prob:%s, h-state:%s' % (query, prob, '->'.join(path)))
