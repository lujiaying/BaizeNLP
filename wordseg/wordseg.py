from load_hmm import load_segment_corpus, STATES
from viterbi import do_viterbi

start_prob, transition_prob, emission_prob = load_segment_corpus('./icwb2-data/training/pku_training.utf8', '  ')
print('Load corpus done')

def _convert_path2segments(query, path):
    segments = []
    seg = ''

    for idx in range(len(path)):
        seg += query[idx]

        if path[idx] == 'S' or path[idx] == 'E':
            segments.append(seg)
            seg = ''
    return segments

def do_wordseg(query):
    prob, path = do_viterbi(query, STATES, start_prob, transition_prob, emission_prob)
    segments = _convert_path2segments(query, path)
    return segments
