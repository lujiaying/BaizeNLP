import sys
import os
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % (cur_dir))
from wordseg import HMM

local_corpus_dir = '%s/icwb2-data' % (cur_dir)
train_file_path = '%s/training/pku_training.utf8' % (local_corpus_dir)
test_file_path = '%s/testing/pku_test.utf8' % (local_corpus_dir)
test_seg_res_path = '%s/testing/pku_test_seg.utf8' % (local_corpus_dir)
train_words_gold_path = '%s/gold/pku_training_words.utf8' % (local_corpus_dir)
test_gold_path = '%s/gold/pku_test_gold.utf8' % (local_corpus_dir)
score_bin_path = '%s/scripts/score' % (local_corpus_dir)
SEP = "  "

def download_sighan_corpus():
    global local_corpus_dir
    if os.path.exists(local_corpus_dir):
        print('sighan corpus already exist')
        return

    dowload_file_path = './icwb2-data.zip' 
    if not os.path.exists(dowload_file_path):
        cmd = 'wget -O %s http://sighan.cs.uchicago.edu/bakeoff2005/data/icwb2-data.zip' % (dowload_file_path)
        os.system(cmd)
    cmd = 'unzip %s' % (dowload_file_path)
    os.system(cmd)
    return 

def do_train_and_test():
    global train_file_path, test_file_path, test_seg_res_path
    global score_bin_path, train_words_gold_path, test_gold_path, cur_dir
    global SEP
    hmm_ins = HMM.HMM()
    hmm_ins.train(train_file_path, SEP)

    # case test
    sentence = '南京市长江大桥'
    prob, path = hmm_ins.infer(sentence)
    print('sentence: %s, prob:%s, h-state:%s' % (sentence, prob, '->'.join(path)))
    print(hmm_ins.cut(sentence))

    # generate test segments result
    with open(test_file_path) as fopen, open(test_seg_res_path, 'w') as fwrite:
        for line in fopen:
            sentence = line.strip()
            if sentence:
                terms = hmm_ins.cut(sentence)
            else:
                terms = []
            fwrite.write(' '.join(terms) + '\n')

    # generate score
    cmd = 'perl %s %s %s %s > %s/score.utf8' % (score_bin_path, train_words_gold_path, test_gold_path, test_seg_res_path, cur_dir)
    os.system(cmd)

if __name__ == '__main__':
    download_sighan_corpus()
    do_train_and_test()
