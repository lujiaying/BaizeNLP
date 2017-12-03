# encoding: utf-8
# Author: Robert Guthrie

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import random
import numpy as np
import sys
import time
from common_conf import STATES_TO_IX, UNK_TAG, START_TAG, STOP_TAG
from HMM import get_word_states

PADDING_IDX = 0  # can not change
PADDING_TAG = "<PAD>"
torch.manual_seed(1)

def to_scalar(var):
    # returns a python float
    return var.view(-1).data.tolist()[0]

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim, PADDING_IDX)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)),
                autograd.Variable(torch.randn(2, 1, self.hidden_dim // 2)))

    def _forward_alg(self, feats, lstm_out_lens):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.Tensor(feats.size()[1], self.tagset_size).fill_(-10000.)
        # START_TAG has all of the score.
        index = torch.LongTensor([self.tag_to_ix[START_TAG]])
        init_alphas.index_fill_(1, index, 0)

        # Wrap in a variable so that we will get automatic backprop
        forward_var = autograd.Variable(init_alphas)

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward variables at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(
                    1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence, is_train=False, sentence_len=[]):
        if not is_train:
            self.hidden = self.init_hidden()
            embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
            lstm_out, self.hidden = self.lstm(embeds, self.hidden)
            lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
            lstm_feats = self.hidden2tag(lstm_out)
            return lstm_feats
        else:
            self.hidden = self.init_hidden()
            embeds = self.word_embeds(sentence).view(len(sentence[0]), len(sentence), -1)
            embeds_p = torch.nn.utils.rnn.pack_padded_sequence(embeds, sentence_len)
            lstm_out_p, self.hidden = self.lstm(embeds_p, None)
            lstm_out, lstm_out_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out_p)
            lstm_feats = self.hidden2tag(lstm_out)
            print('lstm_feats size:', lstm_feats.size())
            return lstm_feats, lstm_out_lens

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = autograd.Variable(torch.Tensor([0]))
        tags = torch.cat([torch.LongTensor([self.tag_to_ix[START_TAG]]), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        #TODO: implement CRF layer as a class
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.Tensor(1, self.tagset_size).fill_(-10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = autograd.Variable(init_vvars)
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id])
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence_batch, tags_batch, raw_len_batch):
        feats, lstm_out_lens = self._get_lstm_features(sentence_batch, True, raw_len_batch)
        forward_score = self._forward_alg(feats, lstm_out_lens)
        gold_score = self._score_sentence(feats, tags_batch)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

def prepare_train_data(corpus_file, sep):
    sentence_list = []
    word_state_list = []
    word_to_ix = {PADDING_TAG:PADDING_IDX}

    with open(corpus_file) as fopen:
        for line in fopen:
            word_state = []
            line_list = line.strip().split(sep)
            for word in line_list:
                word_state += get_word_states(word)
            if len(word_state) <= 0:
                continue
            line_str = ''.join(line_list)
            sentence_list.append(list(line_str))
            word_state_list.append(word_state)

    for sentence in sentence_list:
        for word in sentence:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    word_to_ix[UNK_TAG] = len(word_to_ix)
    return word_to_ix, sentence_list, word_state_list

def do_pack(sentence_batch, tags_batch, word_to_ix, tag_to_ix):
    raw_len_batch = np.array([len(_) for _ in sentence_batch])

    # add packing elem
    max_len = np.max(raw_len_batch)
    sentence_batch_packed = []
    tags_batch_packed = []
    for idx in range(len(sentence_batch)):
        sentence = [word_to_ix[_] for _ in sentence_batch[idx]]
        if len(sentence) < max_len:
            sentence += [PADDING_IDX] * (max_len-len(sentence))
        sentence_batch_packed.append(sentence)
        tags = [tag_to_ix[_] for _ in tags_batch[idx]]
        if len(tags) < max_len:
            tags += [PADDING_IDX] * (max_len-len(tags))
        tags_batch_packed.append(tags)
    sentence_batch_packed = np.array(sentence_batch_packed)
    tags_batch_packed = np.array(tags_batch_packed)

    # sort by length
    sorted_idx_list = np.argsort(-raw_len_batch)
    sentence_batch_packed = sentence_batch_packed[sorted_idx_list]
    tags_batch_packed = tags_batch_packed[sorted_idx_list]
    raw_len_batch = raw_len_batch[sorted_idx_list]

    sentence_batch_packed = autograd.Variable(torch.LongTensor(sentence_batch_packed))
    tags_batch_packed = torch.LongTensor(tags_batch_packed)
    return sentence_batch_packed, tags_batch_packed, raw_len_batch

if __name__ == '__main__':
    EMBEDDING_DIM = 64
    HIDDEN_DIM = 50
    batch_size = 128

    # Make up some training data
    word_to_ix, sentence_list, word_state_list = prepare_train_data('./icwb2-data/training/pku_training.utf8', '  ')
    tag_to_ix = STATES_TO_IX
    print("Train Set Size:%d" % (len(sentence_list)))

    model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
    print(model)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(10):
        print("[%s] Epoch#%d:" % (time.ctime(), epoch))
        loss_list = []
        idx_shuffled = list(range(len(sentence_list)))
        random.shuffle(idx_shuffled)
        sentence_batch = []
        tags_batch = []
        batch_count = 0
        for idx_m in idx_shuffled:
            batch_count += 1
            sentence_batch.append(sentence_list[idx_m])
            tags_batch.append(word_state_list[idx_m])
            if batch_count % batch_size == 0:  #TODO: last batch maybe less than batch_size
                sentence_batch_packed, tags_batch_packed, raw_len_batch = do_pack(sentence_batch, tags_batch, word_to_ix, tag_to_ix)
                neg_log_likelihood = model.neg_log_likelihood(sentence_batch_packed, tags_batch_packed, raw_len_batch)
                sentence_batch = []
                tags_batch = []
            

            """
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is,
            # turn them into Variables of word indices.
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = torch.LongTensor([tag_to_ix[t] for t in tags])

            # Step 3. Run our forward pass.
            neg_log_likelihood = model.neg_log_likelihood(sentence_in, targets)

            # Step 4. Compute the loss, gradients, and update the parameters by
            # calling optimizer.step()
            neg_log_likelihood.backward()
            optimizer.step()

            loss_list.append(neg_log_likelihood.data[0])
            if idx_m % print_per_batch == 0:
                sys.stdout.write('\rbatch #%d, train loss:%s' % (idx_m, np.mean(loss_list)))
                sys.stdout.flush()
                loss_list = []
            """

        # store params
        torch.save(model.state_dict(), './models/BiLSTM_RNN_params_%d.pkl' % (epoch))
