# -*- encoding: utf-8 -*-
# Author: lujiaying93@foxmail.com
# Algorithm source from: http://www.matrix67.com/blog/archives/5044

from __future__ import division
import os
import sys
cur_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append("%s/../" % (cur_dir))

import math
import re
import time
from collections import defaultdict
from worddiscovery.trie import CharTrie

import logging
log_console = logging.StreamHandler(sys.stderr)
default_logger = logging.getLogger(__name__)
default_logger.setLevel(logging.DEBUG)
default_logger.addHandler(log_console)

MAX_INT = 9223372036854775807
RE_SENTENCE_SEPERATOR = r'[,.:;?!\~\-_()[\]<>，、。：；？！~-——（）【】《》\s]\s*'
RE_PUNCTUATION_TO_CLEAN = r'[＃＊＝＋/｜‘’“”￥#*=+\\|\'"^$%`]'


class EntropyBasedWorddiscovery(object):
    def __init__(self, word_max_len=5):
        self._trie = CharTrie()
        self._trie_reversed = CharTrie()  # for left char entropy calculate
        self.word_max_len = word_max_len
        self._word_info = defaultdict(dict)

        self.WORD_MIN_LEN = 2
        self.WORD_MIN_FREQ = 2
        self.WORD_MIN_PMI = 4
        self.WORD_MIN_NEIGHBOR_ENTROPY = 0

    def parse_file(self, file_name, debug=False):
        with open(file_name) as fopen:
            document_text = fopen.read()
            self.parse(document_text, debug)

    def parse(self, document_text, debug=False):
        sentences = self._preprocess(document_text)
        self._build_trie(sentences)
        self.cal_aggregation(debug)
        self.cal_neighbor_char_entropy(debug)
        self.cal_score()

    def get_new_words(self, top=20):
        default_logger.debug("Start sorting to get new words...")
        start_t = time.time()
        sorted_word_info = sorted(self._word_info.items(), key=lambda _: _[1]['score_freq'], reverse=True)
        default_logger.debug("Get new words, which cost %.3f seconds" % (time.time()-start_t))
        top_new_words = [_[0] for _ in sorted_word_info[:top]]
        return top_new_words

    def cal_aggregation(self, debug):
        default_logger.debug("Calculating word internal aggregation score...")
        start_t = time.time()
        for word, count in self._trie.get_all_words():
            if len(word) < self.WORD_MIN_LEN or count < self.WORD_MIN_FREQ:
                continue
            pmi = self._cal_word_aggregation(word, count)
            if debug:
                self._word_info[word]['aggreg'] = self._cal_word_aggregation(word, count)
            else:
                if pmi > self.WORD_MIN_PMI:
                    self._word_info[word]['aggreg'] = self._cal_word_aggregation(word, count)
        default_logger.debug("Internal aggregation has been calculated succesfully, which costs %.3f seconds" % (time.time()-start_t))

    def cal_neighbor_char_entropy(self, debug):
        default_logger.debug("Calculating word neighbor entropy score...")
        start_t = time.time()
        for word, count in self._trie.get_all_words():
            if len(word) < self.WORD_MIN_LEN or count < self.WORD_MIN_FREQ:
                continue
            if not debug:
                if word not in self._word_info:  # to speed up
                    continue
            rc_entropy = self._cal_word_neighbor_char_entropy(self._trie, word)
            if not debug:
                if rc_entropy <= self.WORD_MIN_NEIGHBOR_ENTROPY:   # to speed up
                    self._word_info.pop(word)
                    continue
            lc_entropy = self._cal_word_neighbor_char_entropy(self._trie_reversed, word[::-1])
            neighbor_entropy = min(rc_entropy, lc_entropy)
            if debug:
                self._word_info[word]['nbr_entropy'] = neighbor_entropy
                self._word_info[word]['rc_entropy'] = rc_entropy
                self._word_info[word]['lc_entropy'] = lc_entropy
            else:
                if neighbor_entropy > self.WORD_MIN_NEIGHBOR_ENTROPY:
                    self._word_info[word]['nbr_entropy'] = neighbor_entropy
                else:
                    self._word_info.pop(word)
        default_logger.debug("Neighbor entropy has been calculated succesfully, which costs %.3f seconds" % (time.time()-start_t))

    def cal_score(self):
        for word, d in self._word_info.items():
            self._word_info[word]['score'] = d['aggreg'] + d['nbr_entropy']
            self._word_info[word]['score_freq'] = d['score'] * self._trie.find(word)

    def _build_trie(self, sentences):
        self._trie.clear()
        self._trie_reversed.clear()
        default_logger.debug("Building trie tree...")
        start_t = time.time()
        for s in sentences:
            for n_grams in range(1, min(self.word_max_len+1, len(s)) + 1):
                if len(s) <= n_grams:
                    self._trie.insert(s)
                    self._trie_reversed.insert(s[::-1])
                else:
                    for end_pos in range(n_grams, len(s) + 1):
                        self._trie.insert(s[end_pos-n_grams:end_pos])
                        self._trie_reversed.insert(s[end_pos-n_grams:end_pos][::-1])
        default_logger.debug("Trie tree has been built succesfully, which costs %.3f seconds" % (time.time()-start_t))

    def _preprocess(self, document_text):
        global RE_SENTENCE_SEPERATOR
        global RE_PUNCTUATION_TO_CLEAN
        # split to sentence
        sentences = re.split(RE_SENTENCE_SEPERATOR, document_text)
        # clean
        sentences_clean = []
        for s in sentences:
            s = re.sub(RE_PUNCTUATION_TO_CLEAN, '', s)
            if not s:
                continue
            sentences_clean.append(s)
        return sentences_clean

    def _cal_word_aggregation(self, word, word_count):
        min_aggregation = MAX_INT
        for frag1, frag2 in self._generate_word_fragment(word):
            frag1_count = self._trie.find(frag1)
            frag2_count = self._trie.find(frag2)
            aggregation = word_count * self._trie.total_word_count / frag1_count / frag2_count
            min_aggregation = min(min_aggregation, aggregation)
        return math.log2(min_aggregation)

    def _generate_word_fragment(self, word):
        for pos in range(1, len(word)):
            yield (word[0:pos], word[pos:len(word)])

    def _cal_word_neighbor_char_entropy(self, trie_tree, word):
        children_count_list = []
        for char, char_count in trie_tree.get_children_char_count(word):
            children_count_list.append(char_count)
        total_word_count = sum(children_count_list)
        entropy = sum(map(lambda c: -(c/total_word_count)*math.log2(c/total_word_count), children_count_list))
        return entropy

if __name__ == '__main__':
    discover = EntropyBasedWorddiscovery()
    
    discover.parse("""
    每天都有网友问我：2017年做淘宝客还赚钱吗？我：2017年做淘宝客还可以继续好好做。各大门户虽然也跟我们小站长共分一杯羹，但是毕竟我们可以推广的商品太多了，现在网民购物的也越来越多了，所以淘宝客依然还有很大的发展空间。至少未来两三年内淘宝客大格局估计不会有太大变化。所以就淘宝客赚钱的这一话题，谈谈自己的一些看法。
    纵观这两年的所有网上兼职的工作，淘宝客算的上是最给力的，是最适合个人站长操作的项目，它实现了淘宝、网店商家、个人站长（淘宝客）三方共赢的良好局面，就连各大门户现在也在操作淘宝客。
    但很多人都在说淘客赚不到钱了，为什么做了那么多淘宝客的网站，最后赚钱的就一个呢？让我来跟大家分析一下原因。
    """, debug=False)

    #discover.parse("四是四，十是十；十四是十四，四十是四十~来自《绕口令大全》。")

    #print(discover._trie.find('四'))
    #print(discover._trie.find('四十'))
    #print(discover._trie.find('四十是四十'))
    #print(discover._trie.find('来自'))
    #print(discover._trie_reversed.find('四'))
    #print(discover._trie_reversed.find('四十'))
    #print(discover._trie_reversed.find('自来'))

    #for word, count in discover._trie.get_all_words():
    #    print(word, count)
    #for node, prefix in discover._trie.traverse():
    #    print(node, prefix)

    #print(discover._trie.get_children_char_count('口令'))
    #print(discover._trie_reversed.get_children_char_count('令口'))

    #discover.parse_file('./xiyouji.txt')
    #discover.parse_file('./xijinping.txt')
    print('\n'.join(discover.get_new_words(30)))
