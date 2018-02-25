# BaizeNLP

Baize is a mythical hooved chimerical creature known in Chinese, which knows every dialect and neologism of world.

## Word segmentation

BaizeNLP provides Chinese word segmentation function. The algorithm is based on a hybrid method of n-gram language model and HMM model.

## New word discovering

[Web Demo >>](https://baizenlp.leanapp.cn/baize/)

[More details for this project >>](http://lujiaying.github.io/2018/01/28/%E5%9F%BA%E4%BA%8E%E7%BB%9F%E8%AE%A1%E4%BF%A1%E6%81%AF%E7%9A%84%E6%96%B0%E8%AF%8D%E6%8C%96%E6%8E%98%E5%AE%9E%E8%B7%B5/)

### Usage

0. Download source code.

```
$ mkdir my_project && cd my_project
$ git clone https://github.com/lujiaying/BaizeNLP/tree/master/worddiscovery
```

1. Try the hello world example.
```python
# hello.py
# encoding: utf-8

from BaizeNLP.worddiscovery import entropy_based

discover = entropy_based.EntropyBasedWorddiscovery(word_max_len=4)
discover.parse("""
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的科学。因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。因而它是计算机科学的一部分。
自然语言处理（NLP）是计算机科学，人工智能，语言学关注计算机和人类（自然）语言之间的相互作用的领域。
""")

print('\n'.join(discover.get_new_words(10)))
```

-------------------------------------------

# BaizeNLP
白泽说人话，通万物之情，晓天下万物状貌。

## 分词

TODO: 添加说明


## 新词挖掘

[原理详情请参考 >>]()

[Web Demo >>](https://baizenlp.leanapp.cn/baize/)

### 使用说明

0. 下载源码

```
$ mkdir my_project && cd my_project
$ git clone https://github.com/lujiaying/BaizeNLP/tree/master/worddiscovery
```

1. 尝试编写一个简单的程序
```python
# hello.py
# encoding: utf-8

from BaizeNLP.worddiscovery import entropy_based

discover = entropy_based.EntropyBasedWorddiscovery(word_max_len=4)
discover.parse("""
    自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。自然语言处理是一门融语言学、计算机科学、数学于一体的科学。因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，所以它与语言学的研究有着密切的联系，但又有重要的区别。自然语言处理并不是一般地研究自然语言，而在于研制能有效地实现自然语言通信的计算机系统，特别是其中的软件系统。因而它是计算机科学的一部分。
自然语言处理（NLP）是计算机科学，人工智能，语言学关注计算机和人类（自然）语言之间的相互作用的领域。
""")

print('\n'.join(discover.get_new_words(10)))
```
