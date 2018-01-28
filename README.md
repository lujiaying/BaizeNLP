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
