import numpy as np


class LineParser:
    def __init__(self, parsers):
        self.parsers = sorted(parsers.items())

    def __call__(self, line):
        return [p(line[i]) for i, p in self.parsers]


class DocumentParser:

    def __init__(self, *func):
        self.func = func

    def __call__(self, doc):
        for f in self.func:
            doc = f(doc)
        return doc


def parse_document(sep1='#N#', sep2=' '):
    def f(doc):
        return [list(map(int, d.split(sep2))) for d in doc.split(sep1)]

    return f


def filter_document(vocab):
    def f(doc):
        return [[x for x in d if x <= vocab] for d in doc]

    return f


def pad_document(size, length):
    def f(doc):
        result = np.zeros((size, length))

        i = 0
        for d in doc:
            if d:
                for j, x in enumerate(d):
                    if j == length:
                        break
                    result[i, j] = x
                i += 1
                if i == size:
                    break

        return result

    return f


def bow_document(vocab):
    def f(doc):
        result = np.zeros((vocab + 1,))
        for d in doc:
            for word in d:
                if 0 < word <= vocab:
                    result[word] += 1
        return result

    return f


def flat_document(size):
    def f(doc):
        result = np.zeros((size,))
        i = 0
        for d in doc:
            for word in d:
                if word > 0:
                    result[i] = word
                    i += 1
                    if i == size:
                        return result
        return result

    return f


def kv_parse_document(deliminator=':', sep=' '):
    def f(doc):
        k, v = zip(*[(int(k), int(v)) for k, v in [d.split(deliminator) for d in doc.split(sep)]])
        return k, v

    return f


def kv_pad_document(size):
    def f(doc):
        k, v = doc
        rk = np.zeros((size,))
        rv = np.zeros((size,))
        for ik, iv, i in zip(k, v, range(size)):
            rk[i] = ik
            rv[i] = ik
        return rk, rv

    return f


def clause(split_tokens):
    def f(docs):
        res = []
        for doc in docs:
            sents = []
            lst = -1
            
            for i,token in enumerate(doc):
                if token in split_tokens:
                    sents.append(doc[lst+1:i+1])
                    lst = i
                    
            if lst != i:
                sents.append(doc[lst+1:])
            
            res.append(sents)
        return res
    return f

def pad_docs(sent_cnt,sent_len):
    def f(docs):
        rr = []
        for doc in docs:
            result = np.zeros((sent_cnt, sent_len),dtype='int32')

            i = 0
            for do in doc:
                if len(do)>sent_len:
                    result[i] = do[:sent_len]
                else:
                    result[i,:len(do)] = do
                i += 1
                if i == sent_cnt:
                    break
    
            rr.append(result)
        return np.array(rr)
    return f
    