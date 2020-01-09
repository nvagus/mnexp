from .doc2vec import *


class Doc2VecSoftmax(Doc2Vec):
    def train_sample(self, docs_window, clicked_docs, pos, negs):
        clicked = docs_window.get_doc(clicked_docs)
        clicked_vertical = docs_window.get_vertical(clicked_docs)
        clicked_subvertical = docs_window.get_subvertical(clicked_docs)
        clicked = np.concatenate([clicked_vertical, clicked_subvertical, clicked], axis=-1)

        yield clicked, np.concatenate([self.docs[pos].vertical, self.docs[pos].subvertical, self.docs[pos].doc],
                                      axis=-1), 1
        for neg in negs:
            yield clicked, np.concatenate([self.docs[neg].vertical, self.docs[neg].subvertical, self.docs[neg].doc],
                                          axis=-1), 0
