#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from collections import OrderedDict


# class ConcatenatedDoc2Vec(object):
#     """
#     Concatenation of multiple models for reproducing the Paragraph Vectors paper.
#     Models must have exactly-matching vocabulary and document IDs. (Models should
#     be trained separately; this wrapper just returns concatenated results.)
#     """
#     def __init__(self, models):
#         self.models = models
#         if hasattr(models[0], 'docvecs'):
#             self.docvecs = ConcatenatedDocvecs([model.docvecs for model in models])

#     def __getitem__(self, token):
#         return np.concatenate([model[token] for model in self.models])

#     def infer_vector(self, document, alpha=0.1, min_alpha=0.0001, steps=5):
#         return np.concatenate([model.infer_vector(document, alpha, min_alpha, steps) for model in self.models])

#     def train(self, ignored):
#         pass  # train subcomponents individually

#     def build_vocab(self):
#         pass 


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # # check and process input arguments

    # if len(sys.argv) < 3:
    #     print globals()['__doc__'] % locals()
    #     sys.exit(1)
    # inp, outp = sys.argv[1:3]
    inp = '../small_wiki_subset/small_wiki_subset.en.text'



    # Variations #########
    simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
        Doc2Vec(dm=1, docvecs_mapfile="small_wiki_subset.docvecs_map.dmc", dm_concat=1, size=100, window=5, negative=5, hs=0, min_count=2, workers=multiprocessing.cpu_count()),
    # PV-DBOW 
        Doc2Vec(dm=0, docvecs_mapfile="small_wiki_subset.docvecs_map.dbow", size=100, negative=5, hs=0, min_count=2, workers=multiprocessing.cpu_count()),
    # PV-DM w/average
        Doc2Vec(dm=1, docvecs_mapfile="small_wiki_subset.docvecs_map.dmm", dm_mean=1, size=100, window=10, negative=5, hs=0, min_count=2, workers=multiprocessing.cpu_count()),
    ]

    alldocs = TaggedLineDocument(inp)
    # speed setup by sharing results of 1st model's vocabulary scan
    # simple_models[0].build_vocab(alldocs)  # PV-DM/concat requires one special NULL word so it serves as template
    # print(simple_models[0])
    # for model in simple_models[1:]:
    #     model.build_vocab(alldocs)
    # print(model)

    models_by_name = OrderedDict((str(model), model) for model in simple_models)

    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])

    print models_by_name.keys()

    #Train all 5 models 
    model_indx = 1
    for name, train_model in models_by_name.items():
        print name
        if model_indx is not 4 and model_indx is not 5:
            train_model.build_vocab(alldocs)
        train_model.train(alldocs)
        train_model.init_sims(replace=True)
        train_model.save('small_wiki_subset.' + str(model_indx) + '.model')
        model_indx = model_indx + 1





    #model = Doc2Vec(TaggedLineDocument(inp), docvecs_mapfile="small_wiki_subset.docvecs_map", size=400, window=5, min_count=2, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    #model.init_sims(replace=True)

    #model.save(outp)
