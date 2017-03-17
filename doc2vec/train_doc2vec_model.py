#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import multiprocessing

from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp, outp = sys.argv[1:3]

    model = Doc2Vec(TaggedLineDocument(inp), docvecs_mapfile="small_wiki_subset.docvecs_map", size=400, window=5, min_count=5, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    model.init_sims(replace=True)

    model.save(outp)
