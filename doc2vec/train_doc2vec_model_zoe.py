#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os.path
import sys
import scipy
import multiprocessing

from gensim.corpora import  WikiCorpus
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from collections import OrderedDict
from gensim.models.doc2vec import DocvecsArray
import gensim, logging, os
from gensim.corpora import Dictionary, HashDictionary, MmCorpus, WikiCorpus
from scipy import spatial
import numpy as np
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


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

dimension = 100

class MyArticles(object):
    def __init__(self, dirname):
        self.dirname = dirname
        self.dir_list = os.listdir(self.dirname)
        self.dir_list.sort()

    def articles(self):
        for fname in self.dir_list:
            file_as_string = open(os.path.join(self.dirname, fname)).read()
            yield file_as_string.split()

testing_articles = MyArticles('../testing_articles/articles')


def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    #
    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    index2word_set = set(model.wv.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set:
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    #
    # Divide the result by the number of words to get the average
    if nwords == 0 :
        return featureVec
    featureVec = np.divide(featureVec,nwords)
    return featureVec


def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate
    # the average feature vector for each one and return a 2D numpy array
    #
    # Initialize a counter
    counter = 0.
    #
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    #
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Article %d of %d" % (counter, len(reviews))
       #
       # Call the function (defined above) that makes average feature vectors
       # print counter
       # print type(counter)
       reviewFeatureVecs[int(counter)] = makeFeatureVec(review, model, num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs

def cosine_sim(d1, d2):
    scipy.spatial.distance.cosine(d1, d1)


def get_accuracy(model):
    # ****************************************************************
    # Calculate average feature vectors for testing set

    print "Creating average feature vecs for test articles..."

    clean_test_articles = []
    count = 0
    for article in testing_articles.articles():
        print count
        count = count + 1
        clean_test_articles.append(article)
    #testDataVecs = getAvgFeatureVecs(clean_test_articles, model, dimension)

    # testDataVecs now holds a 2D matrix of (len(reviews),num_features)
    # the average feature vector for each article!

    # Initialize DocVecs Map
    docvecs_map = DocvecsArray("big_wiki_subset.docvecs_map.doctag_syn0")

    ########### Accuracy Evaluation  #################
    print "Evaluating Accuracy..."
    # TODO: This needs to be a mapping from 172 index to index in directory structure {2: [23, 54, 106]}
    article_map = {0: [89, 219, 72], 1: [19, 76, 124], 2: [19, 76, 29], 3: [189, 248, 19], 4: [308, 76, 335], 5: [236, 426, 300], 6: [88, 306, 362], 7: [68, 189, 206], 8: [127, 126, 213], 9: [93, 65, 92], 10: [301, 425, 85], 11: [291, 234, 302], 12: [73, 105, 115], 13: [1, 115, 105], 14: [290, 302, 260], 15: [103, 297, 36], 16: [61, 291, 10], 17: [25, 130, 227], 18: [34, 25, 118], 19: [185, 230, 25], 20: [137, 111, 437], 21: [412, 111, 437], 22: [324, 437, 412], 23: [63, 318, 356], 24: [16, 326, 354], 25: [51, 266, 102], 26: [2, 264, 52], 27: [251, 282, 215], 28: [31, 341, 215], 29: [283, 214, 215], 30: [215, 327, 328], 31: [49, 278, 328], 32: [238, 282, 329], 33: [43, 330, 64], 34: [261, 215, 404], 35: [64, 215, 58], 36: [289, 106, 339], 37: [339, 339, 110], 38: [417, 360, 342], 39: [342, 407, 417], 40: [418, 421, 420], 41: [148, 233, 262], 42: [422, 416, 419], 43: [226, 112, 186], 44: [337, 271, 187], 45: [432, 374, 161], 46: [197, 176, 249], 47: [109, 117, 410], 48: [366, 268, 321], 49: [413, 32, 366], 50: [239, 415, 172], 51: [42, 42, 50], 52: [4, 415, 321], 53: [161, 359, 331], 54: [332, 109, 259], 55: [355, 371, 173], 56: [173, 132, 352], 57: [169, 168, 428], 58: [257, 401, 346], 59: [144, 345, 352], 60: [165, 75, 164], 61: [164, 164, 165], 62: [166, 163, 74], 63: [200, 44, 108], 64: [255, 24, 231], 65: [443, 228, 40], 66: [40, 442, 240], 67: [244, 333, 190], 68: [220, 200, 256], 69: [304, 253, 391], 70: [154, 398, 314], 71: [314, 184, 398], 72: [393, 154, 389], 73: [387, 375, 377], 74: [378, 380, 274], 75: [142, 87, 167], 76: [267, 13, 100], 77: [438, 14, 382], 78: [147, 386, 411], 79: [379, 12, 0], 80: [323, 376, 151], 81: [114, 26, 384], 82: [383, 392, 390], 83: [440, 383, 95], 84: [171, 139, 444], 85: [399, 265, 139], 86: [175, 5, 265], 87: [11, 128, 174], 88: [90, 192, 11], 89: [82, 209, 229], 90: [315, 316, 263], 91: [313, 305, 269], 92: [338, 430, 269], 93: [403, 317, 338], 94: [224, 60, 285], 95: [223, 211, 280], 96: [322, 286, 191], 97: [86, 441, 6], 98: [252, 243, 277], 99: [424, 448, 193], 100: [241, 196, 309], 101: [446, 445, 218], 102: [423, 207, 446], 103: [107, 431, 207], 104: [47, 48, 435], 105: [177, 47, 48], 106: [138, 35, 111], 107: [254, 298, 281], 108: [113, 201, 21], 109: [21, 20, 405], 110: [405, 152, 156], 111: [116, 125, 436], 112: [272, 358, 409], 113: [30, 296, 133], 114: [91, 83, 15], 115: [57, 46, 37], 116: [294, 45, 199], 117: [402, 208, 83], 118: [79, 212, 247], 119: [195, 97, 38], 120: [56, 96, 98], 121: [208, 402, 134], 122: [150, 134, 122], 123: [202, 250, 143], 124: [207, 423, 242], 125: [123, 369, 406], 126: [222, 349, 381], 127: [157, 158, 41], 128: [293, 385, 357], 129: [303, 310, 160], 130: [188, 178, 373], 131: [395, 397, 396], 132: [18, 284, 8], 133: [121, 153, 394], 134: [120, 59, 180], 135: [276, 101, 275], 136: [288, 55, 287], 137: [336, 104, 347], 138: [292, 146, 237], 139: [434, 433, 408], 140: [54, 17, 368], 141: [221, 414, 295], 142: [3, 53, 311], 143: [361, 439, 232], 144: [78, 145, 225], 145: [245, 312, 140], 146: [365, 353, 80], 147: [135, 344, 23], 148: [62, 99, 216], 149: [155, 279, 198], 150: [149, 210, 334], 151: [27, 364, 258], 152: [77, 170, 66], 153: [9, 319, 81], 154: [28, 351, 159], 155: [325, 22, 370], 156: [367, 204, 320], 157: [348, 350, 84], 158: [39, 194, 372], 159: [136, 246, 28], 160: [273, 400, 343], 161: [447, 217, 131], 162: [181, 447, 235], 163: [183, 447, 179], 164: [182, 119, 129], 165: [67, 235, 33], 166: [7, 307, 427], 167: [299, 388, 203], 168: [340, 429, 270], 169: [70, 300, 205], 170: [141, 94, 71], 171: [162, 69, 363]}

    correct_count = 0
    # Loop through triplet data
    num_test_triplets = len(article_map)
    for i in range(0, num_test_triplets):
        article_1_index = article_map[i][0] +1
        article_2_index = article_map[i][1]+1
        article_3_index = article_map[i][2]+1
        # "The content of URLs one and two should be more similar than the content of URLs two and three"
        # Calculate cosine similarities (This must be done manually since word2vec calculates for specific words)
        article1_wl = clean_test_articles[article_1_index]
        article2_wl = clean_test_articles[article_2_index]
        article3_wl = clean_test_articles[article_3_index]
        try:
            article_1_vec = model.infer_vector(clean_test_articles[article_1_index])
        except Exception, e:
            print "article 1 fucked up!"
            print clean_test_articles[article_1_index]

        try:
            article_2_vec = model.infer_vector(clean_test_articles[article_2_index])
        except Exception, e: 
            print "article 2 fucked up!"
            #print clean_test_articles[article_2_index]

        try:
            article_3_vec = model.infer_vector(clean_test_articles[article_3_index])
        except Exception, e:
            print "article 3 fucked up!"
            #print clean_test_articles[article_3_index]
        
        # if model.docvecs.similarity_unseen_docs(model, article1_wl, article2_wl) > model.docvecs.similarity_unseen_docs(model, article2_wl, article3_wl):
        #     correct_count += 1

        if cosine_sim(article_1_vec, article_2_vec) > cosine_sim(article_2_vec, article_3_vec):
            correct_count += 1

    return "ACCURACY: %f" % (correct_count*1.0/num_test_triplets)


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



    #Train all 5 models 
    model_indx = 1
    for name, train_model in models_by_name.items():
        models_by_name[name] = Doc2Vec.load('small_wiki_subset.' + str(model_indx) + '.model')
        # train_model.build_vocab(alldocs)
        # train_model.train(alldocs)
        # train_model.save('small_wiki_subset.' + str(model_indx) + '.model')

        model_indx = model_indx + 1


    models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
    models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])


    print get_accuracy(models_by_name['dbow+dmm'])
    print get_accuracy(models_by_name['dbow+dmc'])

    models_by_name['dbow+dmm'].save('small_wiki_subset.' + str(4) + '.model')
    models_by_name['dbow+dmc'].save('small_wiki_subset.' + str(5) + '.model')

    print models_by_name.keys()

    # to concat the two models, we need to 



    #model = Doc2Vec(TaggedLineDocument(inp), docvecs_mapfile="small_wiki_subset.docvecs_map", size=400, window=5, min_count=2, workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use (much) less RAM
    #model.init_sims(replace=True)

    #model.save(outp)
