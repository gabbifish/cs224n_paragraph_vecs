from gensim.models import Doc2Vec

import bz2
import numpy as np
from numpy import linalg
from numpy.linalg import norm
from scipy.spatial.distance import squareform, pdist

# We import sklearn.
import sklearn
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import scale

# We'll hack a bit with the t-SNE code in sklearn 0.15.2.
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities,
                                    _kl_divergence)
from sklearn.utils.extmath import _ravel

# We'll use matplotlib for graphics.
import matplotlib.pyplot as plt
import matplotlib.patheffects as PathEffects
import matplotlib
# We import seaborn to make nice plots.
import seaborn as sns
sns.set_style('darkgrid')
sns.set_palette('muted')
sns.set_context("notebook", font_scale=1.5,
                rc={"lines.linewidth": 2.5})

NUM_ARTICLES = 515
NUM_CATEGORIES = 26

class doc2vec_tsne(object):
    def __init__(self):
    		print "Loading Doc2Vec model."
		self.model = Doc2Vec.load('big_wiki_subset.en.doc2vec.model', mmap=None) # replace with doc2vec model
    		# self.data =  json.loads(open('../big_wiki_subset/big_wiki_subset.en.txt','r').read()) ## a list
    		# self.processed_speeches = open('all_speech.txt','r')

    def speech_vectors(self):
    	print "Generating vectors for test vectors."
	l = NUM_ARTICLES # num articles--this is ~450?
    	wiki_entry_matrix = np.zeros(shape = (l, 400))

        categories_list = []
        curr_cat = "" # current category to map article indices in wiki_entry_matrix to.
        article_count = 0

        test_articles = open('../testing_articles/wikipedia-hand-triplets-release.txt', 'r')
        article_indices = []
	for line in test_articles:
            if line == "": #When parsing through list file, ignore empty lines
                pass
            if line[0] == "#": # start new dict list for wiki categories and the appropriate content
                if curr_cat != "":
		    categories_list.append((curr_cat, article_indices))
		article_indices = []
		curr_cat = line[2:].replace('\n', '') # get topic name
                #categories_dict[curr_cat] = []
            else:
                files = line.split() # split triplets into list by spaces
                filenames = []

                # Generate vector for each file, set it as row vector.
                for file in files:
		    filename = file.replace('http://en.wikipedia.org/wiki/', '') + '.txt'
                    text = open('../testing_articles/articles/'+filename, 'r')
                    text = text.read().replace('\n', '').split()

                    wiki_entry_matrix[article_count,] = self.model.infer_vector(text) # generate vector for newdoc and save in matrix for graphing
                    article_indices.append(article_count) # article_count serves as index of docvec
                    if article_count % 50 == 0:
                        print "Processed " + str(article_count) + "th article into vector form..."
		    article_count += 1
	
	np.save('wiki_entry_matrix.npy', wiki_entry_matrix)
        self.wiki_entry_matrix = wiki_entry_matrix
        self.categories_dict = categories_list

    	# for v, line in enumerate(self.processed_speeches):
    	# 	words = re.sub('\"','',line).strip().split()
    	# 	speech_matrix[v,] = sum(self.model[word] for word in words if word in self.model)
    	# numpy.save('speech_vectors.npy', speech_matrix)
    	# self.speech_matrix = speech_matrix

    #def tsne(self):
    	#self.speech_2d = bh_sne(self.speech_matrix)

    def scatter(self):
        # We choose a color palette with seaborn.
        palette = np.array(sns.color_palette("hls", NUM_CATEGORIES)) # we have 26 categories

        # X_reduced = TruncatedSVD(n_components=50, random_state=0).fit_transform(vectors)
        X_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(self.wiki_entry_matrix)

	print "Beginning to generate scatterplot!"
        # create color list.
        colors = []
	color_index_tuples = []
        curr_color = 0
	new_begin_index = 0
        for topic, articles in self.categories_dict:
	    topic_range = (new_begin_index, new_begin_index + len(articles))
	    color_index_tuples.append(topic_range)
            for x in xrange(0, len(articles)):
                colors.append(palette[curr_color])
		# color_index.append(curr_color)
	    curr_color += 1
	    new_begin_index = new_begin_index + len(articles)
        # We create a scatter plot.
        f = plt.figure(figsize=(8, 8))
        ax = plt.subplot(aspect='equal')
        sc = ax.scatter(X_embedded[:,0], X_embedded[:,1], lw=0, s=40,
                        c=colors)
        plt.xlim(-25, 25)
        plt.ylim(-25, 25)
        ax.axis('off')
        ax.axis('tight')

        # We add the labels for each digit.
	print "Adding wiki category labels to scatterplot."
        txts = []
	for i in range(NUM_CATEGORIES):
            # Position of each label
            xtext, ytext = np.median(X_embedded[color_index_tuples[i][0]:color_index_tuples[i][1], :], axis=0)
            txt = ax.text(xtext, ytext, self.categories_dict[i][0], fontsize=12)
            txt.set_path_effects([
                PathEffects.Stroke(linewidth=5, foreground="w"),
                PathEffects.Normal()])
            txts.append(txt)

        plt.savefig('images/doc2vec_tsne-generated.png', dpi=120)

        return f, ax, sc, txts


if __name__ =='__main__':
	work = doc2vec_tsne()
	work.speech_vectors()
	#work.tsne()
	work.scatter()
