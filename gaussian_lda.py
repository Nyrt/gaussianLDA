import numpy as np 
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import gutenberg
from gensim.models import word2vec
import sys

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_iterations = 20
num_topics = 10
D = 100
N = len(gutenberg.fileids())


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
corpus = "gutenberg"






#Convert to word vectors
try:
	if "-recompute_vectors" in sys.argv:
		assert False
	model = word2vec.Word2Vec.load("%s.word2vec"%corpus)
	print("Model loaded successfully")
except:
	print("Could not load word vectors. Recomputing")

	documents = [gutenberg.sents(file) for file in gutenberg.fileids()]

	# preprocess the documents (convert to lowercase, remove stop words and punctuation, and stem)
	documents = [[[stemmer.stem(word.lower()) for word in sentence if not word.lower() in stop_words and word[0] not in string.punctuation] for sentence in doc] for doc in documents]
	sentences = []
	for document in documents:
		for sentence in document:
			sentences.append(sentence)
	
	word2vec_model = word2vec.Word2Vec(sentences, size=D,  min_count=1, window=5, workers=4)

	word2vec_model.save("%s.word2vec"%corpus)
	print("training complete")


print("loading and preprocessing documents")

documents = [[stemmer.stem(word.lower()) for word in gutenberg.words(file) if not word.lower() in stop_words and word[0] not in string.punctuation] for file in gutenberg.fileids()]

max_wc = 0
for document in documents:
	max_wc= max(max_wc, len(document))

print max_wc

doc_vecs = np.zeros(N,max_wc,D)
for doc in len(documents):
	for w in len(documents[i]):
		doc_vecs[doc, w, :] = word2vec_model.wc[documents[doc][w]]

print np.average(doc_vecs)

print doc_vecs.shape

num_documents = len(documents)

 # number of words in each topic
topic_counts = np.zeros(num_topics)
# topic_doc_counts[i, j] represents how many words of document j are present in topic i.
topic_doc_counts = np.zeros((num_documents, num_topics)) 
#topic_assignment[i][j] gives the table assignment of word j of the ith document. 
topic_assignment = [np.zeros(len(document)) for document in documents]
topic_means = np.zeros(num_topics)
cov_invs = np.eye(word_vector_dims)
dets = np.zeros(num_topics)

iteration = 0

nu_0 = 0
