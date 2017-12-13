import numpy as np 
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import gutenberg
from gensim.models import word2vec
import sys
from math import *


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


word2vec_model = None

#Convert to word vectors
try:
	if "--recompute_vectors" in sys.argv:
		assert False
	word2vec_model = word2vec.Word2Vec.load("%s.word2vec"%corpus)
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

assert(word2vec_model != None)

print("loading and preprocessing documents")

documents = [[stemmer.stem(word.lower()) for word in gutenberg.words(file) if not word.lower() in stop_words and word[0] not in string.punctuation] for file in gutenberg.fileids()]

max_wc = 0
for document in documents:
	max_wc= max(max_wc, len(document))

#print max_wc

mu_0 = np.zeros(D)
vec_count = 0
doc_vecs = np.zeros((N,max_wc,D))
#print doc_vecs.shape
for doc in range(len(documents)):
	for w in range(len(documents[doc])):
		vec = word2vec_model.wv[documents[doc][w]]
		#print vec.shape
		#print doc_vecs[doc, w, :].shape
		doc_vecs[doc, w, :] = vec
		mu_0 += vec
		vec_count += 1
mu_0 /= vec_count


#print np.average(doc_vecs)

#print doc_vecs.shape

print "Initializing topics & params"

# Initialize parameter values

num_documents = len(documents)

# number of words in each topic
topic_counts = np.zeros(num_topics)
# topic_doc_counts[i, j] represents how many words of document j are present in topic i.
topic_doc_counts = np.zeros((num_documents, num_topics)) 

#topic_assignment[i][j] gives the table assignment of word j of the ith document. 
# NOTE THE DIFFERENT INDEXING due to jagged matrix
topic_assignment = [np.zeros(len(document)) for document in documents]
topic_means = np.zeros((num_topics,D))

# Covariance matrices, their inverses, and their determinants
covs = np.zeros((num_topics,D,D))
cov_invs = np.zeros((num_topics,D,D))
dets = np.zeros(num_topics)

# Storing these for efficiency
topic_sums = np.zeros((num_topics,D))
topic_sums_squared = np.zeros((num_topics,D, D))



# Prior parameters

mu_0 = np.zeros(D)


#mu_0 = np.mean(doc_vecs, -1) # This doesn't work because of the empty word vectors that pad documents
print mu_0.shape
nu_0 = D 
k_0 = 0.1 # this is the value used in the paper
sigma_0 = np.eye(D) * 3 * D # Check the paper about this

deg_freedom = nu_0 - D + 1

sigma_T = sigma_0 * (k_0 + 1.0)/(k_0 * deg_freedom)

sigma_T_inv = np.linalg.inv(sigma_T)

sigma_T_det = np.linalg.det(sigma_T)

# Used in computing sigma_n
k_0mu_0mu_0_T = k_0 *  mu_0[:,None].dot(mu_0[None,:])

# Assign initial topics randomly
for doc in range(len(documents)):
	for w in range(len(documents[doc])):
		wordvec = doc_vecs[doc,w,:]
		topic = np.random.choice(np.arange(num_topics))
		topic_assignment[doc][w] = topic
		topic_counts[topic] += 1
		topic_doc_counts[doc, topic] += 1
		topic_sums[topic,:] += wordvec
		topic_sums_squared[topic,:,:] += wordvec[:,None].dot(wordvec[None,:])

def update_topic_params(topic):
	topic_count = topic_counts[topic]
	nu_n = nu_0 + topic_count
	k_n = k_0 + topic_count
	mu_n = (k_0 * mu_0 + topic_sums[topic,:])/k_n
	topic_means[topic,:] = mu_n

	#Calculate topic covariance
	sigma_n = sigma_0 + topic_sums_squared[topic,:,:] + k_0mu_0mu_0_T - k_n * mu_n[:,None].dot(mu_n[None,:])
	#normalize
	sigma_n *= (k_n+1)/(k_n * (nu_n - D + 1))
	dets[topic] = np.linalg.det(sigma_n)

	covs[topic] = sigma_n
	cov_invs[topic] = np.linalg.inv(sigma_n)

# Find parameters of each topic
for topic in range(num_topics):
	#make sure topic isn't emtpy?
	update_topic_params(topic)

print "Initialization complete"

# Calculate the log multivariate student-T density for a given word vector and topic
def ln_t_density(word, topic):
	mu = topic_means[topic,:]
	sigmaInv = cov_invs[topic,:,:]
	det = determinants[topic]
	count = topic_counts[topic]
	nu = nu_0 + count - D + 1
	print (word-mu)[None,:].dot(sigmaInv).dot(word-mu)
	return np.log(gamma((nu + D)/2)) - (np.log(gamma(nu/2)) + D/2 * (np.log(nu)+np.log(pi)) + 0.5 * np.log(det) + (nu + D)/2* np.log(1+(word-mu)[None,:].dot(sigmaInv).dot(word-mu)/nu));	

# Run gibbs sampler

for iteration in range(num_iterations):
	for doc in range(len(documents)):
		for w in range(len(documents[doc])):
			wordvec = doc_vecs[doc,w,:]
			prev_topic = topic_assignment[doc][w]

			#remove the word from its topic
			#topic_assignment[doc][w] = -1 # So it's clear what's been removed
			topic_counts[prev_topic] -= 1
			topic_doc_counts[doc, prev_topic] -= 1
			topic_sums[prev_topic,:] -= wordvec
			topic_sums_squared[prev_topic,:,:] -= wordvec[:,None].dot(wordvec[None,:])

			#Update the old topic to reflect the change
			update_topic_params(prev_topic)

			# Find posterior over topics given this word
			posterior = np.zeros(num_topics)
			max_prob = float("-inf")
			for topic in range(num_topics):
				count = topic_doc_counts[doc,topic]
				logL = ln_t_density(wordvec, topic)
				log_posterior = logL + np.log(count)
				posterior[topic] = log_posterior
				if log_posterior > max_prob:
					max_prob = log_posterior
			#Normalize
			#618

# output 