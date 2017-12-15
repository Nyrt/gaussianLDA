import numpy as np 
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import brown as corpus
from gensim.models import word2vec
import sys
from math import *
from scipy.stats import multivariate_normal
from scipy.special import gammaln


# use "--recompute_vectors" flag to regenerate word vectors
# use "--recompute_lda" flag to regenerate LDA model
# "--outfile [filename]" to name the LDA model file

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

num_iterations = 20
num_topics = 50
D = 200
N = len(corpus.fileids())


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
corpus_name = "brown"


word2vec_model = None

#Convert to word vectors
try:
	assert "--recompute_vectors" not in sys.argv
	word2vec_model = word2vec.Word2Vec.load("%s.word2vec"%corpus_name)
	print("Model loaded successfully")
except:
	print("Could not load word vectors. Recomputing")

	documents = [corpus.sents(file) for file in corpus.fileids()]

	# preprocess the documents (convert to lowercase, remove stop words and punctuation, and stem)
	documents = [[[stemmer.stem(word.lower()) for word in sentence if not word.lower() in stop_words and word[0] not in string.punctuation] for sentence in doc] for doc in documents]
	sentences = []
	for document in documents:
		for sentence in document:
			sentences.append(sentence)
	
	word2vec_model = word2vec.Word2Vec(sentences, size=D,  min_count=1, window=5, workers=4)

	word2vec_model.save("%s.word2vec"%corpus_name)
	print("training complete")

assert(word2vec_model != None)

print("loading and preprocessing documents")

documents = [[stemmer.stem(word.lower()) for word in corpus.words(file) if not word.lower() in stop_words and word[0] not in string.punctuation] for file in corpus.fileids(categories=['news', 'editorial', 'reviews'])]

max_wc = 0
for document in documents:
	max_wc= max(max_wc, len(document))

doc_vecs = np.zeros((N,max_wc,D))
#print doc_vecs.shape
for doc in range(len(documents)):
	for w in range(len(documents[doc])):  
		vec = word2vec_model.wv[documents[doc][w]]
		#print vec.shape
		#print doc_vecs[doc, w, :].shape
		doc_vecs[doc, w, :] = vec

vocab = set(stemmer.stem(word.lower()) for word in corpus.words(categories=['news', 'editorial', 'reviews']) if not word.lower() in stop_words and word[0] not in string.punctuation)
mu_0 = np.zeros(D)
for word in vocab:
	mu_0 += word2vec_model.wv[word]

mu_0 /= len(vocab)

#print np.average(doc_vecs)

#print doc_vecs.shape

print "Initializing topics & params"

# Prior parameters

#mu_0 = np.mean(doc_vecs, -1) # This doesn't work because of the empty word vectors that pad documents
#print mu_0.shape
nu_0 = D 
k_0 = 0.1 # this is the value used in the paper
sigma_0 = np.eye(D) * 3. * D# Check the paper about this
num_documents = len(documents)
alpha = 1./num_topics
m_0_squared = k_0 * mu_0[:,None].dot(mu_0[None,:])


def update_topic_params(topic):
	topic_count = topic_counts[topic]
	nu_k = nu_0 + topic_count
	k_k = k_0 + topic_count
	mu_k = (k_0 * mu_0 + topic_sums[topic,:])/k_k
	topic_means[topic,:] = mu_k

	#Calculate topic covariance
	sigma_n = sigma_0 + topic_sums_squared[topic,:,:] + m_0_squared - k_k * mu_k[:,None].dot(mu_k[None,:])
	#normalize
	sigma_n *= (k_k+1)/(k_k * (nu_k - D + 1))

	_, dets[topic] = np.linalg.slogdet(sigma_n)

	covs[topic] = sigma_n
	cov_invs[topic] = np.linalg.inv(sigma_n)


# # Working in log space to prevent overflows
# def ln_gamma(x):
# 	return np.sum(np.log(np.arange(1,x)))

# Calculate the log multivariate student-T density for a given word vector and topic
def ln_t_density(word, topic):
	mu = topic_means[topic,:]
	sigmaInv = cov_invs[topic,:,:]
	logdet = dets[topic]
	count = topic_counts[topic]
	nu = nu_0 + count - D + 1
	# I separated some parts of the formula for readability
	a = gammaln((nu + D)/2.)
	LLcomp = ((word-mu)[None,:].dot(sigmaInv).dot(word-mu))
	b = (gammaln(nu/2.) + D/2. * (np.log(nu)+np.log(pi)) + 0.5 * logdet + (nu + D)/2.* np.log(1.+LLcomp/nu))
	return a - b;


try:
	assert "--recompute_lda" not in sys.argv
	print "attempting to load lda model"
	npz = np.load("lda%s.npz"%corpus_name)
	topic_counts=npz['topic_counts'] 
	topic_means=npz['topic_means']
	topic_sums=npz['topic_sums']
	covs=npz['covs']
	topic_doc_counts=npz['topic_doc_counts']
	topic_assignment=npz['topic_assignment']
	assert(topic_counts.shape[0] == num_topics)
	assert(topic_doc_counts.shape[0] == num_documents)
	assert(topic_means.shape[1] == D)
	print "loading successful"
except:
	print "loading failed. Regenerating model"
	# Initialize parameter values


	# number of words in each topic
	topic_counts = np.zeros(num_topics)
	# topic_doc_counts[i, j] represents how many words of document j are present in topic i.
	topic_doc_counts = np.zeros((num_documents, num_topics)) 

	#topic_assignment[i][j] gives the table assignment of word j of the ith document. 
	# NOTE THE DIFFERENT INDEXING due to jagged matrix
	topic_assignment = [np.zeros(len(document)) for document in documents]
	topic_means = np.zeros((num_topics,D))

	# Covariance matrices
	covs = np.zeros((num_topics,D,D))
	
	# Storing these for efficiency
	cov_invs = np.zeros((num_topics,D,D))
	dets = np.zeros(num_topics)
	topic_sums = np.zeros((num_topics,D))
	topic_sums_squared = np.zeros((num_topics,D, D))

	# Used in computing sigma_n

	# Assign initial topics randomly
	for doc in range(len(documents)):
		for w in range(len(documents[doc])):
			wordvec = doc_vecs[doc,w,:]
			topic = int(np.random.choice(np.arange(num_topics)))
			topic_assignment[doc][w] = topic
			topic_counts[topic] += 1
			topic_doc_counts[doc, topic] += 1
			topic_sums[topic,:] += wordvec
			topic_sums_squared[topic,:,:] += wordvec[:,None].dot(wordvec[None,:])

	# Find parameters of each topic
	for topic in range(num_topics):
		#make sure topic isn't emtpy?
		update_topic_params(topic)

	print "Initialization complete. Beginning sampler:"

	# Run gibbs sampler

	for iteration in range(num_iterations):
		print "iteration %i out of %i"%(iteration, num_iterations)
		for doc in range(len(documents)):
			print "topic counts:"
			print topic_counts

			print "doc %i out of %i"%(doc, len(documents))
			for w in range(len(documents[doc])):
				wordvec = doc_vecs[doc,w,:]
				prev_topic = int(topic_assignment[doc][w])

				#remove the word from its topic
				topic_assignment[doc][w] = -1 # So it's clear what's been removed
				topic_counts[prev_topic] -= 1
				topic_doc_counts[doc, prev_topic] -= 1
				topic_sums[prev_topic,:] -= wordvec
				topic_sums_squared[prev_topic,:,:] -= wordvec[:,None].dot(wordvec[None,:])

				#Update the old topic to reflect the change
				update_topic_params(prev_topic)

				# Find posterior over topics given this word
				# Working in log space to prevent overflows
				posterior = np.zeros(num_topics)
				counts = topic_doc_counts[doc,:] + alpha #prior

				# Find log likelihood
				for topic in range(num_topics):
					posterior[topic]  = ln_t_density(wordvec, topic)

				posterior -= np.max(posterior) #Again, to prevent overflows
				posterior += np.log(counts) # Add in the log prior
				#Normalize
				posterior = np.exp(posterior)
				posterior /= np.sum(posterior)

				# Sample a new topic and update the parameters
				new_topic = np.random.choice(np.arange(num_topics), p=posterior)

				topic_assignment[doc][w] = new_topic
				topic_counts[new_topic] += 1
				topic_doc_counts[doc, new_topic] += 1
				topic_sums[new_topic,:] += wordvec
				topic_sums_squared[new_topic,:,:] += wordvec[:,None].dot(wordvec[None,:])
				update_topic_params(new_topic)
		print "saving model"
		if "--outfile" in sys.argv:
			outfile = sys.argv[sys.argv.index("--outfile") +1]+"_checkpoint"
		else:
			outfile = "lda%s_checkpoint.npz"%corpus_name

		np.savez(outfile,   topic_counts=topic_counts, 
											topic_means=topic_means,
											topic_sums=topic_sums, 
											covs=covs, 
											topic_doc_counts=topic_doc_counts,
											topic_assignment=topic_assignment)

	print "Done!"


	print "saving model"
	if "--outfile" in sys.argv:
		outfile = sys.argv[sys.argv.index("--outfile") +1]
	else:
		outfile = "lda%s.npz"%corpus_name

	np.savez(outfile,   topic_counts=topic_counts, 
										topic_means=topic_means,
										topic_sums=topic_sums, 
										covs=covs, 
										topic_doc_counts=topic_doc_counts,
										topic_assignment=topic_assignment)


def print_top_words(n=10):
	print "Finding top 10 words for each topic"
	top_words = [{}]*num_topics

	for doc in range(len(documents)):
		for w in range(len(documents[doc])):
			word = documents[doc][w]
			topic = int(topic_assignment[doc][w])
			wordvec = doc_vecs[doc,w]
			prob = multivariate_normal.pdf(wordvec, mean=topic_means[topic], cov=covs[topic,:,:])
			top_words[topic][word] = prob

	for i in range(num_topics):
		print "Topic %i"%i
		words = sorted(top_words[i].iteritems(), key=lambda(k,v): (v,k), reverse = True)
		for w in range(10):
			print words[w]

print_top_words(10)

# output 