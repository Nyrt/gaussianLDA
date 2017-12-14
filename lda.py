# Generates topics using the Gensim implementation of LDA for comparision


import numpy as np 
import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.corpus import brown as data


from gensim import corpora, models
import gensim


tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

print("loading and preprocessing documents")

documents = [data.words(file) for file in data.fileids(categories=['news', 'editorial', 'reviews'])]

print len(documents[0])

# preprocess the documents (convert to lowercase, remove stop words and punctuation, and stem)
documents = [[stemmer.stem(word.lower()) for word in doc if not word.lower() in stop_words and word[0] not in string.punctuation] for doc in documents]
print len(documents[0])

# Create a dictionary
dictionary = corpora.Dictionary(documents)
    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in documents]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=50, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics = 10, num_words = 10))