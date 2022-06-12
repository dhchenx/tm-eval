from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import pickle
# import pyLDAvis.gensim

MAX_TOPICS=8
MIN_PERC=0.8

dict_symptoms=pickle.load(open("../datasets/covid19_symptoms.pickle","rb"))
list_categories=pickle.load(open("../datasets/covid19_tags.pickle","rb"))

tokenizer = RegexpTokenizer(r'\w+')

# create English stop words list
en_stop = get_stop_words('en')

# Create p_stemmer of class PorterStemmer
p_stemmer = PorterStemmer()

# create sample documents

# compile sample documents into a list
#doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
doc_set=[]
doc_ids=[]
for key in dict_symptoms.keys():
    doc_set.append(dict_symptoms[key].split(","))
    doc_ids.append(str(key))

# max data set
import random
MAX_DS_SIZE=2000
doc_ids=random.sample(doc_ids,MAX_DS_SIZE)
doc_set=[dict_symptoms[id].split(",") for id in doc_ids]

# list for tokenized documents in loop
texts = []

# loop through document list
for tokens in doc_set:
    # clean and tokenize document string

    # stem tokens
    # stemmed_tokens = [p_stemmer.stem(i) for i in tokens]

    # add tokens to list
    texts.append(tokens)

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(texts)

# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in texts]

list_tags = []
for i in range(MAX_TOPICS):
    list_tags.append("t" + str(i + 1))

from gensim.models import CoherenceModel
if __name__ == '__main__':
    print("Number of topics\tPerplexity")
    for topic in range(2,51):
        # generate LDA model
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic, id2word=dictionary, passes=20,
                                                   alpha='auto', eta='auto')

        # print keywords
        '''
        topics = ldamodel.print_topics(num_words=10)
        for topic in topics:
            print(topic)
        '''
        # ldamodel.save('model5.gensim')

        # Compute Perplexity
        perpelxity = ldamodel.log_perplexity(corpus)
        # coherencemodel = CoherenceModel(model=ldamodel, texts=texts, dictionary=dictionary, corpus=corpus, coherence='u_mass')
        # c_score=coherencemodel.get_coherence()
        print(f'{topic}\t{perpelxity}')  # a measure of how good the model is. lower the better.


