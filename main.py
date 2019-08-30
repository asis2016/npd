import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pandas as pd
from stop_words_custom import stop_words
import string

def lda(file, column, start, end, topics, words):
    # Stop words
    stop_en_words = stopwords.words('english')
    stop_en_words.extend(stop_words)
    exclude_punc = set(string.punctuation)
    lemma = WordNetLemmatizer()

    # Read CSV
    data = pd.read_csv(file)
    data_review = data[[column]].dropna(axis=0)

    # List
    data_review_list = data_review[column].tolist()
    data_final = data_review_list[start: end]

    def clean(d):
        stop_free = " ".join([i for i in d.lower().split() if i not in stop_en_words])
        punctutaion_free = ''.join(ch for ch in stop_free if ch not in exclude_punc)
        data_normalized = " ".join(lemma.lemmatize(word) for word in punctutaion_free.split())
        return data_normalized

    data_clean = [clean(d).split() for d in data_final]

    # TD of corpus; unique term is assigned an index
    dictionary = corpora.Dictionary(data_clean)

    # Corpus into Document Term Matrix (dtm)
    dtm = [dictionary.doc2bow(doc) for doc in data_clean]

    ldamodel = gensim.models.ldamodel.LdaModel

    # Run + Train LDA model
    # passes: The number of laps the model will take through corpus
    ldamodel = ldamodel(dtm, num_topics=topics, id2word=dictionary, passes=50)

    print(ldamodel.print_topics(num_words=words))


lda("dataset/women_clothes_review.csv", "Review Text", 1, 7, 4, 4)













