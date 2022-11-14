from pickle import load

import pandas as pd
import numpy as np

from textblob import TextBlob
import nltk
nltk.download()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

# from wordcloud import WordCloud

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import contractions
vectoriseur_pickle=open('./vectoriseur_file','rb')
vectoriseur=load(vectoriseur_pickle)

model_pickle=open('./model_file','rb')
model=load(model_pickle)

tokenizer = RegexpTokenizer(r'\w+')

def tokenize_text(text):
    text_processed = " ".join(tokenizer.tokenize(text))
    return text_processed

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    
    tokens_tagged = nltk.pos_tag(nltk.word_tokenize(text))
    lemmatized_text_list = list()
    
    for word, tag in tokens_tagged:
        if tag.startswith('J'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'a')) # Lemmatise adjectives. Not doing anything since we remove all adjective
        elif tag.startswith('V'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'v')) # Lemmatise verbs
        elif tag.startswith('N'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'n')) # Lemmatise nouns
        elif tag.startswith('R'):
            lemmatized_text_list.append(lemmatizer.lemmatize(word,'r')) # Lemmatise adverbs
        else:
            lemmatized_text_list.append(lemmatizer.lemmatize(word)) # If no tags has been found, perform a non specific lemmatisation
    
    return " ".join(lemmatized_text_list)


def normalize_text(text):
    return " ".join([word.lower() for word in text.split()])
def contraction_text(text):
    return contractions.fix(text)
negative_words = ['not', 'no', 'never', 'nor', 'hardly', 'barely']
negative_prefix = "NOT_"
def get_negative_token(text):
    tokens = text.split()
    negative_idx = [i+1 for i in range(len(tokens)-1) if tokens[i] in negative_words]
    for idx in negative_idx:
        if idx < len(tokens):
            tokens[idx]= negative_prefix + tokens[idx]
    
    tokens = [token for i,token in enumerate(tokens) if i+1 not in negative_idx]
    
    return " ".join(tokens)

from spacy.lang.en.stop_words import STOP_WORDS

def remove_stopwords(text):
    english_stopwords = stopwords.words("english") + list(STOP_WORDS) + ["tell", "restaurant"]
    
    return " ".join([word for word in text.split() if word not in english_stopwords])

def preprocess_text(text):
    
    # Tokenize review
    text = tokenize_text(text)
    
    # Lemmatize review
    text = lemmatize_text(text)
    
    # Normalize review
    text = normalize_text(text)
    
    # Remove contractions
    text = contraction_text(text)

    # Get negative tokens
    text = get_negative_token(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text
    
topics1={0:'les perssonels et les tables',
       1:'mauvaise gout des plats greek',
       2:'mauvaise pizza et retard de livraison',
       3:'retard de pre-commmande et de commande',
       4:'Qualite des repas et des serveurs ne sont pas au attendu',
       5:'mauvais endroit',
       6:'Burger',
       7:'beaucoup attente',
       8:'les poulets et les salades ne sont pas a la hauteur',
       9:'mauvais bar et mauvaise boisson',
       10:'prix elevé par rapport a la quantité',
       11:'livraison',
       12:'sandwich',
       13:'suchi',
       14:'mauvais environnement'}
def predict_topics(model, vectorizer, n_topics, text):
        polarity=TextBlob(text).sentiment.polarity
        print("polarity")
        if polarity<0:
            text=preprocess_text(text)


            text=[text]

            vectorized=vectorizer.transform(text)

            topics_correlations=model.transform(vectorized)
            unsorted_topics_correlations=topics_correlations[0].copy()
            topics_correlations[0].sort()
            sorted=topics_correlations[0][::-1]
            print(sorted)
            topics=[]
            for i in range(n_topics):
                corr_value= sorted[i]
                result = np.where(unsorted_topics_correlations == corr_value)[0]
                topics.append(topics1.get(result[0]))
            print(topics)
        else:
            return polarity
