import numpy as np # linear algebra
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from collections import  Counter
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import re
import pickle
import streamlit as st

# DATA CLEANING
with open("C:/Users/durga prasad/Desktop/project/.venv/newspaper/vectorizer.pkl","rb") as v1:
  vector=pickle.load(v1)


with open("C:/Users/durga prasad/Desktop/project/.venv/newspaper/Classify_Model.pkl","rb") as r1:
  naivebayes=pickle.load(r1)

def cleaning(bbc_text):
    if len(bbc_text)==1:
        str1=" "
        data_string=str1.join(bbc_text)
        word_tokens = data_string.split()
    #print(word_tokens))
    else:
        # Tokenize : dividing Sentences into words
        bbc_text['text_clean'] = bbc_text['News_Headline'].apply(nltk.word_tokenize)


    # Remove stop words
    if len(bbc_text)==1:
        stop_words = set(stopwords.words('english')) 
        filtered_sentence = [w for w in word_tokens if not w in stop_words] 
        filtered_sentence = []   
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w)
    else:
        stop_words=set(nltk.corpus.stopwords.words("english"))
        bbc_text['text_clean'] = bbc_text['text_clean'].apply(lambda x: [item for item in x if item not in stop_words])

    #Will keep words and remove numbers and special characters
    if len(bbc_text)!=1:
        regex = '[a-z]+'
        bbc_text['text_clean'] = bbc_text['text_clean'].apply(lambda x: [char for char in x if re.match(regex, char)])

    


def predict_class(headline):
    list1=[]
    list1.append(headline)
    cleaning(list1)
    vec4 = vector.transform(list1).toarray()
    classify = (str(list(naivebayes.predict(vec4))[0]).replace('0', 'TECH').replace('1', 'BUSINESS').replace('2',
                                                                                                             'SPORTS').replace(
        '3', 'ENTERTAINMENT').replace('4', 'POLITICS').replace('5','WEATHER'))
    return classify

headline=st.text_input('enter your text')

if st.button('Find Category'):

    ans=predict_class(headline)
    st.write(ans)
