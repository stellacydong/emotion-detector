import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json

# load json and create model
json_file = open('model_90acc.json', 'r')
loaded_model_json = json_file.read()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_90acc.h5")

def lower_text(text):
    return text.lower()


contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}
def change_contraction(text):
    return ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in text.split(" ")])

from bs4 import BeautifulSoup
def remove_html(text):
    return BeautifulSoup(text, "lxml").text

import re
def remove_urls(text):
    return re.sub('https?://\S+|www\.\S+', '', text)

from string import ascii_letters
allowed = set(ascii_letters + ' ')
def remove_punctuation(text):
    return ''.join([c if c in allowed else ' ' for c in text])

def tokenize_to_words(text):
    #words = nltk.word_tokenize(text) #tokenizing or splitting a string, text into a list of words.
    return text.split() # split is faster in this case

import nltk
from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))
stopwords.remove('not')
def remove_stopwords(text):
    return [word for word in text if word not in stopwords]

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
def stem_words(text):
    return [stemmer.stem(word) for word in text]


def clean_text(text):
    text = lower_text(text)
    text = change_contraction(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_punctuation(text)
    text = tokenize_to_words(text)
    text = remove_stopwords(text)
    text = stem_words(text)
    return text

import tensorflow
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict_text(text):
    try:
        text = clean_text(text)
    except:
        text = [clean_text(item) for item in text]
    try:
        with open('tokenizer_for_90acc_model.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            try:
                text = tokenizer.texts_to_sequences([text])
            except:
                text = tokenizer.texts_to_sequences(text)
            text = pad_sequences(text, padding='post', maxlen=100)
    except:
        text = np.array([text])
    predictions = (loaded_model.predict(text) > 0.5).astype("int32")
    if len(predictions) == 1:
        if predictions==1:
            return 'Positive'
        return 'Negative'
    return ['Positive' if prediction == 1 else 'Negative' for prediction in predictions]

import requests
from bs4 import BeautifulSoup
import pandas as pd

header = {
    'Host': 'www.amazon.com',
    'User-Agent': 'Mozilla/4.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'TE': 'Trailers'}

def get_soup(url):
    req = requests.get(url, headers = header)
    soup = BeautifulSoup(req.content, "lxml")
    return soup

def get_reviews(soup):
    reviews = soup.find_all('div', {'data-hook': 'review'})
    page_reviews = []
    try:
        for item in reviews:
            review = {
            'product' : soup.title.text.replace('Amazon.com: Customer reviews:', '').strip(),
            'title': item.find('a', {'data-hook': 'review-title'}).text.strip(),
            #'rating':  float(item.find('i', {'data-hook': 'review-star-rating'}).text.replace('out of 5 stars', '').strip()),
            'body': item.find('span', {'data-hook': 'review-body'}).text.strip(),
            }
            page_reviews.append(review)
    except:
        pass
    return page_reviews

def get_reviews_df(url):
    soup = get_soup(url)

    reviewlist = []
    for x in range(1,51):  # the number of pages of the review
        soup = get_soup(url + f'&pageNumber={x}')
        #print(f'Getting page: {x}')
        reviewlist.extend(get_reviews(soup))
        #print(len(reviewlist))
        if not soup.find('li', {'class': 'a-disabled a-last'}):
            pass
        else:
            break

    return pd.DataFrame(reviewlist)

from matplotlib import pyplot as plt
from collections import defaultdict
from wordcloud import WordCloud

def get_colors(labels):
    colors = {'Positive' : 'green',
              'Negative' : 'red'}
    return [colors[label] for label in labels]

def predict_amazon(url):
    start = url[:23]
    if start!='https://www.amazon.com/':
        return '_', '_', 'Not url'
    review_df = get_reviews_df(url)
    if len(review_df) == 0:
        error_code = requests.get(url, headers=header).status_code
        if error_code==200:
            return '_', '_', 'Error 200. Amazon webscrape temporarily down.'
        return '_', '_', f'Error {error_code}'

    review_df = review_df.drop_duplicates()
    review_df['text'] = review_df['title'] + ' ' + review_df['body']
    review_df['sentiment'] = predict_text(review_df['text'].values)

    title = get_soup(url).title.string

    fig = plt.figure()
    labels = review_df['sentiment'].value_counts().index.tolist()
    plt.pie(review_df['sentiment'].value_counts(), labels=labels, autopct='%1.1f%%', colors=get_colors(labels))
    plt.savefig('static/images/pie.png')

    plt.clf()
    plt.bar(x=labels, height=review_df['sentiment'].value_counts(), color=get_colors(labels))
    for index,data in enumerate(review_df['sentiment'].value_counts()):
        plt.text(x=index-0.3, y=max(review_df['sentiment'].value_counts())*0.1, s=f"{data}", fontdict=dict(fontsize=20))
    plt.savefig('static/images/bar.png')

    word2count = defaultdict(dict)
    for sentiment in ['Negative', 'Positive']:
        for data in review_df[review_df['sentiment'] == sentiment]['text'].apply(lower_text).apply(remove_punctuation).apply(tokenize_to_words).apply(remove_stopwords):
            for word in data:
                if word not in word2count['All'].keys():
                    word2count['All'][word] = 1
                else:
                    word2count['All'][word] += 1
                if word not in word2count[sentiment].keys():
                    word2count[sentiment][word] = 1
                else:
                    word2count[sentiment][word] += 1

    word2count['Positive'] = {key: word2count['Positive'][key] - word2count['Negative'].get(key, 0) for key in word2count['Positive']}
    word2count['Negative'] = {key: word2count['Negative'][key] - word2count['Positive'].get(key, 0) for key in word2count['Negative']}

    colors = {'Negative' : 'red',
              'Positive' : 'green',
              'All' : 'grey'}

    for i, sentiment in enumerate(['Negative', 'Positive', 'All']):
        plt.clf()
        wordcloud = WordCloud(
                                background_color=colors[sentiment],
                                max_words=int(len(word2count['All'])*0.1),
                                max_font_size=75,
                                random_state=5
                                ).generate_from_frequencies(word2count[sentiment])
        plt.title(f'{sentiment} text word cloud')
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis('off')
        plt.savefig(f'static/images/{sentiment}_wordcloud.png')

    return title, len(review_df), 'Success'
