import streamlit as st

import tweepy
import pandas as pd
import numpy as np
import re
import pickle
from PIL import Image

from tqdm import tqdm

import os
# import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

# import preprocessor as p
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import string

import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import torch
# import urllib
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report


api_key = 'W4ijIV5mQYqLwrohvA46I6WUj'
api_key_secret = 'Q2p6fvOVLO1OpWYWs1BxIwTAouMHkq0QtHcjzmYpjY6YfNzvdM'
access_token = '1322449438728581122-5WjkuJsCKdWVJfZor76Ns6mcjgROWB'
access_token_secret = 'zPyda5oNbNg7fsLpFKrWVk07Wpc3vLF2mGiu5CCtkfm2i'
bearer_token = 'AAAAAAAAAAAAAAAAAAAAAHWycQEAAAAA%2Bjdcl%2FsasYFptpzIFj98W7z1COI%3DdT9b8mpAUiSjkveO6yOjBnuNPMrCxGg7igy9vZmcmF8cgjOHKL'

client = tweepy.Client(bearer_token=bearer_token)

# Twitter scrape prerquisite
def get_tweets(query):
  attempt = 0
  tweet_data = []
  user_data = []
  request = client.search_recent_tweets(
      query = query, 
      max_results=100,
      tweet_fields=['created_at','lang','conversation_id'], 
      expansions=['author_id']
      )
  tweet_data.extend(tweet_text_process(request))
  user_data.extend(tweet_user_process(request))

  final_data = pd.merge(pd.DataFrame(tweet_data), pd.DataFrame(user_data), on = 'author_id')
  final_data['query'] = query
  print(f"Finishing scrape, total of {len(final_data)} tweets are found.")
  return pd.DataFrame(final_data)

def tweet_text_process(request):
  tweets = []
  for tweet in request.data:
    data = {}
    data['tweet_id'] = tweet.id
    data['tweet_date'] = tweet.created_at
    data['lang'] = tweet.lang
    data['text'] = tweet.text
    data['conversation_id'] = tweet.conversation_id
    data['author_id'] = tweet.author_id
    tweets.append(data)
  return tweets
  
def tweet_user_process(request):
  user_data = []
  users = {u['id']:u for u in request.includes['users']}
  for tweet in request.data:
    if users[tweet.author_id]:
      user = users[tweet.author_id]
      data = {}
      data['author_id'] = tweet.author_id
      data['username'] = user.username
      data['name'] = user.name
      user_data.append(data)
  return user_data

# preprocessing prerequirements

# Normalized words
lexicon_alay = pd.read_csv('https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv')

# NLTK stopwords
nltk_stopwords = stopwords.words('indonesian')

# Sastrawi stopwords
sastrawi_stopwords = StopWordRemoverFactory().get_stop_words()

# Additional stopwords
additional_stopwords = [
    'dengan', 'ia','bahwa','oleh', 'samaaaaaa', 'big', 'mouse', 'mouth', 
    'miracle', 'cell', 'mencuri', 'raden', 'saleh', 'ngeri-ngeri', 'sedap', 'one', 'piece', 
    'red', 'cyberpunk', 'edgerunners', 'ivanna', 'purple', 'heart', 'kkn', 'penari', 'desa', 'thor', 'thunder', 'love'
    ]

# Final stopwords
final_stopwords = list(set(nltk_stopwords+sastrawi_stopwords+additional_stopwords))

# Stemmer
stemmer = StemmerFactory().create_stemmer()

# text cleaning
def cleaning_proc(text_col):
  cleaned_text = []
  for text in text_col:
    clean_str = text.lower() # lowercase
    clean_str = re.sub(r"(?:\@|#|https?\://)\S+", " ", clean_str) # eliminate username, url, hashtags
    clean_str = re.sub(r'&amp;', '', clean_str) # remove &amp; as it equals &
    clean_str = re.sub(r'[^\w\s]',' ', clean_str) # remove punctuation
    clean_str = re.sub(r'[0-9]', '', clean_str) # remove number
    clean_str = re.sub(r'[\s\n\t\r]+', ' ', clean_str) # remove extra space
    clean_str = clean_str.strip() # trim
    cleaned_text.append(clean_str)
  return cleaned_text

# text normalization
def normalization_proc(text_col):
    normalized_text = []
    for text in text_col:
        fw = [w for w in text.split(" ")]
        example_rec = ''
        for word in fw:
            if (word in lexicon_alay['slang'].to_list()):
                propper = lexicon_alay[lexicon_alay['slang'] == word]['formal'].drop_duplicates().iloc[0]
                # iloc[0] karena mungkin 1 slang punya definisi formal yg berbeda jadi ditarik yg paling atas biar seragam.
                example_rec += propper + " "
            elif (word not in lexicon_alay['slang'].to_list()):
                propper = word
                example_rec += propper + " "
        example_rec = example_rec[:-1]
        normalized_text.append(example_rec)
    return normalized_text

# stopwords removal
def stopwords_proc(text_col, stopwords):
  nostopwords_text = []
  for text in text_col:
    clean_tokens = [w for w in text.split(" ") if (w not in stopwords) and (w != "")]
    clean_text = " ".join(clean_tokens)
    nostopwords_text.append(clean_text)
  return nostopwords_text

# text stemming
def stem_proc(text_col):
    stemmed_text = []
    for text in text_col:
        stemming = stemmer.stem(text)
        stemmed_text.append(stemming)
    return stemmed_text

# final preproc function
def preprocess_this(text_col):
  text_col_df = pd.DataFrame(text_col)
  text_col_df['cleaned_text'] = cleaning_proc(text_col)
  text_col_df['normalized_text'] = normalization_proc(text_col_df['cleaned_text'])
  text_col_df['no_stopwords_text'] = stopwords_proc(text_col_df['normalized_text'], final_stopwords)
  text_col_df['stemmed_text'] = stem_proc(text_col_df['normalized_text'])
  text_col_df = text_col_df[text_col_df['stemmed_text'] != ""]
  # text_col_df.drop_duplicates(inplace = True)
  # text_col_df.reset_index(inplace = True, drop = True) 
  print(f"Finishing text preprocessing, predicting sentiment.")
  return text_col_df

# modelling prerequirements
# roBERTa
device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained("roberta_model")
model = AutoModelForSequenceClassification.from_pretrained("roberta_model")

def roberta_predict(text_col):
  BATCH_SIZE = 100 
  labels = ['positive', 'neutral', 'negative']
  scores_all = np.empty((0,len(labels)))
  text_all = text_col.to_list()
  n = len(text_all)
  with torch.no_grad():
      for start_idx in tqdm(range(0, n, BATCH_SIZE)):
          end_idx = min(start_idx + BATCH_SIZE, n) 
          encoded_input = tokenizer(text_all[start_idx:end_idx], return_tensors='pt', padding=True, truncation=True).to(device)
          
          output = model(**encoded_input)
          scores = output[0].detach().cpu().numpy()
          scores = softmax(scores, axis=1)
          scores_all = np.concatenate((scores_all, scores), axis=0)

          del encoded_input, output, scores 
          torch.cuda.empty_cache()
  sample_preprocessed = pd.DataFrame(scores_all, columns=labels)
  sample_preprocessed.insert(len(sample_preprocessed.columns), 'sentiment', " ")
  for i in range(len(sample_preprocessed)):
    if sample_preprocessed['negative'][i] > sample_preprocessed['positive'][i] and sample_preprocessed['negative'][i] > sample_preprocessed['neutral'][i]:
      sample_preprocessed['sentiment'][i] = 'Negative'
    elif sample_preprocessed['positive'][i] > sample_preprocessed['negative'][i] and sample_preprocessed['positive'][i] > sample_preprocessed['neutral'][i]:
      sample_preprocessed['sentiment'][i]= 'Positive'
    else:
      sample_preprocessed['sentiment'][i] = 'Neutral'

  sample_preprocessed.drop(['negative','positive','neutral'], axis=1, inplace=True)
          
  return sample_preprocessed


st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!",
    }
)

logo = Image.open('images/edts.png')
st.image(logo, width=100)

st.markdown("<h3 style='text-align: center; color: #2a93d6;'>DMT Share</h3>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: #105588;'>Twitter Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #6D6E70;'>This project is about the Sentiment Analysis of public opinion on Twitter. With this app, you can search some Twitter keywords<br> and return the corresponding tweets, then classify the tweets' sentiment and word clouds, whether it's positive, negative, or neutral.</p>", 
    unsafe_allow_html=True)
st.write(" ")

twitter_search = st.text_input("Let's search some Twitter keywords 👇", help="Type any keyword you want to search")
if st.button('Search tweet'):
    col1, col2 = st.columns([2,1])
    with col1:
      def deploy_sentiment():
        query = twitter_search
        query  =  query.lower() + ' -is:retweet lang:id'
        data = get_tweets(query)
        data = data[['tweet_id', 'tweet_date','text']]
        data['tweet_date'] = pd.to_datetime(data['tweet_date'], format="%F")
        clean_data = preprocess_this(data['text'])
        data = pd.concat([data, clean_data['stemmed_text']], axis=1)
        data['sentiment'] = roberta_predict(data['stemmed_text'])
        data.drop_duplicates(inplace = True)
        data.reset_index(inplace = True, drop = True) 
        output_data = data[['tweet_id', 'tweet_date', 'text', 'sentiment']]
        return output_data
      
      tes=deploy_sentiment()

      st.table(tes)

        # sentiment_label=["Positive", "Neutral", "Negative"]
        # df = pd.DataFrame(
        #     {
        #     "Tweets": [
        #         "Keren Banget!",
        #         "Biasa aja",
        #         "best sangat ke mencuri raden saleh ni",
        #         "Ternyata film kkn di desa penari biasa aja, dulu takut nonton di bioskop soalnya thread yang di twitter serem eh pas jadi film malah b aja engga ada seremnya 😂",
        #         "GUE BARU NONTON THOR LOVE AND THUNDER. WITH NO SPOILERS. GILA... BANGGA BGT GW BISA BEBAS DARI SPOILER 😍😍😍",
        #         "Film Thor love and Thunder seru, kocag juga wkwkwk 😂😂",
        #         "yeyyy udh selse ntn one piece film red ... keren bangett 🤩 rasanya pengen bawa lightstick sumpahhh jiwa ngidolku meronta-ronta",
        #         "Big mouse endingnya jelek bgt anj. Kesel gue. Masih banyak pertanyaan yg belom kejawab. Anaknya chairman kang? Jihoon gatau duitnya di choi doha? Fix harusnya 21 episode anying",
        #         "big mouse gini doang nih endingnya?",
        #         "BTW kemarin abis nonton Jujutsu Kaisen 0, kirain Rika bakal nempel terus sama Yuta, ternyata di ending filmnya mereka akhirnya misah, hmm jadi Yuta sekarang kekuatannya apa",
        #         "asdasdasd",
        #         "waedwadasd",
        #         "asdwadasd",
        #         "wadawdaadw",
        #         "csacacsc"
        #         ],
        #     "Sentiment": np.random.choice(sentiment_label,15),
        #     }
        # )
        # st.table(df)
    with col2:
        st.markdown("<p style='text-align: center; color: #6D6E70;'>Positive Word Cloud</p>", unsafe_allow_html=True)
        positive = Image.open('images/positive.png')
        st.image(positive)

        st.markdown("<p style='text-align: center; color: #6D6E70;'>Negative Word Cloud</p>", unsafe_allow_html=True)
        negative = Image.open('images/negative.png')
        st.image(negative)

        st.markdown("<p style='text-align: center; color: #6D6E70;'>Neutral Word Cloud</p>", unsafe_allow_html=True)
        neutral = Image.open('images/neutral.png')
        st.image(neutral)