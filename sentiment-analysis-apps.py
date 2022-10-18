import streamlit as st

import tweepy
import pandas as pd
import numpy as np
import re
import pickle
from PIL import Image

from tqdm import tqdm

import os
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

import preprocessor as p
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import string

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import torch
import urllib
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

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

title = st.text_input("Let's search some Twitter keywords 👇", help="Type any keyword you want to search")
if st.button('Search tweet'):
    col1, col2 = st.columns([2,1])
    with col1:
        sentiment_label=["Positive", "Neutral", "Negative"]
        df = pd.DataFrame(
            {
            "Tweets": [
                "Keren Banget!",
                "Biasa aja",
                "best sangat ke mencuri raden saleh ni",
                "Ternyata film kkn di desa penari biasa aja, dulu takut nonton di bioskop soalnya thread yang di twitter serem eh pas jadi film malah b aja engga ada seremnya 😂",
                "GUE BARU NONTON THOR LOVE AND THUNDER. WITH NO SPOILERS. GILA... BANGGA BGT GW BISA BEBAS DARI SPOILER 😍😍😍",
                "Film Thor love and Thunder seru, kocag juga wkwkwk 😂😂",
                "yeyyy udh selse ntn one piece film red ... keren bangett 🤩 rasanya pengen bawa lightstick sumpahhh jiwa ngidolku meronta-ronta",
                "Big mouse endingnya jelek bgt anj. Kesel gue. Masih banyak pertanyaan yg belom kejawab. Anaknya chairman kang? Jihoon gatau duitnya di choi doha? Fix harusnya 21 episode anying",
                "big mouse gini doang nih endingnya?",
                "BTW kemarin abis nonton Jujutsu Kaisen 0, kirain Rika bakal nempel terus sama Yuta, ternyata di ending filmnya mereka akhirnya misah, hmm jadi Yuta sekarang kekuatannya apa",
                "asdasdasd",
                "waedwadasd",
                "asdwadasd",
                "wadawdaadw",
                "csacacsc"
                ],
            "Sentiment": np.random.choice(sentiment_label,15),
            }
        )
        st.table(df)
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