import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import streamlit as st 

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model   

st.markdown(
    "<h1 style = 'color: red;'>Movie Review Sentiment Analysis</h1>",unsafe_allow_html=True
    )

st.write("This is a simple movie review sentiment analysis app. You can enter a movie review and the app will predict whether the review is positive or negative.") 

review = st.text_input("Enter your movie review here:")

model = load_model(r"D:\ml\movie_review_prediction\Movie_Review_sentiment\rnn_imdb_model.h5",compile=False)

def preprocess_review(review):
    # Tokenize the review and convert it to a sequence of integers
    word_index = imdb.get_word_index()
    review_sequence = [word_index.get(word, 2)+3 for word in review.split()]
    # Pad the sequence to ensure it has the same length as the training data
    review_sequence = sequence.pad_sequences([review_sequence], maxlen=500)
    return review_sequence


if st.button("Predict Sentiment"):
    if review:
        processed_review = preprocess_review(review)
        prediction = model.predict(processed_review)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        
        st.markdown(f" <p style='color: blue;font-size:14px;'>The  review is: {review}</p>", unsafe_allow_html=True)
        if sentiment == "Positive":
            st.markdown("<h2 style='color: green;'>Positive Review 😀</h2>",unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color: red;'>Negative Review  😞</h2>",unsafe_allow_html=True)   
        
        #st.write(f"The predicted sentiment of the review is: {sentiment}")
    else:
        st.write("Please enter a movie review to predict its sentiment.")