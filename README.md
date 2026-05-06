## Movie Review Sentiment Analysis

This project provides a simple Streamlit app for predicting whether a movie review is **positive** or **negative** using a trained RNN model saved as `rnn_imdb_model.h5`.

## What `main.py` Does

`main.py`:

- starts a Streamlit web app
- loads the trained TensorFlow/Keras model
- accepts a movie review from the user
- converts the review into an IMDb-style integer sequence
- pads the sequence to length `500`
- predicts the sentiment and displays the result

## Project Files

- `main.py` - Streamlit application entry point
- `rnn_imdb_model.h5` - trained sentiment analysis model
- `END_TO_END_RNN.ipynb` - notebook used for model development/training
- `utils.py` - currently empty

## Requirements

Install the required Python packages:

```bash
pip install streamlit tensorflow numpy
```

## How to Run

From the project root:

```bash
streamlit run Movie_Review_sentiment/main.py
```

## How It Works

1. The user enters a movie review in the text box.
2. The app uses the IMDb word index from `tensorflow.keras.datasets.imdb`.
3. Each word is mapped to its corresponding token id.
4. The token sequence is padded to `500` words.
5. The trained model predicts the review sentiment.
6. A label is shown as either `Positive` or `Negative`.

## Notes

- The model path in `main.py` is currently hardcoded as:


- If the project is moved to another machine or folder, update this path or replace it with a relative path.
- Review preprocessing is basic and splits text by spaces before token lookup.

## Example Use

Input:

```text
This movie was emotional, well acted, and beautifully directed.
```

Possible output:

```text
Positive
```
# Movie_Review_sentiment
