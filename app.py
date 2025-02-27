import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import seaborn
import streamlit as st
import warnings
import time
warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')


@st.cache_data
def load_data():
    df = pd.read_csv('spotify_millsongdata.csv')
    df = df.drop('link', axis=1).reset_index(drop=True)
    df['cleaned_text'] = df['text'].apply(clean_lyrics)
    df[['Polarity_Score', 'Sentiment']] = df['cleaned_text'].apply(lambda x: pd.Series(textblob_sentiment(x)))
    df['Mood'] = df['Polarity_Score'].apply(classify_mood)
    df['combined'] = df['cleaned_text'] + " " + df['Mood']
    return df

def clean_lyrics(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    if not isinstance(text, str):  # Handle NaN values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords and lemmatize
    return ' '.join(words)

def textblob_sentiment(text):
    if isinstance(text, str):
        polarity = TextBlob(text).sentiment.polarity
        # Classify sentiment based on polarity score
        if polarity > 0:
            sentiment = "Positive"
        elif polarity < 0:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        return polarity, sentiment
    return None, "Unknown"

def classify_mood(polarity):
    if polarity > 0.3:
        return "Happy / Cheerful"
    elif 0.1 < polarity <= 0.3:
        return "Positive / Romantic / Uplifting"
    elif -0.1 <= polarity <= 0.1:
        return "Neutral / Calm / Reflective"
    elif -0.3 <= polarity < -0.1:
        return "Sad / Melancholic"
    else:
        return "Dark / Depressing / Angry"
    
def recommend_songs(song_titles, df, model, tfidf_matrix):
    if isinstance(song_titles, str):  # Convert single song to list
        song_titles = [song_titles]

    song_indices = [df[df['song'] == title].index[0] for title in song_titles if title in df['song'].values]

    if not song_indices:
        return "None of the input songs found in the dataset. Try another.", None
    
    all_recommendations = set()
    total_similarity_scores = []
    
    for song_index in song_indices:
        distances, indices = model.kneighbors(tfidf_matrix[song_index], n_neighbors=6)
        
        for i, idx in enumerate(indices.flatten()[1:]):  # Exclude input song itself
            all_recommendations.add((df.iloc[idx]['song'], df.iloc[idx]['artist']))
            total_similarity_scores.append(1 - distances.flatten()[i + 1])  # Cosine similarity

    avg_similarity = sum(total_similarity_scores) / len(total_similarity_scores) if total_similarity_scores else 0

    return list(all_recommendations), avg_similarity
    
# Load data
df = load_data()

# Build TF-IDF Model
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df['combined'])

# Train Nearest Neighbors Model
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6)
model.fit(tfidf_matrix)

# Streamlit UI
st.title("🎵 Song Recommendation System")
st.subheader("Find similar songs based on lyrics & mood")

# User Input
song_input = st.text_area("Enter a song name:", "")

if st.button("Find Similar Songs"):
    if song_input:
        song_list = [song.strip() for song in song_input.split(",")]
        recommendations, avg_score = recommend_songs(song_list, df, model, tfidf_matrix)
        
        if isinstance(recommendations, str):
            st.warning(recommendations)
        else:
            st.success(f"Here are songs similar to **{song_list}**:")
            for song, artist in recommendations:
                st.write(f"🎶 {song} - {artist}")
            
            st.info(f"**Average Similarity Score:** {avg_score:.2f}")
    else:
        st.error("Please enter a song name.")

