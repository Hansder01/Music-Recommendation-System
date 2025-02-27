# 🎵 Music Recommendation System  

A **Streamlit-based Song Recommendation System** that suggests similar songs based on **lyrics and mood analysis** using **Natural Language Processing (NLP) and Machine Learning**.  

## 📌 Features  
✅ **Multi-song input**: Users can input **one or more song names** (comma-separated).  
✅ **Lyrics & Mood-based recommendations**: Uses **TF-IDF Vectorization** and **Cosine Similarity** to find similar songs.  
✅ **Sentiment Analysis**: Determines song sentiment (**Positive, Negative, or Neutral**) using **TextBlob**.  
✅ **Mood Classification**: Categorizes songs into moods like **Happy, Sad, Romantic, or Dark**.  
✅ **Streamlit UI**: Interactive and user-friendly interface for recommendations.  

## 🚀 Installation  

### **1️⃣ Clone the Repository**  
```bash
git clone https://github.com/your-username/song-recommendation.git
cd song-recommendation
```

### **2️⃣ Install Dependencies**  
Ensure you have Python **3.7+** installed. Then, install the required libraries:  
```bash
pip install -r requirements.txt
```

### **3️⃣ Download NLTK Resources**  
Run the following command in Python to download necessary NLP resources:  
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### **4️⃣ Run the Streamlit App**  
```bash
streamlit run app.py
```

## 📂 Dataset  
The model is trained on the **Spotify Millsong Data** dataset (`spotify_millsongdata.csv`), containing **song lyrics** and metadata.  

## 🎯 How It Works  
1. **User enters one or more song names** in the search bar.  
2. **Lyrics are preprocessed** (cleaning, stopword removal, lemmatization).  
3. **TF-IDF Vectorization** converts lyrics into numerical form.  
4. **Nearest Neighbors Model (Cosine Similarity)** finds similar songs.  
5. **Recommendations are displayed** along with their similarity score.  

## 🛠️ Technologies Used  
- **Python** (pandas, numpy, scikit-learn, textblob, nltk)  
- **Machine Learning** (TF-IDF, Nearest Neighbors Model)  
- **Natural Language Processing (NLP)** (Sentiment Analysis, Mood Classification)  
- **Streamlit** (Web UI)  
