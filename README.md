import pandas as pd
import nltk
from transformers import pipeline
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

# Download NLTK data
nltk.download('punkt')

# Simulated social media conversation (replace with real data)
conversation = [
    "I'm so happy today! Everything is going well 🌞",
    "I'm feeling really down. Life is hard sometimes.",
    "This is hilarious 😂 I can't stop laughing.",
    "I need help. No one seems to understand what I'm going through.",
    "Thank you all for your kind words and support!",
    "Why is the app crashing all the time? So frustrating.",
    "Just finished my workout. Feeling strong and energized!",
    "I miss how things used to be before the pandemic.",
]

# === Step 1: Sentiment & Emotion Analysis ===
# Load pre-trained models
sentiment_model = pipeline("sentiment-analysis")
emotion_model = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# Function to analyze one message
def analyze_message(text):
    sentiment = sentiment_model(text)[0]
    emotions = emotion_model(text)[0]
    top_emotion = max(emotions, key=lambda x: x['score'])

    return {
        "text": text,
        "sentiment": sentiment["label"],
        "sentiment_score": round(sentiment["score"], 3),
        "emotion": top_emotion["label"],
        "emotion_score": round(top_emotion["score"], 3)
    }

# Analyze conversation
analysis_results = [analyze_message(msg) for msg in conversation]
df_analysis = pd.DataFrame(analysis_results)

# === Step 2: Thought/Topic Modeling ===
# BERTopic to detect "thoughts" or themes in the conversation
topic_model = BERTopic(verbose=False)
topics, probs = topic_model.fit_transform(conversation)

# Add topic info to DataFrame
df_analysis["topic"] = topics
df_analysis["topic_label"] = topic_model.get_topic_info().loc[topics]["Name"].values

# === Final Output ===
print(df_analysis)
