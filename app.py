import streamlit as st
import nltk
from textblob import TextBlob
import pandas as pd
import plotly.express as px
from collections import Counter
import re

# Download required NLTK data at startup
@st.cache_resource  # This decorator ensures the download happens only once
def download_nltk_data():
    try:
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('tokenizers/punkt/english.pickle')
    except Exception as e:
        st.error(f"Error downloading NLTK data: {e}")

# Download data at startup
download_nltk_data()

def clean_text(text):
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Convert to lowercase
    text = text.lower()
    
    return text

def get_sentiment(text):
    # Create TextBlob object
    blob = TextBlob(text)
    
    # Get sentiment polarity (-1 to 1)
    polarity = blob.sentiment.polarity
    
    # Get sentiment subjectivity (0 to 1)
    subjectivity = blob.sentiment.subjectivity
    
    # Determine sentiment category
    if polarity > 0:
        category = "Positive"
    elif polarity < 0:
        category = "Negative"
    else:
        category = "Neutral"
        
    return {
        "polarity": polarity,
        "subjectivity": subjectivity,
        "category": category
    }

def analyze_text_details(text):
    try:
        # Tokenize the text
        tokens = nltk.word_tokenize(text)
        
        # Get part of speech tags
        pos_tags = nltk.pos_tag(tokens)
        
        # Count word types
        pos_counts = Counter(tag for word, tag in pos_tags)
        
        # Count words
        word_count = len(tokens)
        
        # Count sentences
        sentences = nltk.sent_tokenize(text)
        sentence_count = len(sentences)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "pos_counts": pos_counts
        }
    except Exception as e:
        st.error(f"Error in text analysis: {e}")
        return {
            "word_count": 0,
            "sentence_count": 0,
            "pos_counts": Counter()
        }

def main():
    st.title("Text Sentiment Analyzer")
    
    # Text input
    text_input = st.text_area("Enter the text to analyze:", height=150)
    
    if text_input:
        # Clean the text
        cleaned_text = clean_text(text_input)
        
        # Get sentiment analysis
        sentiment_results = get_sentiment(cleaned_text)
        text_details = analyze_text_details(text_input)
        
        # Display results in columns
        col1, col2, col3 = st.columns(3)
        
        # Sentiment Category with color coding
        with col1:
            st.subheader("Sentiment")
            category_color = {
                "Positive": "green",
                "Negative": "red",
                "Neutral": "blue"
            }
            st.markdown(f"<h3 style='color: {category_color[sentiment_results['category']]};'>{sentiment_results['category']}</h3>", 
                       unsafe_allow_html=True)
        
        # Polarity and Subjectivity
        with col2:
            st.subheader("Polarity")
            st.write(f"{sentiment_results['polarity']:.2f}")
        
        with col3:
            st.subheader("Subjectivity")
            st.write(f"{sentiment_results['subjectivity']:.2f}")
        
        # Text Statistics
        st.subheader("Text Statistics")
        stats_col1, stats_col2 = st.columns(2)
        
        with stats_col1:
            st.write(f"Word Count: {text_details['word_count']}")
            st.write(f"Sentence Count: {text_details['sentence_count']}")
        
        if text_details['pos_counts']:
            # Create a DataFrame for parts of speech
            pos_df = pd.DataFrame({
                'Part of Speech': list(text_details['pos_counts'].keys()),
                'Count': list(text_details['pos_counts'].values())
            })
            
            # Plot parts of speech distribution
            fig = px.bar(pos_df, 
                        x='Part of Speech', 
                        y='Count',
                        title='Parts of Speech Distribution')
            st.plotly_chart(fig)
            
            # Display explanation of POS tags
            if st.checkbox("Show POS Tags Explanation"):
                st.markdown("""
                **Common POS Tags:**
                - NN: Noun
                - VB: Verb
                - JJ: Adjective
                - RB: Adverb
                - DT: Determiner
                - IN: Preposition
                - CC: Conjunction
                - PRP: Personal Pronoun
                """)
        
        # Add history tracking
        if 'history' not in st.session_state:
            st.session_state.history = []
        
        # Add current analysis to history
        current_analysis = {
            'text': text_input[:50] + "..." if len(text_input) > 50 else text_input,
            'sentiment': sentiment_results['category'],
            'polarity': sentiment_results['polarity']
        }
        
        if current_analysis not in st.session_state.history:
            st.session_state.history.append(current_analysis)
        
        # Display analysis history
        if st.checkbox("Show Analysis History"):
            st.subheader("Previous Analyses")
            history_df = pd.DataFrame(st.session_state.history)
            st.dataframe(history_df)

if __name__ == "__main__":
    main()