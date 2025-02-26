import re
import contractions
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from collections import Counter
import emoji
import string
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
import google.generativeai as ggi
from google.api_core.exceptions import ServiceUnavailable
import time
import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('opinion_lexicon')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# Preprocessing Function
def preprocess(text):
    # Convert text to lowercase
    text = contractions.fix(text.lower())
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if token not in stopwords.words('english')]
    # Join tokens back into a single cleaned text string
    cleaned_text = ' '.join(tokens)
    return tokens, cleaned_text

# 1. Function to calculate emoji percentage
def calculate_emoji_percentage(text):
    emojis = [char for char in text if char in emoji.EMOJI_DATA]
    total_chars = len(text)
    emoji_count = len(emojis)
    emoji_percentage = (emoji_count / total_chars) * 100 if total_chars > 0 else 0
    emoji_percentage = round(emoji_percentage, 2)
    return emojis, emoji_percentage

# 2. Function to extract punctuation usage features
def extract_punctuation_usage_features(text):
    punctuation_chars = set(string.punctuation)
    punctuation_counts = Counter(char for char in text if char in punctuation_chars)
    total_punctuation = sum(punctuation_counts.values())
    total_chars = len(text)
    punctuation_percentage = (total_punctuation / total_chars) * 100 if total_chars > 0 else 0
    return punctuation_counts, round(punctuation_percentage, 2)

# 3. Function to negative words percentage
negative_words = set(nltk.corpus.opinion_lexicon.words('negative-words.txt'))
word_list = ['Sense of Loss', 'Sentimental', 'Psychological Distress', 'Suffocation', 'Lose Status',
             'Self-reproach', 'Self-hate', 'Low Self-esteem', 'At a Loss', 'Flustered', 'Terrified',
             'Numbness', 'Meaninglessness', 'Treat Unjustly', 'Automutilation', 'Mental Health', 'Psychology',
             'Suicidal Ideation', 'Mental Disease', 'Quarantine', 'Suicidal Behavior', 'Social Isolation',
             'Suicide Attempts', 'Post-traumatic Stress', 'Mental Health Service', 'Mental Stress', 'Insomnia',
             'Distress Syndrome', 'Mental Health Care', 'Social Distancing', 'Psychology Wellbeing', 'Coping Behavior',
             'Psychosis', 'Psychological Stress', 'Domestic Violence', 'Lockdown', 'Sleep Disorders', 'Mood Disorders',
             'Psychological Resilience', 'Alcoholism', 'Job Stress', 'Substance Abuse', 'melancholic', 'morose', 'joyless',
             'heartbroken', 'heartsick', 'doleful', 'blue', 'sombre', 'grey', 'crestfallen', 'elegiac', 'gray', 'brokenhearted',
             'saddened', 'droopy', 'saturnine', 'hangdog', 'black', 'low-spirited', 'anguished', 'funereal', 'down', 'darkening',
             'drear', 'elegiacal', 'comfortless', 'down in the mouth', 'wailing', 'agonized', 'heartsore', 'unquiet', 'weeping',
             'low', 'cast down', 'tearful', 'lachrymose', 'plaintive', 'discouraged', 'dolorous', 'disheartened', 'rueful']

def neg_word_percentage(tokens):
    negative_words_extended = negative_words.union(word_list)
    negative_emotion_count = sum(1 for word in tokens if word in negative_words_extended)
    total_words = len(tokens)
    negative_emotion_percentage = (negative_emotion_count / total_words) * 100 if total_words > 0 else 0
    negative_emotion_percentage = round(negative_emotion_percentage, 2)
    return negative_emotion_percentage

# 4. Function to extract semantic features
def extract_semantic_features(text):
    sid = SentimentIntensityAnalyzer() # VADER
    sentiment_scores = sid.polarity_scores(text)
    sentiment_scores['compound'] = round(sentiment_scores['compound'], 2)
    sentiment = 'positive' if sentiment_scores['compound'] > 0 else 'negative' if sentiment_scores['compound'] < 0 else 'neutral'
    subjectivity = round(TextBlob(text).sentiment.subjectivity, 2)
    return sentiment, sentiment_scores['compound'], subjectivity

# 5. Function to extract modality features
def extract_modality_percentage(tokens):
    modal_verbs = ['can', 'could', 'may', 'might', 'shall', 'should', 'will', 'would', 'must']
    total_words = len(tokens)
    if total_words == 0:
        return 0
    modal_count = sum(1 for word in tokens if word in modal_verbs)
    modal_percentage = (modal_count / total_words) * 100
    return round(modal_percentage, 2)

# 6a. Function to extract n-grams with collocations and scores
def extract_ngrams(tokens):
    bigrams, bigram_scores, trigrams, trigram_scores = [], [], [], []

    bigram_finder = BigramCollocationFinder.from_words(tokens)
    trigram_finder = TrigramCollocationFinder.from_words(tokens)
    bigram_scored = bigram_finder.score_ngrams(BigramAssocMeasures().raw_freq)
    trigram_scored = trigram_finder.score_ngrams(TrigramAssocMeasures().raw_freq)

    for gram, score in bigram_scored:
        bigrams.append(' '.join(gram))
        bigram_scores.append(score)

    for gram, score in trigram_scored:
        trigrams.append(' '.join(gram))
        trigram_scores.append(score)

    return bigrams, bigram_scores, trigrams, trigram_scores

# 6b
def add_ngram_features(df):
    # Apply extract_ngrams and expand the resulting tuples into separate columns
    df[['Bigrams', 'Bigram_Scores', 'Trigrams', 'Trigram_Scores']] = df['tokens'].apply(lambda x: pd.Series(extract_ngrams(x)))

    # Calculate and add additional n-gram statistics
    df['Bigram_Sum'] = df['Bigram_Scores'].apply(sum)
    df['Bigram_Mean'] = df['Bigram_Scores'].apply(lambda x: sum(x) / len(x) if x else 0)
    df['Bigram_Count'] = df['Bigram_Scores'].apply(len)

    df['Trigram_Sum'] = df['Trigram_Scores'].apply(sum)
    df['Trigram_Mean'] = df['Trigram_Scores'].apply(lambda x: sum(x) / len(x) if x else 0)
    df['Trigram_Count'] = df['Trigram_Scores'].apply(len)

    return df

# 7. Function to extract lexical and syntactic features
def extract_lexical_syntactic_features(tokens, cleaned_text):
    pos_tags = nltk.pos_tag(tokens)
    pos_tag_counts = Counter(tag for word, tag in pos_tags)

    word_count = len(tokens)
    char_count = len(cleaned_text)
    avg_word_length = round(char_count / word_count, 2) if word_count > 0 else 0

    first_person_pronouns = {'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'}
    pronoun_count = sum(1 for word in tokens if word in first_person_pronouns)
    pronoun_percentage = round((pronoun_count / word_count) * 100, 2) if word_count > 0 else 0

    repetitive_word_count = word_count - len(set(tokens))
    repetitive_word_percentage = round((repetitive_word_count / word_count) * 100, 2) if word_count > 0 else 0

    sentences = sent_tokenize(cleaned_text)
    sentence_count = len(sentences)
    avg_sentence_length = round(len(tokens) / sentence_count, 2) if sentence_count > 0 else 0

    content_word_tags = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}
    content_word_count = sum(1 for word, tag in pos_tags if tag in content_word_tags)
    lexical_density = round(content_word_count / word_count, 2) if word_count > 0 else 0
    lexical_diversity = round(len(set(tokens)) / word_count, 2) if word_count > 0 else 0

    return pos_tag_counts, word_count, char_count, avg_word_length, pronoun_percentage, repetitive_word_percentage, avg_sentence_length, lexical_density, lexical_diversity

# 8. Function to extract speech acts
def extract_speech_acts(text):
    num_question_marks = text.count('?')
    num_exclamation_marks = text.count('!')
    if num_question_marks > num_exclamation_marks:
        return 'question'
    elif num_exclamation_marks > num_question_marks:
        return 'exclamation'
    else:
        return 'statement'

# 9. Function to extract TFIDF
def apply_tfidf(dataframe, text_column, methods=['average', 'sum', 'max']):
    vectorizer = TfidfVectorizer(stop_words='english', tokenizer=nltk.word_tokenize)
    tfidf_matrix = vectorizer.fit_transform(dataframe[text_column])
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = {}

    for method in methods:
        if method == 'average':
            tfidf_scores[method] = np.asarray(tfidf_matrix.mean(axis=1)).flatten()
        elif method == 'sum':
            tfidf_scores[method] = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
        elif method == 'max':
            tfidf_scores[method] = np.asarray(tfidf_matrix.max(axis=1).todense()).flatten()


    for method in methods:
        dataframe[f'tfidf_{method}'] = tfidf_scores[method]
    return dataframe

# 10. Function to apply Word2Vec
def apply_word2vec(dataframe, tokens_column):
    # Initialize Word2Vec model
    w2v_model = Word2Vec(sentences=dataframe[tokens_column], vector_size=100, window=5, min_count=1, workers=4, alpha=0.025, min_alpha=0.0001)
    w2v_model.train(dataframe[tokens_column], total_examples=len(dataframe[tokens_column]), epochs=10)

    # Aggregate function to compute sum, average, and max embeddings
    def aggregate_embeddings(tokens):
        embeddings = [w2v_model.wv[word] for word in tokens if word in w2v_model.wv]
        if embeddings:
            sum_embedding = np.sum(embeddings, axis=0)
            avg_embedding = np.mean(embeddings, axis=0)
            max_embedding = np.max(np.array(embeddings), axis=0)  # Ensure max is computed correctly

            # Compute scalar values
            sum_scalar = np.sum(sum_embedding)
            avg_scalar = np.mean(avg_embedding)
            max_scalar = np.max(max_embedding)
        else:
            sum_scalar = 0.0
            avg_scalar = 0.0
            max_scalar = 0.0

        return pd.Series([sum_scalar, avg_scalar, max_scalar])

    # Apply aggregate function to compute scalar embeddings and assign them to new columns
    dataframe[['sum_embedding', 'avg_embedding', 'max_embedding']] = dataframe[tokens_column].apply(aggregate_embeddings)

    return dataframe

import pandas as pd
from collections import Counter

def convert_pos_tags(df):
    # Fixed list of POS tags
    pos_tags_list = ['NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
                     'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']
    # Ensure 'pos_tag_counts' is converted from string to Counter
    df['pos_tag_counts'] = df['pos_tag_counts'].apply(lambda x: Counter(eval(x)))
    
    # Create new columns for each POS tag category
    for tag in pos_tags_list:
        df[tag] = df['pos_tag_counts'].apply(lambda x: x.get(tag, 0))
    # Drop the original 'pos_tag_counts' column if no longer needed
    df.drop(columns=['pos_tag_counts'], inplace=True)
    return df

def extract_datetime_features(date_str):
    """Extract date and time features from a date string."""
    try:
        # Convert string to datetime object
        date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        
        # Extract features
        features = {
            'Day_of_Week': date_obj.weekday(),  # 0=Monday, 6=Sunday
            'Year': date_obj.year,
            'Month': date_obj.month,
            'Day': date_obj.day,
            'Hour': date_obj.hour
        }
        
        return features
    except ValueError:
        # Handle incorrect date format
        return {
            'Day_of_Week': None,
            'Year': None,
            'Month': None,
            'Day': None,
            'Hour': None
        }

def format_issues(issues):
    formatted_issues = ""
    for issue in issues:
        formatted_issues += f"✔️ **{issue}**\n"
    return formatted_issues

def get_advice_from_gemini(text):
    prompt = f"""
    The user shared the following: "{text}"
    1. Identify the primary cause of the user's suicidal thoughts from the following categories: Personal, Psychological, Family, Friends, Relationship, Social, Educational, Work, Financial, or others.
    2. Provide advice and words of encouragement for the individual experiencing suicidal thoughts due to the identified issue. This is a MUST!

    Respond like a compassionate mental health professional, not like a bot. You are speaking directly like a human to the individual with suicidal ideation, no need to have "Dear user" or similar terms in front.
    Respond in two sections. First, list the issues out. Then provide the advice in a new paragraph. Use the following format:
   
    Issue:
    - Issue 1
    - Issue 2
    - ...
   
    Advice:
    Write your advice here.
    """


    max_retries = 5
    retry_delay = 1  # initial delay in seconds

    for attempt in range(max_retries):
        try:
            model = ggi.GenerativeModel("gemini-2.0-flash")
            chat = model.start_chat()
            response = chat.send_message(prompt, stream=False)  # Set stream to False for a single response
            print(response)  # Add this line to inspect the response structure
           
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                return response['text'].strip()

        except ServiceUnavailable as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                raise e

def parse_response(response_text):
    issues = []
    advice = []
    is_advice_section = False
    for line in response_text.split('\n'):
        if "Advice:" in line:
            is_advice_section = True
            continue
        if is_advice_section:
            advice.append(line.strip())
        elif line.startswith("- "):
            issues.append(line.strip("- ").strip())
    print(f"Parsed issues: {issues}")  # Debug print
    print(f"Parsed advice: {' '.join(advice)}")  # Debug print
    return issues, '\n'.join(advice)

# Define feature list used in training
features_used_in_training = [
    'Day_of_Week', 'Year', 'Month', 'Day', 'Hour', 'emoji_percentage',
    'punctuation_percentage', 'negative_emotion_percentage', 'sentiment_score', 'subjectivity',
    'modal_percentage', 'Bigram_Sum', 'Bigram_Count', 'avg_word_length', 'pronoun_percentage',
    'repetitive_word_percentage', 'lexical_density', 'tfidf_max', 'avg_embedding', 'max_embedding',
    'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR',
    'RBS', 'Is_Weekend', 'sentiment', 'speech_acts', 'cleaned_text'
]


from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Load the BERT model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert_suicidal_classifier")
bert_model = DistilBertForSequenceClassification.from_pretrained("distilbert_suicidal_classifier")
bert_model = bert_model.half()

# Set up the device for BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert_model.to(device)
bert_model.eval()  # Set the model to evaluation mode



def predict(text, bert_model, datetime_str):
    # Ensure the BERT model is of the correct type
    if not isinstance(bert_model, DistilBertForSequenceClassification):
        raise TypeError("The model provided is not an instance of DistilBertForSequenceClassification.")
    
    # Tokenize the input text for BERT
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Perform the prediction with BERT
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]

    # Convert prediction index to label
    predicted_label = 'suicidal' if predicted_label_idx == 1 else 'nonsuicidal'

    # If the prediction is 'suicidal', fetch advice from Gemini
    if predicted_label == "suicidal":
        advice_text = get_advice_from_gemini(text)
        issues, advice = parse_response(advice_text)
        if issues:
            formatted_issues = format_issues(issues)
            return predicted_label, f"Your problem is likely related to:\n{formatted_issues}\n\n{advice}"
        else:
            return predicted_label, "No immediate issues detected. However, it's always good to check in with yourself regularly and seek support if needed."
    else:
        return predicted_label, "No immediate issues detected. However, it's always good to check in with yourself regularly and seek support if needed."




def get_advice_from_gemini2(text):
    prompt = f"""
    The user shared the following: "{text}"
    
    Respond like a compassionate mental health professional, speaking directly to the individual. Write a heartfelt, natural message that addresses their feelings and provides support without using formal sections. Your response should include empathy, actionable advice, and encouragement in a conversational tone, as if you were a friend or counselor offering support in person.
    
    """


    max_retries = 5
    retry_delay = 1  # initial delay in seconds

    for attempt in range(max_retries):
        try:
            model = ggi.GenerativeModel("gemini-2.0-flash")
            chat = model.start_chat()
            response = chat.send_message(prompt, stream=False)  # Set stream to False for a single response
            print(response)  # Add this line to inspect the response structure
           
            if hasattr(response, 'text'):
                return response.text.strip()
            else:
                return response['text'].strip()

        except ServiceUnavailable as e:
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
                retry_delay *= 2  # exponential backoff
            else:
                raise e
            

def predict_for_community(text, bert_model):
    # Ensure the BERT model is of the correct type
    if not isinstance(bert_model, DistilBertForSequenceClassification):
        raise TypeError("The model provided is not an instance of DistilBertForSequenceClassification.")
    
    # Tokenize the input text for BERT
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Perform the prediction with BERT
    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_label_idx = torch.argmax(logits, dim=1).cpu().numpy()[0]

    # Convert prediction index to label
    predicted_label = 'suicidal' if predicted_label_idx == 1 else 'nonsuicidal'

    # If the prediction is 'suicidal', fetch advice from Gemini without categorizing issues
    if predicted_label == "suicidal":
        advice_text = get_advice_from_gemini2(text)
        return predicted_label, f"{advice_text}"
    else:
        return predicted_label, None
