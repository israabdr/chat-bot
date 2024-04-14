import nltk
import numpy as np
import string
import random
import streamlit as st

nltk.download('wordnet')
nltk.download('punkt')

# Charger le texte
with open('text.txt', 'r', encoding='utf-8') as f:
    raw_doc = f.read()

raw_doc = raw_doc.lower()
print(raw_doc)
# Traitement des phrases et des mots
sentences = nltk.sent_tokenize(raw_doc)
word_tokens = nltk.word_tokenize(raw_doc)
lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

# Fonction de salutation
greet_inputs = ('hello', 'hi', 'greetings', 'sup', 'what\'s up')
greet_responses = ('hi', 'hey', '*nods*', 'hi there', 'hello', 'I am glad! You are talking to me')

def greet(sentence):
    for word in sentence.split():
        if word.lower() in greet_inputs:
            return random.choice(greet_responses)

# TF-IDF et réponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def response(user_response):
    rep = ''
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sentences)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if req_tfidf == 0:
        rep = 'I am sorry! I don\'t understand you'
        return rep
    else:
        rep = rep + sentences[idx]
        return rep

def main():
    st.title("Chatbot Streamlit")
    st.write("Bonjour, je suis un bot. Posez-moi des questions ou dites simplement bonjour!")

    user_response = st.text_input("Vous:", "")

    if st.button("Envoyer"):
        if greet(user_response) != None:
            st.write("Bot:", greet(user_response))
        else:
            st.write("Bot:", response(user_response))

if __name__ == "__main__":
    main()
