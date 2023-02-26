import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from statistics import harmonic_mean

def load_df(data):
    df = pd.read_csv(data)
    return df

df = load_df('deafult10.csv')
df1= load_df('df1.csv')
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df1.course_title)

def recommend_by_course_title (title, recomm_count=10) : 
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    idx = np.argsort(np.array(cosine_sim[:,0]))[-recomm_count:]
    sdf = df1.iloc[idx].sort_values(by='overall_rating', ascending=False)
    return sdf


st.title('Top 10 Courses')
course_text = st.text_input(label="Enter course you are intrested in")
defaults = st.write(df)

if course_text:
    st.write(recommend_by_course_title(course_text))
    



