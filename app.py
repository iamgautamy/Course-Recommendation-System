from flask import Flask
from flask import request
import pandas as pd
from flask import render_template
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from statistics import harmonic_mean
import numpy as np

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<h1>Course Recommendation System<h1>"

def load_df(data):
    df = pd.read_csv(data)
    return df

df = load_df('Default10.csv')
df1 = load_df('CourseraData.csv')

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df1['Course Name'])

def recommend_by_course_title (title, recomm_count=10) : 
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    idx = np.argsort(np.array(cosine_sim[:,0]))[-recomm_count:]
    sdf = df1.iloc[idx].sort_values(by='Course Rating', ascending=False)
    return sdf


@app.route("/default")
def default():
    #drop Unnamed: 0 column
    df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
    #multiply course_rating by 100 and to int type
    return render_template('default10tt.html',data=df)

@app.route("/search",methods=["POST"])
def search():
    if 'q' in request.form:
        query = request.form['q']
        # recommend courses based on query
        results = recommend_by_course_title(query)
        # convert the DataFrame to a list of dictionaries
        data = results.to_dict('records')
        return render_template('search_results.html', query=query, data=data)
    else:
        return "Invalid request"


if __name__ == "__main__":
    app.run(debug=True)