import os
import json
import pymongo
from flask import Flask
from flask import request
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

usr = 'gy1950'
pwd = 'gy1950'

client = pymongo.MongoClient("mongodb+srv://" + usr + ":" + pwd + "@cluster0.jt7p7.mongodb.net/?retryWrites=true&w=majority")

db = client['Course-Recommend-System']

collection = db['course-rec-v1']
collection_df10 = db['default_10']
collection_query = db['last_search_query']
###############################################################################################################################
@app.route("/")
def hello_world():
    return "<h1>Course Recommendation System<h1>"


df = pd.DataFrame(list(collection_df10.find()))
df1 = pd.DataFrame(list(collection.find()))

def save_query(qq):
    comm = {'last_query':qq,'user_id':'gautam_test'}
    collection_query.insert_one(comm)
    print("Query saved")

vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(df1['Course Name'])

def recommend_by_course_title(title, recomm_count=10):
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    if cosine_sim.any() < 0.5:
        return("Invalid")
    else:
        idx = np.argsort(np.array(cosine_sim[:, 0]))[-recomm_count:]
        sdf = df1.iloc[idx].sort_values(by='Course Rating', ascending=False)
        return sdf

@app.route("/default")
def default():
    #if user_id = 'gautam_test' does not exit in collection_query write the if statement
    if collection_query.find_one({'user_id':'gautam_test'}) is None :
        #drop Unnamed: 0 column
        df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
        #multiply course_rating by 100 and to int type
        return render_template('default10tt.html',data=df)
    else:
        #if last_query == '' render default10tt.html with data= df
        if collection_query.find_one({'user_id':'gautam_test'})['last_query'] == '' or 'Invalid' or ' ':
            return render_template('default10tt.html',data=df)
        else: 
            lst_q = collection_query.find({'user_id':'gautam_test'})
            df_lst_q = pd.DataFrame(list(lst_q))
            return render_template('default10tt.html',data=recommend_by_course_title(str(df_lst_q['last_query'][0])))

@app.route("/search",methods=["POST"])
def search():
    if 'q' in request.form:
        query = request.form['q']
        #FUNCTION TO SAVE QUERY 
        #if gautam_test already exit update it with query
        if collection_query.find_one({'user_id':'gautam_test'}) is not None:
            collection_query.update_one({'user_id':'gautam_test'},{'$set':{'last_query':query}})
        else:save_query(query)
        # recommend courses based on query
        results = recommend_by_course_title(query)

        if str(results)== "Invalid":
            return render_template('invalidpg.html')

        # convert the DataFrame to a list of dictionaries
        else:
            data = results.to_dict('records')
            return render_template('search_results.html', query=query, data=data)
    else:
        return "Invalid request"

if __name__ == "__main__":
    app.run(debug=True)
