from flask import Flask, render_template, redirect, request, url_for
import psycopg2
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

conn = psycopg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
cur = conn.cursor()

def fetch_recipe(id_num):
    """
    takes a recipe id, queries the database, and returns a dictionary of the values
    """
    cur.execute("""SELECT name,
                          minutes, 
                          tags,
                          description, 
                          ingredients 
                     FROM raw_recipes
                     WHERE id = %s""" % (str(id_num)))
    fetch = cur.fetchall()[0]
    return {'name':fetch[0],'minutes':fetch[1],'tags':fetch[2],'description':fetch[3],'ingedients':fetch[4]}

def create_recipe_df(recommendations):
    """
    takes a list of recipe id numbers and creates a dataframe from the sql database 
    """
    print(recommendations)
    df = pd.DataFrame(data=None, columns=['name','minutes','tags','description','ingredients'])
    for recipe_id in recommendations:
        recipe = fetch_recipe(recipe_id)
        df = df.append(recipe, ignore_index=True)
    return df

def fetch_recommendations(id_num):
    """
    takes a user id number and returns a list of recipe ids from the sql database
    """
    cur.execute('SELECT recommendation FROM recommendations WHERE id = %s' % (str(id_num)))
    recommendations = cur.fetchall()[0]
    return recommendations[0].split(', ')

def reformat(id_list):
    """
    takes a list of id numbers (as strings) and reformats to a string for a sql query
    """
    string = ""
    for idn in id_list:
        string += "recipe_" + idn + ", "
    return string[:-2]

def get_neighbors(vector, recipes, n=5):
    """
    takes a vector returns similar users from the predictions table
    """
    cur.execute(f"SELECT id, {reformat(recipes)} FROM predictions")
    all_users = np.array(cur.fetchall())
    similarities = cosine_similarity(all_users[:,1:], vector.reshape(1,-1))
    indices = similarities.flatten().argsort()[::-1][:n]
    return all_users[indices][:,0].flatten()
    


def predict_vector(neighbor_id_list):
    """
    takes a list of ids and returns recommendations for the user
    the function queries the database with the id numbers, then
    creates a composite vector for the user and returns the recipes
    with the highest predicted ratings
    """
    matrix = []
    for id_num in neighbor_id_list:
        cur.execute(f"SELECT * FROM predictions where id = {id_num}")
        vector = np.array(cur.fetchall()[0])
        matrix.append(vector)
    composite = np.array(matrix).mean(axis=0)
    return composite

def make_recommendations(prediction_vector, n=10):
    """
    takes a predicted vector and returns a list of the highest rated recipe ids
    """
    # get recipe id's as an array
    cur.execute("SELECT * FROM predictions LIMIT 0")
    columns = np.array([col_name[7:] for col_name in np.array(cur.description)[1:,0]])
    indices = prediction_vector.argsort()[::-1][:n]
    return columns[indices]


def create_recipe_df(recipe_id_list):
    """
    takes a list of recipe ids and returns the desired information from
    the sql database as a pandas dataframe
    """
    df = pd.DataFrame(data=None, columns=['id','name','description','ingredients'])
    for id_num in recipe_id_list:
        cur.execute(f"SELECT id, name, description, ingredients FROM raw_recipes WHERE id = {id_num}")
        info = cur.fetchall()[0]
        df = df.append({'id':info[0],'name':info[1],'description':info[2], 'ingredients':info[3]}, ignore_index=True)
    return df


def get_most_popular(n=5):
    """
    queries the predictions database and returns the n most popular items
    """
    # return column names and store in list
    cur.execute('SELECT * FROM predictions LIMIT 0')
    columns_in_db = [col_name[7:] for col_name in np.array(cur.description)[1:,0]]

    # transpose predictions and calculate average rating for each item
    cur.execute('SELECT * FROM predictions')
    predictions = np.array(cur.fetchall())[:,1:]
    df = pd.DataFrame(data=predictions.T, index=columns_in_db)
    averages = df.mean(axis=1)
    most_popular = averages.sort_values(0, ascending=False).index[:n]
    return most_popular


app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET'])
def home():
    most_popular = get_most_popular()
    information = create_recipe_df(most_popular)
    return render_template('home.html', df=information)

@app.route('/recommend', methods=['POST'])
def recommend():
    first = request.form["0"]
    second = request.form["1"]
    third = request.form["2"]
    fourth = request.form["3"]
    fifth = request.form["4"]
    vector = np.array([first, second, third, fourth, fifth])
    recipe_ids = get_most_popular()
    neighbors = get_neighbors(vector, recipe_ids)
    predicted_vector = predict_vector(neighbors)
    recommendation_ids = make_recommendations(predicted_vector)
    df = create_recipe_df(recommendation_ids)
    return render_template('results.html', df=df)

@app.route('/search', methods=['GET'])
def search():
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    user_id = request.form['user_id']
    recs = fetch_recommendations(user_id)
    df = create_recipe_df(recs)
    return render_template('results.html', df=df)


if __name__ == '__main__':
    app.run(debug=True)