from flask import Flask, render_template, redirect, request, url_for
import psycopg2
import pandas as pd
import numpy as np
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity
import time


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
    df = pd.DataFrame(data=None, columns=['name','minutes','tags','description','ingredients'])
    for recipe_id in recommendations:
        recipe = fetch_recipe(recipe_id)
        df = df.append(recipe, ignore_index=True)
    return df

def fetch_recommendations(id_num, model, n=10):
    """
    takes a user id number and returns a list of recipe ids from the predictions DataFrame
    """
    table_dict = {'svd':'svd_recommendations',
                'nmf':'nmf_recommendations',
                'knnb':'knnb_recommendations',
                'knnwm':'knnwm_recommendations',
                'coclust':'coclust_recommendations',
                'slope1':'slope1_recommendations'}

    cur.execute(f"""SELECT recommendation FROM {table_dict[model]}
                    WHERE id = {id_num}""")
    recs = cur.fetchall()[0][0]
    split = recs.split(',')
    return [int(id_num) for id_num in split]

def generate_recommendations(uid, model_type, thresh=4.5, n=10):
    """
    takes a user id that the model was trained on and makes predictions for a set
    of recipes in the dataset, then compares those the threshold and returns n recipe ids
    """
    model_dict = {'svd':'../models/SVD_model',
                  'nmf':'../models/NMF_model',
                  'knnb':'../models/KNNBasic_model',
                  'knnwm':'../models/KNNWithMeans_model',
                  'coclust':'../models/CoClustering_model',
                  'slope1':'../models/SlopeOne_model'}
    model = dump.load(model_dict[model_type])[1]

    iids = np.loadtxt('../data/item_ids.txt', dtype='int32')
    np.random.shuffle(iids)
    iids = iids[:5000]
    predictions = np.zeros(5000)
    start = time.time()
    for ind, iid in enumerate(iids):
        predictions[ind] = model.predict(uid,iid)[3]
    end = time.time()
    indices = predictions > thresh
    print(indices)
    return iids[indices][:n], round(end - start, 6)

def reformat(id_list):
    """
    takes a list of id numbers (as strings) and reformats to a string for a sql query
    """
    string = ""
    for idn in id_list:
        string +=  + idn + ", "
    return string[:-2]

def get_neighbors(vector, model_type, n=5):
    """
    takes a vector returns similar users from the predictions table
    """
    table_dict = {'svd':'svd_predictions',
                'nmf':'nmf_predictions',
                'knnb':'knnb_predictions',
                'knnwm':'knnwm_predictions',
                'coclust':'coclust_predictions',
                'slope1':'slope1_predictions'}
    
    cur.execute(f"""SELECT *
                   FROM {table_dict[model_type]}""")
    info = np.array(cur.fetchall(), dtype='float64')
    uids = info[:,0]
    predictions = info[:,1:]
    similarities = cosine_similarity(predictions, vector.reshape(1,-1))
    indices = similarities.flatten().argsort()[::-1][:n]
    return uids[indices]
    


def predict_vector(neighbor_id_list, model_type, n=10):
    """
    takes a list of ids and returns recommendations for the user
    the function plugs the ids into the model and makes predictions 

    """
    # select which model to use
    model_dict = {'svd':'../models/SVD_model',
                  'nmf':'../models/NMF_model',
                  'knnb':'../models/KNNBasic_model',
                  'knnwm':'../models/KNNWithMeans_model',
                  'coclust':'../models/CoClustering_model',
                  'slope1':'../models/SlopeOne_model'}
    model = dump.load(model_dict[model_type])[1]
    
    # compile list of item id numbers
    ids = np.loadtxt('../data/item_ids.txt', dtype='int32')
    # add randomness to predictions
    np.random.shuffle(ids)
    full_neighbor_vectors = np.zeros((len(neighbor_id_list), 1000))
    for i, uid in enumerate(neighbor_id_list):
        for j, iid in enumerate(ids[:1000]):
            full_neighbor_vectors[i][j] = model.predict(uid, iid)[3]
    composite = full_neighbor_vectors.mean(axis=0)
    indices = composite.argsort()[::-1]
    return ids[indices][:n]


def create_recipe_df(recipe_id_list):
    """
    takes a list of recipe ids and returns the desired information from
    the sql database as a pandas dataframe
    """
    df = pd.DataFrame(data=None, columns=['id','name','description','tags','ingredients'])
    for id_num in recipe_id_list:
        cur.execute(f"SELECT id, name, description, tags, ingredients FROM raw_recipes WHERE id = {id_num}")
        info = cur.fetchall()[0]
        df = df.append({'id':info[0],'name':info[1],'description':info[2],'tags':info[3], 'ingredients':info[4]}, ignore_index=True)
    return df


def get_most_popular():
    """
    returns a preset list of recipe ids from popular recipes
    """
    presets = [419298, 102114, 297701, 232044, 271471, 217489, 493687, 82998, 471951, 109146]
    return presets


# connect to database
conn = psycopg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
cur = conn.cursor()

app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET'])
def home():
    most_popular = get_most_popular()
    information = create_recipe_df(most_popular)
    return render_template('home.html', df=information)

@app.route('/recommend', methods=['POST'])
def recommend():
    vector = np.array([request.form["0"],
                       request.form["1"],
                       request.form["2"],
                       request.form["3"],
                       request.form["4"],
                       request.form["5"],
                       request.form["6"],
                       request.form["7"],
                       request.form["8"],
                       request.form["9"]])
    model = request.form['select_model']
    neighbors = get_neighbors(vector, model)
    start = time.time()
    recommendation_ids = predict_vector(neighbors, model)
    end = time.time()
    df = create_recipe_df(recommendation_ids)
    return render_template('results.html', df=df, user_id=None, model=model, time=round(end - start, 6))

@app.route('/search', methods=['GET'])
def search():
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    user_id = request.form['user_id']
    model = request.form['model']
    recs, time = generate_recommendations(user_id, model)
    #recs = fetch_recommendations(user_id, model)
    print(recs)
    df = create_recipe_df(recs)
    return render_template('results.html', df=df, user_id=user_id, model=model, time=time)


if __name__ == '__main__':
    app.run(debug=False)