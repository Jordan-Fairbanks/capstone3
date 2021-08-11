import psycopg2
import pandas as pd
import numpy as np
import time
import string
import nltk
from flask import Flask, render_template, request, url_for
from surprise import dump
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

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
    return iids[indices][:n], round(end - start, 4)

def reformat(id_list):
    """
    takes a list of id numbers (as strings) and reformats to a string for a sql query
    """
    string = ""
    for idn in id_list:
        string +=  + idn + ", "
    return string[:-2]


def bind(x, y):
    """
    takes to interables and binds them as a list of tuples, just like the zip() function
    but it returns a tuple object
    """
    bound = []
    for ind, val in enumerate(x):
        bound.append((val, y[ind]))
    return bound


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
    


def predict_vector(neighbor_id_list, model_type, n=9):
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
    start = time.time()
    for i, uid in enumerate(neighbor_id_list):
        for j, iid in enumerate(ids[:1000]):
            full_neighbor_vectors[i][j] = model.predict(uid, iid)[3]
    end = time.time()
    composite = full_neighbor_vectors.mean(axis=0)
    indices = composite.argsort()[::-1]
    return ids[indices][:n], round(end-start, 3)


def create_recipe_df(recipe_id_list):
    """
    takes a list of recipe ids and returns the desired information from
    the sql database as a pandas dataframe
    """
    df = pd.DataFrame(data=None, columns=['id','name','description','tags','ingredients'])
    for id_num in recipe_id_list:
        cur.execute(f"SELECT id, name, description, tags, ingredients FROM raw_recipes WHERE id = {id_num}")
        info = cur.fetchall()[0]
        df = df.append({'id':info[0],'name':string.capwords(info[1]),'description':info[2],'tags':info[3], 'ingredients':info[4]}, ignore_index=True)
    return df

def recipe_dict_for_display(recipe_id):
    """
    takes a recipe id number and returns a dictionary with the desired information
    """
    recipe_info = {}
    cur.execute(f"""SELECT name,
                           description, 
                           nutrition, 
                           ingredients, 
                           steps, 
                           minutes,
                           tags
                    FROM raw_recipes
                    WHERE id = {recipe_id}""")
    info = cur.fetchall()[0]
    recipe_info['name'] = string.capwords(info[0])
    recipe_info['description'] = info[1]
    recipe_info['nutrition'] = str(info[2]).split(", ")
    recipe_info['ingredients'] = info[3]
    recipe_info['steps'] = info[4]
    recipe_info['minutes'] = info[5]
    recipe_info['tags'] = info[6]
    return recipe_info
    
def get_most_popular():
    """
    returns a preset list of recipe ids from popular recipes
    """
    presets = [419298, 102114, 297701, 232044, 271471, 217489, 493687, 82998, 471951, 109146, 96210,444189]
    return presets

def sort_by_category(recipe_ids):
    """
    takes a list of recipe ids and sorts the ids into six categories
    """
    df = create_recipe_df(recipe_ids)
    appetizers, sides, mains, condiments, beverages, desserts = [],[],[],[],[],[]
    app_tags = ['appetizers']
    side_tags = ['side-dishes','salads']
    main_tags =['dinner','main-dish']
    cond_tags = ['condiments','condiments-etc','sauces','canning']
    bev_tags = ['beverages','cocktails','smoothies']
    des_tags = ['desserts','chocolate','frozen-desserts']

    # search for tags in each recipe and split into appropriate category
    for ind, row in df.iterrows():
        for tag in row['tags'].split(', '):
            if tag in app_tags:
                appetizers.append(row['id'])
                break
            elif tag in side_tags:
                sides.append(row['id'])
                break
            elif tag in main_tags:
                mains.append(row['id'])
                break
            elif tag in cond_tags:
                condiments.append(row['id'])
                break
            elif tag in bev_tags:
                beverages.append(row['id'])
                break
            elif tag in des_tags:
                desserts.append(row['id'])
                break
    return (appetizers, sides, mains, condiments, beverages, desserts)

def analyze_ingredients(recipes_sorted_by_category):
    """
    takes a tuple of lists of recipe ids and analyzes the most common ingredients for each
    set and returns a list of lists of the most commonly recommended ingredients for each category
    """
    stopWords = set(nltk.corpus.stopwords.words('english'))
    extra_stops = ['sauce','meat','oil','reduced','grill','grilled','hidden','mix','seasoning','heavy','head','active',
                    'marinated','nonfat','healthy','light','long','karo','old','original','non','liquid','obrien','leave',
                    'medium','well','rare','new','dry','accent','powder','pam','prepared','low','packed','shortening','salt',
                    'minced','chopped','maid','mixed','frozen','fried','baked','baby','marinade','wild','stick','white','green',
                    'red','vinegar','superfine','simple','syrup','sprig','leaf','double','triple','kikkoman','fresh','pods',
                    'whole','paste','product','bottled','sweet','water','ground','instant','pure','rolled','seed','zest','sticks',
                    'dark','pitted','little','big','granny','leg','flavored','flavor','sprayed','organic','gala','reeses','raw','57',
                    'whipping','plain','fat','whole','small','navy','unsweetened','sweetened','flaked','brown','natural','inch',
                    'lopez','strong','steeped','hot','sea','mashed','stewed','ice','purpose','baking','gum','margarine','butter','crushed',
                    'italian','hard','dried','fine','coarse','leaves','nonstick','mini','miniature','rising','self','acting','double','quick',
                    'foster','recipe','sour','provence','french','freshly','picked','pie','morsel','wrappers','aluminum','stock','pieces',
                    'shake','relish','black','mountain','monterey','preserves','southern','whites','pico','de','gayo','processed',
                    'wedge','golden','smith','smoked','granny','mild','sparkling','paper','sharp','wild','large','cracked','links',
                    'wooden','minute','powdered','popsicle','american','puff','mexican','fashioned','seasoned','cold','smart','refrigerated',
                    'purple','chips','jack','kernel','extract','without','kidney','whip','hash','free','giant','miracle','garden','virgin',
                    'pace','greek','granulated','kosher','heinz','extra','ritz','meal','peel','english','elbow','eye','ms','mr','mrs','miss','mister',
                    'sir','spray','food','devils','gold','table','halves','sliced','hearts','homestyle','homemade','xanthan','gum','locust','bean',
                    'guar','aged','stuffed','filled','rolled','roll','flavoring','artificial','natural','soup','hunts','hair','condensed','dairy',
                    'grain','granules','confectioners','confectioner','salted','cured','semi','semisweet','bittersweet','bitter','part','napa','blanched',
                    'see','star','pound','substitute','skim','puree','flat','fresco','toasted','roasted','comfort','evaporated','refried',
                    'unsalted','hoagie','sub','grinder','sandwich','hero','crust','crusts','pink']   
    for extra in extra_stops:
        stopWords.add(extra)
    cat_list = []
    start = time.time()
    for recipe_ids in recipes_sorted_by_category:
        if len(recipe_ids) == 0:
            continue
        df = create_recipe_df(recipe_ids)
        cv = CountVectorizer(stop_words=stopWords,ngram_range=(1,1),max_features=500)
        mat = cv.fit_transform(df['ingredients'].values)
        all_ingredients = np.array(cv.get_feature_names())
        most_frequent = mat.todense().sum(axis=0).argsort()[::-1][:15]
        category = all_ingredients[most_frequent[0]][0]
        cat_list.append(category[:20])
    end = time.time()
    return cat_list , round(end-start, 3)
# connect to database
conn = psycopg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
cur = conn.cursor()
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/')
@app.route('/home', methods=['GET'])
def home():
    imgs = ['static/img/recipe1.jpeg','static/img/recipe2.jpeg','static/img/recipe3.jpeg','static/img/recipe4.jpeg','static/img/recipe5.jpeg',
            'static/img/recipe6.jpeg','static/img/recipe7.jpeg','static/img/recipe8.jpeg','static/img/recipe9.jpeg','static/img/recipe10.jpeg',
            'static/img/recipe11.jpeg','static/img/recipe12.jpeg',]
    most_popular = get_most_popular()
    information = create_recipe_df(most_popular)
    return render_template('home.html', df=information, imgs=imgs)

@app.route('/recommend', methods=['POST'])
def recommend():
    display_dict = {'svd':'SVD',
                  'nmf':'NMF',
                  'knnb':'KNN Basic',
                  'knnwm':'KNN With Means',
                  'coclust':'Co-Clustering',
                  'slope1':'Slope One'}
    vector = np.array([request.form["id0"],
                       request.form["id1"],
                       request.form["id2"],
                       request.form["id3"],
                       request.form["id4"],
                       request.form["id5"],
                       request.form["id6"],
                       request.form["id7"],
                       request.form["id8"],
                       request.form["id9"],
                       request.form["id10"],
                       request.form["id11"]])
    model = request.form['select_model']
    neighbors = get_neighbors(vector, model)
    recommendation_ids, run_time = predict_vector(neighbors, model, n=9)
    df = create_recipe_df(recommendation_ids)
    return render_template('results.html', df=df, user_id=None, model=display_dict[model], n=1000,time=run_time)

@app.route('/search', methods=['GET'])
def search():
    return render_template('search.html')

@app.route('/results', methods=['POST'])
def results():
    display_dict = {'svd':'SVD',
                  'nmf':'NMF',
                  'knnb':'KNN Basic',
                  'knnwm':'KNN With Means',
                  'coclust':'Co-Clustering',
                  'slope1':'Slope One'}
    user_id = request.form['user_id']
    model = request.form['model']
    recs, time = generate_recommendations(user_id, model,n=9)
    df = create_recipe_df(recs)
    return render_template('results.html', df=df, user_id=user_id, model=display_dict[model],n=5000, time=time)

@app.route('/recipe', methods=["POST"])
def recipe():
    recipe_id = request.form['recipe_id']
    print(recipe_id)
    recipe = recipe_dict_for_display(recipe_id)
    return render_template("recipe.html", recipe=recipe)


@app.route("/generate_menu",methods=['GET'])
def menu_search():
    return render_template('menu_search.html')


@app.route('/menu_results', methods=['POST'])
def generate_menu():
    display_dict = {'svd':'SVD',
                  'nmf':'NMF',
                  'knnb':'KNN Basic',
                  'knnwm':'KNN With Means',
                  'coclust':'Co-Clustering',
                  'slope1':'Slope One'}
    titles = ["Appetizer","Side","Main","Condiment","Beverage","Dessert"]
    user_id = request.form['user_id']
    model = request.form['model']
    recs, time = generate_recommendations(user_id, model, n=100)
    by_category = sort_by_category(recs)
    ingredients, itime = analyze_ingredients(by_category)
    bound = bind(titles, ingredients)
    return render_template('menu_results.html',ingredients=bound, model=display_dict[model], time=time, itime=itime)

if __name__ == '__main__':
    app.run(debug=False)