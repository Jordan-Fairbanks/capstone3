import pandas as pd
import numpy as np
import psycopg2 as pg2
from collections import defaultdict
import os
from surprise import dataset, dump, KNNBasic, KNNWithMeans, SlopeOne, CoClustering
from surprise.model_selection.validation import cross_validate
from surprise.prediction_algorithms.matrix_factorization import SVD, NMF
from surprise.reader import Reader 



def array_to_string(arr):
    """
    takes a numpy array of integers and converts it to a comma separated string
    """
    string = ''
    for i in arr:
        string += str(i) + ', '
    return string[:-2]

def parse(rec_list):
    """
    takes a tuple representing a user id and a list of recommended recipes and returns 
    the recipes id_s as a string
    """
    parsed = '\''
    for recipe_id,_ in rec_list:
        parsed += str(recipe_id) + ', '
    parsed = parsed[:-2] + '\''
    return parsed



def train_and_test_model(trainset, model):
    """
    takes a surprise model with a trainset object and a testset object,
    trains the model, then returns the predictions from the test set 
    """
    test_ids = [419298, 102114, 297701, 232044, 271471, 217489, 493687, 82998, 471951, 109146, 96210, 444189]
    model.fit(trainset)
    inner_uids = trainset.all_users()
    raw_uids = [trainset.to_raw_uid(user) for user in inner_uids]
    predictions = np.zeros((len(raw_uids),12))
    for i,uid in enumerate(raw_uids):
        for j, iid in enumerate(test_ids):
            predictions[i][j] = model.predict(uid, iid)[3]
    return predictions, raw_uids


def clear_table(table_name):
    """
    takes a table name as input and clears the corresponding from the SQL database
    """
    # for setting up new models in the pipeline
    cur.execute(f"CREATE TABLE IF NOT EXISTS {table_name} (id INTEGER, num NUMERIC)")
    conn.commit()

    # truncate and clear table, fast and brutal
    cur.execute(f"TRUNCATE TABLE {table_name};DROP TABLE {table_name}")
    conn.commit()
    print(f"DELETED TABLE {table_name}...")
    return

def dump_model(model, name, path):
    """
    takes a trained model, along with a filename and a path, then dumps the model
    """
    file_name = os.path.expanduser(path+name)
    dump.dump(file_name, algo=model)
    print(f"{name} SAVED...")
    return

def create_table(table, columns):
    """
    creates a tabe in the sql database
    """
    cur.execute(f"CREATE TABLE {table}({columns})")
    conn.commit()
    return 

def insert_predictions(table, predictions, uids):
    """
    inserts predictions into a sql table
    """
    columns = """uid, r_419298, r_102114, r_297701, r_232044, r_271471, 
               r_217489, r_493687, r_82998, r_471951, r_109146, r_96210, r_444189"""
    i = 0
    for prediction, uid in zip(predictions, uids):
        cur.execute(f"""INSERT INTO {table}({columns})
                        VALUES ({uid},{array_to_string(prediction)})""")
        i += 1
    conn.commit()
    print(f"INSERTED {i} ROWS INTO TABLE {table}")
    return 

def insert_recommendations(table, recommendations):
    """
    inserts predictions into a sql table
    """
    cur.execute(f"SELECT * FROM {table} LIMIT 0")
    description = np.array(cur.description)
    columns = ', '.join(list(description[:,0]))
    i = 0
    for uid, rec_list in recommendations.items():
        cur.execute(f"""INSERT INTO {table}({columns})
                        VALUES ({uid},{parse(rec_list)})""")
        i += 1
    conn.commit()
    print(f"INSERTED {i} ROWS INTO TABLE {table}")
    return 

# taken from surprise FAQ page
def get_top_n(predictions, n=10):
    """Return the top-N recommendation for each user from a set of predictions.

    Args:
        predictions(list of Prediction objects): The list of predictions, as
            returned by the test method of an algorithm.
        n(int): The number of recommendation to output for each user. Default
            is 10.

    Returns:
    A dict where keys are user (raw) ids and values are lists of tuples:
        [(raw item id, rating estimation), ...] of size n.
    """

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

if __name__ == "__main__":
    # Establish connection to database
    conn = pg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
    cur = conn.cursor()
    print('CONNECTED TO DATABASE...')

    # fetch interactions from database only from users who have rated over 100 items
    cur.execute("""SELECT interactions.user_id, recipe_id, rating 
                   FROM interactions 
                   JOIN (SELECT user_id, COUNT(recipe_id) as count 
                         FROM interactions 
                         GROUP BY user_id)as inter1 
                   ON interactions.user_id = inter1.user_id 
                   WHERE count > 100
                   Limit 20000""")
    interactions = np.array(cur.fetchall())
    print('INTERACTIONS FETCHED...')

    # generate sparse matrix
    sparse_matrix = pd.DataFrame(data=interactions, columns=['user_id','recipe_id','rating'] )
    print('SPARSE MARTRIX CREATED...')
    sparse_matrix['rating'][sparse_matrix['rating'] < 1] = .01

    # set up surprise dataset object
    reader = Reader(sep=',', rating_scale=(1,5))
    data = dataset.Dataset.load_from_df(sparse_matrix,reader)
    print('DATASET LOADED...')

    # prepare trainset and testset for models
    trainset = data.build_full_trainset()
    full_testset = trainset.build_anti_testset()

    # compile and save recipe_ids for use in app
    inner_iids = trainset.all_items()
    raw_iids = np.array([trainset.to_raw_iid(iid) for iid in inner_iids], dtype='int32')
    np.savetxt('data/item_ids.txt', raw_iids)
    
    # First train an SVD algorithm on ratings data (code snippets from surprise documentation)
    svd = SVD()
    svd_predictions, raw_uids = train_and_test_model(trainset, svd)
    print('SVD MODEL TRAINED. PREDICTIONS MADE...')

    nmf = NMF(biased=True)
    nmf_predictions,_ = train_and_test_model(trainset, nmf)
    print('NMF MODEL TRAINED. PREDICTIONS MADE...')

    knnb = KNNBasic(k=20, verbose=False)
    knnb_predictions,_= train_and_test_model(trainset, knnb)
    print('KNNBasic MODEL TRAINED. PREDICTIONS MADE...')

    knnwm = KNNWithMeans(k=20, verbose=False)
    knnwm_predictions,_ = train_and_test_model(trainset, knnwm)
    print('KNNWithMeans MODEL TRAINED. PREDICTIONS MADE...')

    coclust = CoClustering()
    coclust_predictions,_= train_and_test_model(trainset, coclust)
    print('CoClusteting MODEL TRAINED. PREDICTIONS MADE...')

    slope1 = SlopeOne()
    slope1_predictions,_= train_and_test_model(trainset, slope1)
    print('SlopeOne MODEL TRAINED. PREDICTIONS MADE...')

    path = "~/Desktop/coding/galvanize/dsi/capstones/capstone3/models/"
    dump_model(svd, 'SVD_model', path)
    dump_model(nmf, 'NMF_model', path)
    dump_model(knnb, 'KNNBasic_model',path)
    dump_model(knnwm, 'KNNWithMeans_model',path)
    dump_model(coclust, 'CoClustering_model',path)
    dump_model(slope1, 'SlopeOne_model',path)


    # delete old tables
    clear_table('svd_predictions')
    clear_table('nmf_predictions')
    clear_table('knnb_predictions')
    clear_table('knnwm_predictions')
    clear_table('coclust_predictions')
    clear_table('slope1_predictions')

    # set up predictions columns in sql table
    columns = """uid INTEGER,r_419298 NUMERIC, r_102114 NUMERIC, r_297701 NUMERIC, r_232044 NUMERIC, r_271471 NUMERIC, 
               r_217489 NUMERIC, r_493687 NUMERIC, r_82998 NUMERIC, r_471951 NUMERIC, r_109146 NUMERIC, r_96210 NUMERIC, r_444189 NUMERIC"""    
    # create tables 
    create_table('svd_predictions', columns)
    create_table('nmf_predictions', columns)
    create_table('knnb_predictions', columns)
    create_table('knnwm_predictions', columns)
    create_table('coclust_predictions', columns)
    create_table('slope1_predictions', columns)
    print('NEW TABLES CREATED...')

    # insert predictions into table
    insert_predictions('svd_predictions', svd_predictions, raw_uids)
    insert_predictions('nmf_predictions', knnb_predictions, raw_uids)
    insert_predictions('knnb_predictions', knnb_predictions, raw_uids)
    insert_predictions('knnwm_predictions', knnwm_predictions, raw_uids)
    insert_predictions('coclust_predictions', coclust_predictions, raw_uids)
    insert_predictions('slope1_predictions', slope1_predictions, raw_uids)

    # print number of unique users and items
    num_users = trainset.n_users
    num_items = trainset.n_items
    print(f"UNIQUE USERS IN TABLE: {num_users}\nUNIQUE ITEMS IN TABLE: {num_items}")
    
	# close connection
    conn.close()
    print('CONNECTION CLOSED')

    #compare models 
    print('SVD\n----------')
    cross_validate(svd, data, verbose=True)
    print('----------\n')
    print('NMF\n----------')
    cross_validate(nmf, data, verbose=True)
    print('----------\n')
    print('KNN Basic\n----------')
    cross_validate(knnb, data, verbose=True)
    print('----------\n')
    print('KNN With Means\n----------')
    cross_validate(knnwm, data, verbose=True)
    print('----------\n')
    print('CoClustering\n----------')
    cross_validate(coclust, data, verbose=True)
    print('----------\n')
    print('Slope One\n----------')
    cross_validate(slope1, data, verbose=True)
    print('----------\n')