import pandas as pd
import numpy as np
from surprise import dataset
from surprise.prediction_algorithms.matrix_factorization import SVD
from surprise.reader import Reader 
import psycopg2 as pg2
from collections import defaultdict
from re import sub


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

def array_to_string(arr):
    """
    takes a numpy array of integers and converts it to a comma separated string
    """
    string = ''
    for i in arr:
        string += str(i) + ', '
    return string[:-2]

def sql_create_statement(arr):
    """
    takes an array and returns a comma separated string thats can be used in a SQL 
    statement as the column names/types
    """
    string = ''
    for i in arr:
        string += 'recipe_' + str(i) + ' INTEGER, '
    return string[:-2]

def sql_insert_statement(arr):
    """
    takes an array and returns a comma separated string that can be used in a SQL
    statement as columns names, with no types
    """
    string = ''
    for i in arr:
        string += ' recipe_' + str(i) + ', '
    return string[:-2]


if __name__ == "__main__":
    # Establish connection to database
    conn = pg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
    cur = conn.cursor()

    print('CONNECTED TO DATABASE...')
    # fetch data from all the users in the database
    cur.execute("""SELECT interactions.user_id, recipe_id, rating 
                   FROM interactions 
                   JOIN (SELECT user_id, COUNT(recipe_id) as count 
                         FROM interactions 
                         GROUP BY user_id)as inter1 
                   ON interactions.user_id = inter1.user_id 
                   WHERE count > 200
                   Limit 1000""")
    interactions = np.array(cur.fetchall())
    print('INTERACTIONS FETCHED...')
    # generate sparse matrix
    sparse_matrix = pd.DataFrame(data=interactions, columns=['user_id','recipe_id','rating'] )
    
    print('SPARSE MARTRIX CREATED...')
    # set up surprise dataset object
    reader = Reader(sep=',', rating_scale=(1,5))
    data = dataset.Dataset.load_from_df(sparse_matrix,reader)

    print('DATASET LOADED...')

    # First train an SVD algorithm on ratings data (code snippets from surprise documentation)
    algo = SVD()
    trainset = data.build_full_trainset()
    algo.fit(trainset)
    print('MODEL TRAINED...')

    # Then predict ratings for all pairs (u, i) that are NOT in the training set.
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    print('PREDICTIONS MADE...')

    # delete old predictions table
    cur.execute('TRUNCATE predictions; DROP TABLE predictions')
    conn.commit()
    print('OLD TABLE predictions DELETED...')

    # vectorize user predictions
    predictions = np.array(predictions)
    predictions_df = pd.DataFrame(data=None,
                                  index=np.unique(predictions[:,0]), 
                                  columns=np.unique(predictions[:,1]))
    for pred in predictions:
        predictions_df.loc[pred[0]][pred[1]] = pred[3]
    predictions_df.fillna(0, inplace=True)
    print(f"NUMBER OF USERS:{predictions_df.shape[0]}")
    print(f"NUMBER OF ITEMS:{predictions_df.shape[1]}")
    # create new table
    cur.execute('CREATE TABLE predictions(id INTEGER, %s)' % (sql_create_statement(np.unique(predictions[:,1]))))
    conn.commit()
    print('NEW TABLE predictions MADE...')
    # insert vectors into sql table
    i = 0
    for user in predictions_df.index:
        cur.execute(f"INSERT INTO predictions(id , {sql_insert_statement(np.unique(predictions[:,1]))})\
                      VALUES({str(user)},{array_to_string(predictions_df.loc[user])})")
        i += 1
        conn.commit()
    print(f"inserted {i} user vectors into TABLE predictions")

    top_n = get_top_n(predictions, n=15)
    print('RECOMMENDATIONS GENERATED...')

    # delete old recommendations
    cur.execute('TRUNCATE recommendations; DROP TABLE recommendations')
    conn.commit()
    print('OLD TABLE recommendations DROPPED...')

    # recreate table
    cur.execute('CREATE TABLE recommendations(id INTEGER, recommendation VARCHAR)')
    print('NEW TABLE recommendations MADE...')

    # insert new predictions into table
    i = 0
    for user, rating_list in top_n.items():
        parsed = '\''
        for recipe_id, rating in rating_list:
            parsed += str(recipe_id) + ', '
        parsed = parsed[:-2] + '\''
        cur.execute("INSERT INTO recommendations(id, recommendation)\
                     VALUES(%s)" % (str(user) + ', ' + parsed))
        i += 1
        conn.commit()
    print(f"recommendations for {i} users inserted into TABLE recommendations")
    
	# close connection
    conn.close()
    print('CONNECTION CLOSED')
