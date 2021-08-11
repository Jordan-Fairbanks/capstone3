import pandas as pd
import psycopg2 as pg2
from re import sub

def features_as_string(row):
    """
    takes a row from a dataframe and returns a comma separated string of the values
    formatted for a sql query
    """
    string = ''
    for val in row:
        if type(val) == str:
            if val[0] == "[":
                string += ("\'" + sub("\'","",val[1:-1]) + "\', ")
            elif val.__contains__('\''):
                string += ("\'" + sub("'", "", val) + "\', ")
            else:
                string += ("\'" + val + "\', ")
        elif val == None or pd.isna(val):
            string += "\'None\',"
        else:
            string += (str(val) + ', ')
    return string[:-2]


if __name__ == "__main__":
    
    # Establish connection to database
    conn = pg2.connect(dbname='Food', user='postgres', password='galvanize', host='localhost', port='5432')
    cur = conn.cursor()
    print('CONNECTION ESTABLISHED\n-----------------')
    
    try:
        cur.execute("CREATE TABLE IF NOT EXISTS raw_recipes(name VARCHAR,\
                                                            id INTEGER,\
                                                            minutes INTEGER,\
                                                            contributor_id INTEGER,\
                                                            submitted VARCHAR,\
                                                            tags VARCHAR,\
                                                            nutrition VARCHAR,\
                                                            n_steps INTEGER, \
                                                            steps VARCHAR,\
                                                            description VARCHAR,\
                                                            ingredients VARCHAR,\
                                                            n_ingredients INTEGER)")
    except (Exception, pg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        print('Could not create table: raw_recipes')
        
    
    try:
        cur.execute("CREATE TABLE IF NOT EXISTS interactions(user_id INTEGER,\
                                                             recipe_id INTEGER,\
                                                             date VARCHAR,\
                                                             rating INTEGER,\
                                                             review VARCHAR)")
    except (Exception, pg2.DatabaseError) as error:
        print("Error: %s" % error)
        conn.rollback()
        print('Could not create table: interactions')
      
    conn.commit()
    
    # read in csv files
    raw_recipes = pd.read_csv('data/RAW_recipes.csv')
    interactions = pd.read_csv('data/RAW_interactions.csv')
    interactions.fillna('None', inplace=True)


    # write data to interactions table
    i = 0
    for _, row in interactions.iterrows():
        try:
            cur.execute("""INSERT INTO interactions(user_id,\
                                                  recipe_id,\
                                                  date,\
                                                  rating,\ 
                                                  review) \
                        VALUES(%s)""" % features_as_string(row))
            i += 1
        except (Exception, pg2.DatabaseError) as error:
            print("Error: %s" % error)
            conn.rollback()
            print('Could not insert row into TABLE interactions')
        conn.commit()
    
    print(f"{i} items INSERTED INTO TABLE interactions")

    i = 0
    for _, row in raw_recipes.iterrows():
        try:
            cur.execute("""INSERT INTO raw_recipes(name,\
                                                 id,\
                                                 minutes,\
                                                 contributor_id,\
                                                 submitted,\
                                                 tags,\
                                                 nutrition,\
                                                 n_steps,\
                                                 steps,\
                                                 description,\
                                                 ingredients,\
                                                 n_ingredients)\
                          VALUES(%s)""" % features_as_string(row))
            i += 1
        except (Exception, pg2.DatabaseError) as error:
            print(features_as_string(row))
            print("Error: %s" % error)
            conn.rollback()
            print('Could not insert row into TABLE raw_recipes')
        conn.commit()
    print(f"{i} items INSERTED INTO TABLE raw_recipes")

    print('DATABASE COMPILED\n-----------------')
    conn.close()
    print('CONNECTION CLOSED\n-----------------')