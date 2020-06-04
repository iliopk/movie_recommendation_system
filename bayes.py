import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import Counter
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

targetUser = int(input("Enter user id: "))

userMovieMatrix = pd.pivot_table(ratings, index='userId', columns='movieId', values='rating')
userMovieMatrix=userMovieMatrix.T.fillna(userMovieMatrix.mean(axis=1)).T
userMovieMatrix=userMovieMatrix.astype('int64')
movieUserMatrix=userMovieMatrix.T

def bayes_prediction(matrix,target,r,type ='user'):
    y_pred_d = dict()
    if type=='user':
        # define samples , target, test and train sets
        X = matrix.drop([target], axis=1)
        y = matrix[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        # create model, fit values and make prediction
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_d = {x: y for x, y in zip(y_test.index, y_pred)}
        # print accuracy of model
        print(" model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    elif type=='movie':
        matrix2 = matrix.iloc[:, :1000]
        user = [i for i in matrix.index if i != target]
        for col in matrix2.columns:
            y = matrix[col]
            X = matrix.drop([col], axis=1)
            X_test = (pd.DataFrame(X.loc[target, :])).T
            X_train = X.drop([target])
            y_test = y.drop(user)
            y_train = y.drop(target)
            model = MultinomialNB(alpha=0.1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_d[col]=y_pred[0]
            print(" model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)
    return y_pred_d


def bayes_recommendation(y_pr,r,target,m):
    goodscore_ids = []
    # get ids of movies that predicted rating is greater than 3
    for key, value in y_pr.items():
        if value >= 3:
            goodscore_ids.append(key)
    # create list with ids of movies that targetUser has already seen
    moviesTargetUserHasSeen = (r.loc[r['userId'] == target]['movieId']).tolist()
    # create list of movies ids that user has not seen but predicted score is grater than 4
    rec_ids = [id for id in goodscore_ids if id not in moviesTargetUserHasSeen]
    # get titles of recommended movies
    rec_titles = m.loc[m['movieId'].isin(rec_ids)]['title']
    return rec_titles


def comb_pred(userBased,movieBased,r,m,target):
    goodscore_ids=[]
    comb=Counter(userBased)+Counter(movieBased)
    comb=dict(comb)
    comb={k:v/2 for k,v in comb.items()}
    for key,value in comb.items():
        if value >= 3:
            goodscore_ids.append(key)

    # create list with ids of movies that targetUser has already seen
    moviesTargetUserHasSeen = (r.loc[r['userId'] == target]['movieId']).tolist()
    # create list of movies ids that user has not seen but predicted score is grater than 3
    rec_ids = [id for id in goodscore_ids if id not in moviesTargetUserHasSeen]
    # get titles of recommended movies
    rec_titles = m.loc[m['movieId'].isin(rec_ids)]['title']
    return rec_titles,comb


def evaluation(r, target, pred):
    moviesTargetUserHasSeen = (r.loc[r['userId'] == target]['movieId']).tolist()
    pred_ids = list(pred.keys())
    non_common = set(moviesTargetUserHasSeen) ^ set(pred_ids)
    common = list(set(pred_ids).intersection(moviesTargetUserHasSeen))
    data = r.loc[r['userId'] == target]
    data2 = data[data['movieId'].isin(common)]
    data3 = data[data['movieId'].isin(common)]['rating'].tolist()

    new_dict = {key: value for key, value in pred.items() if key in common}
    pred_ratings = list(new_dict.values())

    data4 = [1 if x >= 3.0 else 0 for x in data3]
    data5 = [1 if x >= 3.0 else 0 for x in pred_ratings]
    acc = accuracy_score(data4, data5)

    return acc

#-------------------------------------------klisi sinartisewn--------------------------------

#user-based predictions with bayes
user_pred=bayes_prediction(movieUserMatrix,targetUser,ratings,type ='user')
user_rec=bayes_recommendation(user_pred,ratings,targetUser,movies)

#item-based predictions with bayes
movie_pred=bayes_prediction(userMovieMatrix,targetUser,ratings,type ='movie')
movie_rec=bayes_recommendation(movie_pred,ratings,targetUser,movies)

#user-based and item-based combined score prediction with bayes
comb,avg=comb_pred(user_pred,movie_pred,ratings,movies,targetUser)

#accuracy calculation
movie_accuracy=evaluation(ratings,targetUser,movie_pred)
user_accuracy=evaluation(ratings,targetUser,user_pred)
comb_accuracy=evaluation(ratings,targetUser,avg)