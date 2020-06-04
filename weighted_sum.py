import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import accuracy_score

targetUser=int(input("Enter user id: "))

ratings=pd.read_csv('ml-latest-small/ratings.csv')
movies=pd.read_csv('ml-latest-small/movies.csv')

userMovieMatrix = pd.pivot_table(ratings, index='userId', columns='movieId', values='rating')
userMovieMatrix=userMovieMatrix.T.fillna(userMovieMatrix.mean(axis=1)).T
movieUserMatrix=userMovieMatrix.T

userMovieMatrix_e = pd.pivot_table(ratings, index='userId', columns='movieId', values='rating')
movieUserMatrix_e= pd.pivot_table(ratings, index='movieId', columns='userId', values='rating')

def similarity(matrix,metric='cosine',type='user'):
    if type=='user':
        if metric == 'cosine':
            # calculate cosine similarity between users
            sim2 = cosine_similarity(matrix)
            # replace diagonal values with 0.0
            # diagonal values represent similarity between same user
            np.fill_diagonal(sim2, 0.0)
        elif metric == 'euclidean':
            # calculate euclidean distance between users
            sim = euclidean_distances(matrix)
            sim=np.where(sim==0,1,sim)
            sim2 = np.reciprocal(sim)
            np.fill_diagonal(sim2, 0.0)
    elif type=='movie':
        if metric == 'cosine':
            # calculate cosine_similarity between movies
            sim2 = cosine_similarity(matrix)
            # replace diagonal values with 0.0
            # diagonal values represent similarity between same movie
            np.fill_diagonal(sim2, 0.0)
        elif metric == 'euclidean':
            # calculate euclidean distance between movies
            sim = euclidean_distances(matrix)
            sim = np.where(sim == 0, 1, sim)
            sim2=np.reciprocal(sim)
            np.fill_diagonal(sim2, 0.0)

    return sim2,matrix

def prediction(matrix,sim,type='user'):
    if type == 'user':
        dot = pd.DataFrame(np.dot(sim, matrix))
        dot.columns = [i for i in list(matrix)]
        dot.index += 1
        # prediction for user-based collaborative filtering
        pred = dot.div((sim.sum(axis=1)), axis=0)
    elif type == 'movie':
        dot = pd.DataFrame(np.dot(sim, matrix))
        dot.columns = [i for i in list(matrix)]
        dot.index =matrix.index.tolist()
        # prediction for item-based collaborative filtering
        pred=dot.div((sim.sum(axis=1)),axis=0)
    return pred

def recommendation(pred_matrix,target,r,m,type='user'):
    moviesTargetUserHasSeen = (r.loc[r['userId'] == target]['movieId']).tolist()
    if type == 'user':
        # drop columns that represent movies that user has already seen
        pred_matrix = pred_matrix.drop(moviesTargetUserHasSeen, axis=1)
        #pred_matrix=pred_matrix.astype('int64')
        # find top 20 movies with highest score for user based
        top_movies = (pred_matrix.T).nlargest(20, target)
        ids = (top_movies.index).tolist()
        # get titles of top 20 movies
        titles = m.loc[m['movieId'].isin(ids)]['title']
    elif (type == 'movie') | (type == 'comb'):

        # drop columns that represent movies that user has already seen
        pred_matrix = pred_matrix.drop(moviesTargetUserHasSeen)
        #pred_matrix = pred_matrix.astype('int64')
        # find top 20 movies with highest score for user based
        top_movies = (pred_matrix).nlargest(20, target)
        ids = (top_movies.index).tolist()
        # get titles of top 20 movies
        titles = m.loc[m['movieId'].isin(ids)]['title']

    return titles

def comb_pred(matrix_m,matrix_u):
   comb_matrix = (matrix_u.T+matrix_m).div(2)
   return comb_matrix


def evaluation(pred_matrix,norm_matrix,matrix,target):
    max_col=(pred_matrix.max()).max()
    max_row=(pred_matrix.max(axis=1)).max()
    min_col = (pred_matrix.min()).min()
    min_row = (pred_matrix.min(axis=1)).min()
    max_value = max(max_col, max_row)
    min_value = min(min_col, min_row)
    thresh = max_value * 0.7
    # convert dataframe to numpy_array and replace elements
    cos_user_pred_arr = pred_matrix.to_numpy()
    cos_user_pred_arr = np.where(cos_user_pred_arr >= thresh, 1, 0)
    cos_user_pred_rev = pd.DataFrame(data=cos_user_pred_arr)
    cos_user_pred_rev.index = pred_matrix.index
    cos_user_pred_rev.columns = pred_matrix.columns

    cos_user_matrix_arr = norm_matrix.to_numpy()
    cos_user_matrix_arr = np.where(cos_user_matrix_arr >= 3.5, 1, 0)
    cos_user_matrix_rev = pd.DataFrame(data=cos_user_matrix_arr)
    cos_user_matrix_rev.index = matrix.index
    cos_user_matrix_rev.columns = matrix.columns

    m1 = cos_user_matrix_rev.where(matrix.notnull())
    m2 = cos_user_pred_rev.where(matrix.notnull())

    row1 = m1.loc[target, :].dropna()
    row2 = m2.loc[target, :].dropna()

    acc=accuracy_score(row1,row2)

    return acc




#cosine similarity for user-based
cos_sim_user,cos_user_matrix = similarity(userMovieMatrix,metric='cosine',type='user')
#cosine similarity for item-based
cos_sim_movie,cos_movie_matrix = similarity(movieUserMatrix,metric='cosine',type='movie')
#euclidean similarity for user-based
euc_sim_user,euc_user_matrix = similarity(userMovieMatrix,metric='euclidean',type='user')
#euclidean similarity for item-based
euc_sim_movie,euc_movie_matrix = similarity(movieUserMatrix,metric='euclidean',type='movie')


#cosine prediction for user-based
cos_user_pred = prediction(cos_user_matrix,cos_sim_user,type='user')
#cosine prediction for item-based
cos_movie_pred = prediction(cos_movie_matrix,cos_sim_movie,type='movie')
#euclidean prediction for user-based
euc_user_pred = prediction(euc_user_matrix,euc_sim_user,type='user')
#euclidean prediction for item-based
euc_movie_pred = prediction(euc_movie_matrix,euc_sim_movie,type='movie')



#calculate combination matrix cosine
cos_comb=comb_pred(cos_movie_pred,cos_user_pred)
#calculate combination matrix euclidean
euc_comb=comb_pred(euc_movie_pred,euc_user_pred)

#cosine recommendation for combined user & item based
cos_res_comb = recommendation(cos_comb,targetUser,ratings,movies,type='comb')
#cosine recommendation for user-based
cos_res_user = recommendation(cos_user_pred,targetUser,ratings,movies,type='user')
#cosine recommendation for item-based
cos_res_movie = recommendation(cos_movie_pred,targetUser,ratings,movies,type='movie')

#euclidean recommendation for combined user & item based
euc_res_comb = recommendation(euc_comb,targetUser,ratings,movies,type='comb')
#euclidean recommendation for user-based
euc_res_user = recommendation(euc_user_pred,targetUser,ratings,movies,type='user')
#euclidean recommendation for item-based
euc_res_movie = recommendation(euc_movie_pred,targetUser,ratings,movies,type='movie')


#cos_user_accuracy
cos_user_accuracy= evaluation(cos_user_pred,cos_user_matrix,userMovieMatrix_e,targetUser)
#cos_movie_accuracy
cos_movie_accuracy= evaluation(cos_movie_pred.T,cos_movie_matrix.T,userMovieMatrix_e,targetUser)
#cos_comb_accuracy
cos_comb_accuracy= evaluation(cos_comb.T,cos_movie_matrix.T,userMovieMatrix_e,targetUser)

#euc_user_accuracy
euc_user_accuracy= evaluation(euc_user_pred,euc_user_matrix,userMovieMatrix_e,targetUser)
#euc_movie_accuracy
euc_movie_accuracy= evaluation(euc_movie_pred.T,euc_movie_matrix.T,userMovieMatrix_e,targetUser)
#euc_comb_accuracy
euc_comb_accuracy= evaluation(euc_comb.T,euc_movie_matrix.T,userMovieMatrix_e,targetUser)

