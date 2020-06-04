import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.metrics import accuracy_score

targetUser=int(input("Enter user id: "))

BM25_matrix = pd.read_csv('BM25_scores.csv').fillna(0)
BM25_matrix.index = list(BM25_matrix)
tfidf_matrix = pd.read_csv('TFIDF_scores.csv')
tfidf_matrix.index = list(BM25_matrix)
user_genre_matrix = pd.read_csv('user_genre_matrix.csv')
user_genre_matrix2=user_genre_matrix.drop(columns=['userId'])
ratings = pd.read_csv('ml-latest-small/ratings.csv')
termToStem = pd.read_csv('tokenToStem.csv')
tfidf_terms = pd.read_csv('TFIDF_terms.csv')
terms_list=tfidf_terms['0'].tolist()
plots = pd.read_csv('plots_with_tmdb_cleaned.csv')

def perform_PCA(data_matrix):
    pca = PCA(n_components=15)
    pca_components = pca.fit_transform(data_matrix)
    # create pca dataframe
    pca_components_DF = pd.DataFrame(pca_components)
    explained_variance = pca.explained_variance_ratio_
    features = range(pca.n_components)
    return pca_components_DF, explained_variance, features

def plot_variance(explained_variance, features):
    fig = plt.figure()
    # Plot the explained variances
    bar = plt.bar(features, explained_variance, color='#9370DB')
    plt.xlabel('PCA features')
    plt.ylabel('Variance ')
    plt.xticks(features)
    return fig

def plot_PCA(pca_components_DF):
    fig = plt.figure()
    # Plot PCA
    plt.scatter(pca_components_DF[0], pca_components_DF[1], alpha=.1, s=5)
    plt.title('PCA')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    #plt.savefig(str('PCA.jpg'), format='jpg', dpi=800)

    return fig



def plot_elbow_curve(data_matrix):
    Nc = range(1, 20)
    inertias = []
    for c in Nc:
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=c)
        # Fit model to samples
        model.fit(data_matrix)
        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)

    fig = plt.figure()
    plt.plot(Nc, inertias, '-o', color='black')
    plt.xlabel('number of clusters, c')
    plt.ylabel('inertia')
    plt.xticks(Nc)
    #plt.savefig(str('Elbow_Curve_inertias.jpg'), format='jpg', dpi=800)

    return fig



def kmeans_clustering(pca_components_DF,type='user'):
    if type == 'user':
        kmeans = KMeans(n_clusters=5)
        kmeans.fit(pca_components_DF.iloc[:, :2])
        labels = kmeans.predict(pca_components_DF.iloc[:, :2])
        centroids = kmeans.cluster_centers_
        clusters = kmeans.labels_.tolist()
    elif type == 'movie_tfidf':
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(pca_components_DF.iloc[:, :2])
        labels = kmeans.predict(pca_components_DF.iloc[:, :2])
        centroids = kmeans.cluster_centers_
        clusters = kmeans.labels_.tolist()
    else:
        kmeans = KMeans(n_clusters=4)
        kmeans.fit(pca_components_DF.iloc[:, :2])
        labels = kmeans.predict(pca_components_DF.iloc[:, :2])
        centroids = kmeans.cluster_centers_
        clusters = kmeans.labels_.tolist()

    return labels,centroids,clusters


def plot_kmeans(pca_components_DF,pca_labels):
    colmap = {1: '#8A2BE2', 2: '#FF4040', 3: '#458B00', 4: '#DC143C', 5: '#006400',
              6: '#FF8C00', 7: '#E9967A', 8: '#F08080', 9: '#87CEFA', 10: '#BA55D3',
              11: '#AEEEEE', 12: '#308014', 13:'#8B3E2F', 14:'#CAFF70', 15:'#FFB6C1', 16:'#FF34B3', 17:'#191970'}

    # Plot k-means
    colors = list(map(lambda x: colmap[x + 1], pca_labels))
    fig = plt.figure()
    plt.scatter(pca_components_DF[0], pca_components_DF[1], color=colors, alpha=0.7, s=6)
    #for idx, centroid in enumerate(user_centroids):
        #plt.scatter(*centroid, color='#000000', s=6)

    plt.title('K-Means')

    return fig


# kaleitai mono me tf-idf
def most_common_words(data_matrix,labels,tfidf_terms ,termToStem):
    df = pd.DataFrame(data_matrix).groupby(labels).mean()
    words=dict()
    for i, r in df.iterrows():
        keys = [tfidf_terms [t] for t in np.argsort(r)[-10:]]
        words[i]=','.join((termToStem.loc[termToStem['Stem'].isin(keys)]['Token']).tolist())

    return words




def userBased_prediction(labels,matrix,r,target):
    label_DF = pd.DataFrame(data=labels)
    df = label_DF.join(matrix['userId'])
    df.columns = ['cluster', 'userId']
    rat_ext = pd.merge(df, r.drop(columns='timestamp'), how='inner', on='userId')
    userMovieMatrix = pd.pivot_table(rat_ext, index='userId', columns='movieId', values='rating')
    true_rating=userMovieMatrix.loc[target,:]
    c=df.loc[df['userId']==target]['cluster']
    temp = rat_ext.loc[rat_ext['cluster'] == int(c)]
    userMovieMatrix2 = pd.pivot_table(temp, index='userId', columns='movieId', values='rating')
    userMovieMatrix3 = userMovieMatrix2.drop(target)
    prediction = userMovieMatrix3.mean()


    return prediction,true_rating

def itemBased_prediction(labels,p,r,target):
    label_DF = pd.DataFrame(data=labels)
    movieToCluster = label_DF.join(p['movieId'])
    movieToCluster.columns = ['Cluster', 'movieId']
    ratings2 = r.loc[r['movieId'].isin(p['movieId'].values.tolist())]
    rat_ext = pd.merge(movieToCluster, ratings2.drop(columns='timestamp'), how='inner', on='movieId')
    userMovieMatrix = pd.pivot_table(rat_ext, index='userId', columns='movieId', values='rating')
    true_rating = userMovieMatrix.loc[target, :]
    # 3) count number of movies user has seen in each cluster
    counts = rat_ext.groupby(['userId', 'Cluster']).size().reset_index(name='counts')
    # 4) assign user to cluster containing bigger count of movies he has seen
    userToCluster = counts[counts['counts'] == counts.groupby(['userId'])['counts'].transform(max)].drop_duplicates(
        subset='userId', keep='first')
    userToCluster = userToCluster.sort_values(by=['Cluster'])
    c = userToCluster.loc[userToCluster['userId'] == target]['Cluster']
    temp = rat_ext.loc[rat_ext['Cluster'] == int(c)]
    userMovieMatrix2 = pd.pivot_table(temp, index='userId', columns='movieId', values='rating')
    userMovieMatrix3=userMovieMatrix2.drop(target)
    prediction = userMovieMatrix3.mean()

    return prediction,true_rating


def recommendation(prediction,rating,p):
    pr=prediction.dropna().index.tolist()
    r=rating.dropna().index.tolist()
    non_common = list(set(r) ^ set(pr))
    rec=prediction.reindex(non_common).nlargest(20).index.tolist()
    titles=p.loc[p['movieId'].isin(rec)]['title']
    return titles


def evaluation(pred_matrix, matrix, target):
    moviesTargetUserHasSeen = matrix.dropna()
    predicted_movies = pred_matrix.dropna()

    seen_ids = moviesTargetUserHasSeen.index.tolist()
    pred_ids = predicted_movies.index.tolist()
    common = list(set(pred_ids).intersection(seen_ids))

    l1 = (moviesTargetUserHasSeen[common]).tolist()
    l2 = (predicted_movies[common]).tolist()

    l3 = [1 if x >= 3.5 else 0 for x in l1]
    l4 = [1 if x >= 3.5 else 0 for x in l2]

    acc = accuracy_score(l3, l4)
    return acc

# PCA and clustering
user_components,user_variance, user_features= perform_PCA(user_genre_matrix2)
user_labels,user_centroids,user_clusters = kmeans_clustering(user_components, type='user')

bm25_components,bm25_variance, bm25_features= perform_PCA(BM25_matrix)
bm25_labels,bm25_centroids,bm25_clusters = kmeans_clustering(bm25_components, type='movie_bm25')

tfidf_components,tfidf_variance,tfidf_features= perform_PCA(tfidf_matrix)
tfidf_labels,tfidf_centroids,tfidf_clusters = kmeans_clustering(tfidf_components, type='movie_tfidf')


# prediction
user_prediction,user_rating=userBased_prediction(user_labels,user_genre_matrix2,ratings,targetUser)
tfidf_prediction,tfidf_rating=itemBased_prediction(tfidf_labels,plots,ratings,targetUser)
bm25_prediction,bm25_rating=itemBased_prediction(bm25_labels,plots,ratings,targetUser)

# recommendation
user_rec=recommendation(user_prediction,user_rating,plots)
tfidf_rec=recommendation(tfidf_prediction,tfidf_rating,plots)
bm25_rec=recommendation(bm25_prediction,bm25_rating,plots)


# accuracy calculation
user_accuracy= evaluation(user_prediction,user_rating,targetUser)
tfidf_accuracy= evaluation(tfidf_prediction,tfidf_rating,targetUser)
bm25_accuracy= evaluation(bm25_prediction,bm25_rating,targetUser)

'''
plot_elbow_curve(user_genre_matrix2)
plot_elbow_curve(tfidf_matrix)
plot_elbow_curve(BM25_matrix)
most_common_words(tfidf_matrix ,tfidf_labels, terms_list ,termToStem)
'''

'''plot_variance(user_variance, user_features)
plot_PCA(user_components)
plot_variance(tfidf_variance,tfidf_features)
plot_PCA(tfidf_components)
plot_variance(bm25_variance, bm25_features)
plot_PCA(bm25_components)'''












