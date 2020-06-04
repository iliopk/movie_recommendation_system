import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt



# Function for splitting and mapping genre categories
def splitNmap(category_list):
    n_categories = len(category_list)
    return pd.Series(dict(zip(category_list, [1] * n_categories)))

ratings = pd.read_csv('ml-latest-small/ratings.csv')
movies = pd.read_csv('ml-latest-small/movies.csv')

# Movie ratings equal or greater than 3.5
good_ratings = ratings.loc[ratings['rating'] >= 3.5]

# DF containing all movies and their mapping to different genres
genres = movies.genres.dropna().str.split('|').apply(splitNmap)
# Add movieID to the DF
genres.insert(0, 'movieId', movies['movieId'])

# DF containing values of DF-genres plus userID and rating of movies higher or equal to 3.5
genres_extended_after = pd.merge(good_ratings.drop(columns='timestamp'), genres, how='inner', on='movieId').sort_values(by=['userId'])
# Set DF's index to default value i.e 0,1,...,n-1
genres_extended_after = genres_extended_after.reset_index(drop=True)



# DF containing values of DF-genres plus userID and rating of all movies
genres_extended_before = pd.merge(ratings.drop(columns='timestamp'), genres, how='inner', on='movieId').sort_values(by=['userId'])
# Set DF's index to default value i.e 0,1,...,n-1
genres_extended_before = genres_extended_before.reset_index(drop=True)

# Multiply each row(movie) with respective rating
mul_after = genres_extended_after.iloc[:,3:].multiply(genres_extended_after['rating'], axis="index")
mul_before = genres_extended_before.iloc[:,3:].multiply(genres_extended_before['rating'], axis="index")

# DF containing userID and rating for each movie's genre
temp_after = pd.concat([genres_extended_after[['userId']],mul_after], axis=1)
temp_before = pd.concat([genres_extended_before[['userId']],mul_before], axis=1)

# User rating per category after removing movies with rating <3.5
rating_genre_after = temp_after.groupby('userId').mean().reset_index()

# User rating per category considering all movies
rating_genre_before = temp_before.groupby('userId').mean().reset_index()

# DF containing number of movies each user has watched per category
df_count_after = pd.DataFrame()
df_count_before = pd.DataFrame()
for id in list(set(temp_after['userId'])):
   # Calculate number of movies each user has watched per category and in total
   df_count_after = df_count_after.append(temp_after.loc[temp_after['userId'] == id].count(), ignore_index=True)

df_count_after =df_count_after[temp_after.columns.tolist()]
df_count_after = df_count_after.rename(columns={'userId': 'userTotalCount'})
df_count_after.insert(0, 'userId', list(set(temp_after['userId'])))

for id in list(set(temp_before['userId'])):
   # Calculate number of movies each user has watched per category and in total
   df_count_before = df_count_before.append(temp_before.loc[temp_before['userId'] == id].count(), ignore_index=True)

df_count_before =df_count_before[temp_before.columns.tolist()]
df_count_before = df_count_before.rename(columns={'userId': 'userTotalCount'})
df_count_before.insert(0, 'userId', list(set(temp_before['userId'])))



# DF containing user preference percetange per category after removing movies with rating <3.5
dataset=(df_count_after.iloc[:,2:]).div(df_count_after['userTotalCount'],axis=0)
dataset.insert(0, 'userId', list(set(temp_after['userId'])))
#dataset.index=dataset['userId'].tolist()
#dataset=dataset.drop(columns=['userId'])
dataset.to_csv('user_genre_matrix.csv', encoding='utf-8', index=False)

# DF containing average rating per genre after removing movies with rating <3.5
avg_genre_after=rating_genre_after.mean()
# DF containing average rating per genre considering all movies
avg_genre_before=rating_genre_before.mean()

# DF containing number of users per genre
users_per_genre_before=df_count_before.astype(bool).sum(axis=0)
users_per_genre_after=df_count_after.astype(bool).sum(axis=0)




# Set Fonts for plotting
font_ticks = {
        'color':  'black',
        'weight': 'normal',
        'size': 8,
        }

font = {
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }

# Plot num of movies per genre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.bar(list(genres)[1:], genres.count().tolist()[1:], color='#9370DB', align='center')
#ax.set_title('Number of movies per genre', fontdict=font)
ax.set_xticklabels(labels=list(genres)[1:], fontdict=font_ticks, rotation=65)
#ax.set_yticklabels(labels=genres.count().tolist()[1:], fontdict=font_ticks)
ax.set_xlabel('Genres',fontdict=font)
ax.set_ylabel('Number Of Movies',fontdict=font)
plt.tight_layout()
fig.savefig(str('numOfmoviesPerGenre2.jpg'), format='jpg', dpi=800)


# Plot rating per genre before and after removing movies with rating <3.5
fig1, ax1 = plt.subplots()
N = 20
width = 0.35
ind = np.arange(N)  # the x locations for the groups
after = ax1.bar(ind+width, avg_genre_after.tolist()[1:],width, label='After',color='#8B1C62')
before = ax1.bar(ind, avg_genre_before.tolist()[1:], width, label='Before',color='#9370DB')
ax1.set_xticks(ind + width / 2)
ax1.set_xlabel('Genres',fontdict=font)
ax1.set_ylabel('Rating',fontdict=font)
#ax1.set_title('Rating Per Genre', fontdict=font)
ax1.set_xticklabels(avg_genre_after.index.tolist()[1:], fontdict=font_ticks, rotation=65)
ax1.legend()
fig1.tight_layout()
fig1.savefig(str('ratingPerGenre.jpg'), format='jpg', dpi=800)

# Plot users per genre before and after removing movies with rating <3.5
fig2, ax2 = plt.subplots()
N = 20
width = 0.35
ind = np.arange(N)  # the x locations for the groups
a = ax2.bar(ind+width, users_per_genre_after.tolist()[2:],width, label='After',color='#8B1C62')
b = ax2.bar(ind, users_per_genre_before.tolist()[2:], width, label='Before',color='#9370DB')
ax2.set_xticks(ind + width / 2)
ax2.set_xlabel('Genres',fontdict=font)
ax2.set_ylabel('Users',fontdict=font)
#ax2.set_title('Users Per Genre')
ax2.set_xticklabels(users_per_genre_after.index.tolist()[2:], fontdict=font_ticks, rotation=65)
ax2.legend()
fig2.tight_layout()
fig2.savefig(str('UsersPerGenre.jpg'), format='jpg', dpi=800)
