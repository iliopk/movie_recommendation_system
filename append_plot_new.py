import pandas as pd
from bs4 import BeautifulSoup
import requests
import re

movies=pd.read_csv('ml-latest-small/links.csv',converters={'imdbId': lambda x: str(x), 'tmdbId': lambda x: str(x)})
movies['plot_imdb']='default value'
movies['plot_tmdb']='default value'

for idx, row in movies.iterrows():
    line=movies.loc[idx,'imdbId']
    page = requests.get("http://www.imdb.com/title/tt"+line)
    soup = BeautifulSoup(page.content, 'html.parser')
    desc = soup.find("meta", property="og:description")
    #desc = soup.findAll(attrs={"name": re.compile(r"description", re.I)})
    if desc:
        movies.at[idx, 'plot_imdb'] = desc['content']
    else:
        movies.at[idx, 'plot_imdb'] = 'N/A'

    line2 = movies.loc[idx, 'tmdbId']
    page2 = requests.get("https://www.themoviedb.org/movie/" + line2)
    soup2 = BeautifulSoup(page2.content, 'html.parser')
    desc2 = soup2.findAll('div', class_='overview')
    # print(desc[0]['content'])  # .encode('utf-8'))
    if desc2:
        movies.at[idx, 'plot_tmdb'] = desc2[0].p.text
    else:
        movies.at[idx, 'plot_tmdb'] = 'N/A'


movies2=pd.read_csv('ml-latest-small/movies.csv')
movies2['plot_imdb']=movies['plot_imdb'].tolist()
movies2['plot_tmdb']=movies['plot_tmdb'].tolist()
movies2['genres'] = movies2['genres'].str.replace('|', ' ')
movies2.to_csv('plots_with_tmdb.csv', encoding='utf-8', index=False)