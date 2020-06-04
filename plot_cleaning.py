import pandas as pd
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
from nltk.corpus import gutenberg
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from string import digits



plots =pd.read_csv('plots_with_tmdb.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')

# Remove rows with no plots
plots2 = plots[plots.plot_tmdb != 'default value']
plots3 = plots2.dropna().reset_index(drop=True)
# Combine plots from imdb and tmdb
plots3['plot'] = plots3['plot_imdb'] + ' ' + plots3['plot_tmdb']

corpus = plots3['plot'].tolist()
titles = plots3['title'].tolist()

# Build a large collection of text to train the tokenizer
text = ""
for file_id in gutenberg.fileids():
    text += gutenberg.raw(file_id)

trainer = PunktTrainer()
trainer.INCLUDE_ALL_COLLOCS = True
# Collects training data from the given text
trainer.train(text)

# Learn parameteres
tokenizer = PunktSentenceTokenizer(trainer.get_params())

# Define stop words
stop_words = stopwords.words('english')
stop_words.append('this')

stemmer = SnowballStemmer("english")
st = PorterStemmer()
synopses_token = []
synopses_stem = []
map_token_stem = []
for syn in corpus:
    # Split each synopses to distinct sentences
    sentence = ' '.join(tokenizer.tokenize(syn)[2:])
    s = ' '.join([w.lower() for w in word_tokenize(sentence) if w.lower() not in stop_words])
    # synopses.append(s)
    # s = ' '.join([w.lower() for w in word_tokenize(sentence) if w.lower() not in stop_words])
    remove_digits = str.maketrans('', '', digits)
    res = s.translate(remove_digits)
    remove_punkt = str.maketrans('', '', punctuation)
    res2 = ' '.join([w for w in res.translate(remove_punkt).split() if len(w) > 2])
    synopses_token.append(res2)
    map_token_stem.extend([(t, st.stem(t)) for t in res2.split()])
    stems = ' '.join([st.stem(t) for t in res2.split()])
    synopses_stem.append(stems)


plots3['plot_cleaned'] = synopses_stem

plots3.to_csv('plots_with_tmdb_cleaned.csv', encoding='utf-8', index=False)



map_DF = pd.DataFrame(map_token_stem, columns=['Token', 'Stem'])
map_DF.drop_duplicates(keep='first', inplace=True)
map_DF.drop_duplicates(subset=['Stem'], keep='first', inplace=True)

map_DF.to_csv('tokenToStem.csv', encoding='utf-8', index=False)