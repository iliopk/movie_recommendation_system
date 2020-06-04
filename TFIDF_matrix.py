import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

plots =pd.read_csv('plots_with_tmdb_cleaned.csv')
synopses=plots['plot_cleaned'].tolist()

#define vectorizer parameters
tfidf_vectorizer = TfidfVectorizer(max_df=195, max_features=200000,min_df=2, use_idf=True, ngram_range=(1,1))
# fit the vectorizer to synopses
tfidf_matrix = tfidf_vectorizer.fit_transform(synopses)

print(tfidf_matrix.shape)

terms = tfidf_vectorizer.get_feature_names()

pd.DataFrame(tfidf_matrix.todense()).to_csv("TFIDF_scores.csv", encoding='utf-8', index=False)
pd.DataFrame(terms).to_csv("TFIDF_terms.csv", encoding='utf-8', index=False)

