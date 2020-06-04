import pandas as pd
from math import log

movies =pd.read_csv('plots_with_tmdb_cleaned.csv')

def create_corpus(m):
    corpus = dict()
    for idx, row in m.iterrows():
        key = m.loc[idx, 'movieId']
        corpus[key] = (m.loc[idx, 'plot_cleaned']).split()
    return corpus

def invdx_plotLenTbl(c):
    invIndex = dict()
    plotLength = dict()
    for plotId in c:
        # build inverted index
        for term in c[plotId]:
            if term in invIndex:
                if plotId in invIndex[term]:
                    invIndex[term][plotId] += 1
                else:
                    invIndex[term][plotId] = 1
            else:
                d = dict()
                d[plotId] = 1
                invIndex[term] = d
        # build plot length table
        length = len(c[plotId])
        plotLength[plotId] = length
    return invIndex,plotLength

def avg_plot_length(pLength):
    # calculate average plot length
    sum = 0
    for plotId, length in pLength.items():
        sum += length
    avgl = float(sum) / float(len(pLength))
    return avgl

def BM25score(cor,inv_index,pLength,avg_length):
    b = 0.75
    k1 = 1.5
    #bm25sim=pd.DataFrame(columns=list(cor.keys())).set_index(list(cor.keys()))
    bm25sim = pd.DataFrame(columns=list(cor.keys()))
    for key,values in cor.items():
        query_result = dict()
        q=cor[key]
        for term in q:
            if term in inv_index:
                nt = len(inv_index[term])  # number of plots containing the term
                for plotId, freq in inv_index[term].items():
                    c = k1 * ((1 - b) + b * pLength[plotId] / avg_length)
                    score = ((k1 * freq) / (freq + c)) * log((len(pLength) + 1) / nt)
                    if plotId in query_result:
                        query_result[plotId] += score
                    else:
                        query_result[plotId] = score
        df = pd.DataFrame([query_result], columns=list(query_result.keys()))
        bm25sim = pd.concat([bm25sim, df], axis=0, ignore_index=True)

    bm25sim.index = list(bm25sim)
    bm25sim.to_csv('BM25_scores.csv', encoding='utf-8', index=False)
    return bm25sim




corpus = create_corpus(movies)
invdx,plotLength = invdx_plotLenTbl(corpus)
avgl = avg_plot_length(plotLength)
result = BM25score(corpus,invdx,plotLength,avgl)

