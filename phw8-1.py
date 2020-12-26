from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
import pandas as pd
import re
from sklearn.metrics.pairwise import linear_kernel

# Query
query = "fake news corona vaccine"

# Documents
sentences = list()
sentences.append("WHO is gathering the latest international multilingual scientific findings and knowledge on COVID-19. The global literature cited in the WHO COVID-19 database is updated daily (Monday through Friday) from searches of bibliographic databases, hand searching, and the addition of other expert-referred scientific articles. This database represents a comprehensive multilingual source of current literature on the topic. While it may not be exhaustive, new research is added regularly.")
sentences.append("A COVID-19 vaccine candidate made of tiny artificial particles could be more powerful than other leading varieties at triggering a protective immune response. When the team injected mice with the nanoparticle vaccine, the animals produced virus-blocking antibodies at levels comparable to or greater than those produced by people who had recovered from COVID-19. Mice that received the vaccine produced about ten times more of these antibodies than did rodents vaccinated only with the spike protein, on which many COVID-19 vaccine candidates rely.")
sentences.append("The rise of fake news in the American popular consciousness is one of the remarkable growth stories in recent years—a dizzying climb to make any Silicon Valley unicorn jealous. Just a few years ago, the phrase was meaningless. Today, according to a new Pew Research Center study, Americans rate it as a larger problem than racism, climate change, or terrorism.")
sentences.append("\"Falsehood flies, and the Truth comes limping after it,\" Jonathan Swift once wrote. It was hyperbole three centuries ago. But it is a factual description of social media, according to an ambitious and first-of-its-kind study published Thursday in Science. The massive new study analyzes every major contested news story in English across the span of Twitter’s existence—some 126,000 stories, tweeted by 3 million users, over more than 10 years—and finds that the truth simply cannot compete with hoax and rumor. By every common metric, falsehood consistently dominates the truth on Twitter, the study finds: Fake news and false rumors reach more people, penetrate deeper into the social network, and spread much faster than accurate stories.")
sentences.append("The anti-vaccination movement has gained traction online in recent years, and campaigners opposed to vaccination have moved their focus to making claims relating to the coronavirus. First, a video containing inaccurate claims about coronavirus vaccine trials, made by osteopath Carrie Madej, that has proved popular on social media. Carrie Madej's video makes a false claim that the vaccines will change recipients' DNA (which carries genetic information).\"The Covid-19 vaccines are designed to make us into genetically modified organisms.\" She also claims - without any evidence - that vaccines will \"hook us all up to an artificial intelligence interface\".")
sentences.append(query)
index = [1,2,3,4,5,6]
data = pd.DataFrame({"index":index,"doc":sentences})

# TF-IDF Matrix
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['doc'])
# print(tfidf_matrix.shape) # 6 documents, 197 words

# Cosine Similarity
cosine_sim= linear_kernel(tfidf_matrix,tfidf_matrix)

indices = pd.Series(data.index,index=data['doc']).drop_duplicates()

# match query
def get_recommendations(query, cosine_sim=cosine_sim):
    idx=indices[query]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    rank_indices = [i[0] for i in sim_scores]
    k = 0
    doc = []
    scores = []
    rank = []
    for i in rank_indices:
        doc.append(data['doc'].iloc[i])
        scores.append(sim_scores[k][1])
        rank.append(k)
        k += 1
    result = pd.DataFrame({"rank":rank, "doc": doc, "Similarity": scores})
    print(result)


get_recommendations(query)

