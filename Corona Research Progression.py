#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
import os
import PyPDF2 as ppd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# Data for more than **5,000** reseach papers on Coronavirus from year 1968 to 2020. The data set size is **6GB**. Goal is to understand how our knowledge of Coronavirus has developed in the last 50 years
# 

# In[2]:


stop_words = set(stopwords.words("english"))
year_list = [i for i in range(1900,2050)]
year_list_str = [str(i) for i in range(1900,2050)]
stop_words = stop_words | set(year_list) | set(year_list_str)


# In research papers, there is often a mention of earlier years reseach as scientist build upon the existing knowledge. Due to these mentions, year comes out to be top word some times.

# In[3]:


#pip install jupyter_to_medium
#import jupyter_to_medium


# In[187]:


def remove_stopwords(doc):
    word_tokens = word_tokenize(doc)
    filtered_sentence = [w for w in word_tokens if not w in stop_words]    
    filtered_sentence = [] 
    for w in word_tokens:
        if w not in stop_words:  
            filtered_sentence.append(w) 
    return ' '.join(filtered_sentence)

ps = PorterStemmer()

def stemmer(doc):
    words = word_tokenize(doc)
    stemmed_words = []
    for word in words:
        stemmed_words.append(ps.stem(word))
    return ' '.join(stemmed_words)


# change the value to black
def black_color_func(word, font_size, position,orientation,random_state=None, **kwargs):
    return("hsl(0,100%, 1%)")
# set the wordcloud background color to white
# set max_words to 1000
# set width and height to higher quality, 3000 x 2000

def plot_wordcloud(year):
    wordcloud = WordCloud(font_path = None, background_color="white", width=3000, height=2000, max_words=500).generate_from_frequencies(data[year])
    # set the word color to black
    wordcloud.recolor(color_func = black_color_func)
    # set the figsize
    plt.figure(figsize=[15,10])
    # plot the wordcloud
    plt.imshow(wordcloud, interpolation="bilinear")
    # remove plot axes
    plt.axis("off")
    # save the image
    plt.savefig(path +"wc_images\\" + str(year) + '_wc.png')
    
def text_from_word(y,p,txt):
    doc = data_agg[y-1968]
    pos = [i.start() for i in re.finditer(txt, doc)]
    d1 = pos[p]-500
    d2 = pos[p]+500
    print(pos)
    print(doc[d1:d2])
    
    
def word_mentions(y,txt):
    doc = corpus[y-1968]
    pos = [i.start() for i in re.finditer(txt, doc)]
    return len(pos)


# In[6]:


path = "C:\\Users\\apathak\\Documents\\Additional\\Stuff\\corona\\"
path1 = "E:\\CoronaVirusPapers\\"


# ### import all the 5,329 PDFs
# 
# Data is obtained from https://the-eye.eu/public/Papers/CoronaVirusPapers/

# In[9]:


get_ipython().run_cell_magic('capture', '', '\nfile_names = []\nfor file in os.listdir(path1):\n    if file.endswith(".pdf"):\n        file_names.append(file)        \n        \ndocs = []\nfor file_name in file_names:\n    file = open(path1 + file_name, \'rb\')\n    try:\n        file_reader = ppd.PdfFileReader(file)\n    except:\n        print(str("file_read_error") + "-" + str(file_name))\n        pass\n    try:\n        num_pages = file_reader.numPages\n    except:\n        print(str("num_page_error") + "-" + str(file_name))\n        pass\n    pdf_pages =  []\n    for i in range(num_pages):\n        try: \n            pdf_pages.append(file_reader.getPage(i).extractText())\n        except:\n            pass\n    doc = \' \'.join(pdf_pages)\n    docs.append(doc)\n    file.close()')


# In[354]:


print(len(file_names))


# ### Data Processing 1

# In[12]:


data = docs
data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]
data = [re.sub('\s+', ' ', sent) for sent in data]
data = [re.sub("\'", "", sent) for sent in data]


# In[13]:


year_from_fname = [f[:4] for f in file_names]
docs_len = [len(doc) for doc in data]
df_from_fname = pd.DataFrame(file_names)
df_from_fname.columns = ['file_names']
df_from_fname['year'] = year_from_fname
df_from_fname['doc_len'] = docs_len


# In[14]:


d1 = df_from_fname.groupby(['year'])['file_names'].count().reset_index()
plt.figure(figsize= (16,4))
plt.xticks(rotation = 60, color= 'black')
sns.lineplot('year', 'file_names', data = d1)
plt.title('# Research papers published on Coronavirus')
plt.xlabel('')
plt.ylabel('# research papers')


# Research on Coronovirus has picked up more interest in the last 20 years
# 
# ###### A peak in # research papers in 2003 and 2004 is because a distinct coronavirus (SARS-CoV) was identified as the etiological agent of SARS, an acute pulmonary syndrome characterized by an atypical pneumonia that results in progressive respiratory failure and death in close to 10% of infected individuals
# 
# ###### 2020 data is incomplete

# ### Data aggregation by year
# Combining all the content from each research paper in a particular year to create one single document for each year. This changes the shape of our corpus from 5329 documents to just 53 documents (1967-2020)

# In[ ]:


data_agg = []
data_a= []

for i,doc in enumerate(data):
    if i ==0:
        data_a.append(doc)
    elif year_from_fname[i] == year_from_fname[i-1]:
        if i == len(data)-1:
            data_a.append(doc)
            data_agg.append(' '.join(data_a))
        else: 
            data_a.append(doc)
    else:
        data_agg.append(' '.join(data_a))
        data_a = []
        data_a.append(doc)
        
del data_a
print(len(data_agg))
print([len(d) for d in data_agg])


# ### Data Processing 2

# In[16]:


corpus = []
for doc in data_agg: 
    corpus.append(stemmer(remove_stopwords(doc)))
    
print(len(corpus))
print([len(d) for d in corpus])


# ## Term frequency metric from Corpus

# In[211]:


tf_vectorizer = CountVectorizer(stop_words='english', ngram_range = (1,2), max_df = .6, min_df = .01, max_features = 10000)
X = tf_vectorizer.fit_transform(corpus)
feature_names = tf_vectorizer.get_feature_names()
dense = X.todense()
denselist = dense.tolist()
tf_df = pd.DataFrame(denselist, columns=feature_names)
tf_df.head()


# #### Processing term frequency dataframe

# In[212]:


unique_year_from_fname = list(set(year_from_fname))
year_list = [int(year) for year in unique_year_from_fname]
year_list.sort()

tf_data = tf_df.transpose()
tf_data.columns = year_list
tf_data = data[~data.index.isin(year_list_str)]

## filter
s_l = []
for s in tf_data.index:
    if 'http' in s or 'jstor' in s:
        s_l.append(s)
tf_data = tf_data[~tf_data.index.isin(s_l)]

print(tf_data.shape)
tf_data.tail()


# ## TFIDF metric from corpus
# 
# Word cloud can be created through term frequency as well, but it wouldn't be as meaningful as TF-IDF metric. Instead of more common words, we need to understand most salient words in each year to fully comprehend the progress on corona research
# 
# limiting number of features to 10K, instead of 2.5M
# 
# Taking bigrams as well to account for words like SARS COV

# In[205]:


tfidf_vectorizer = TfidfVectorizer(stop_words='english', ngram_range = (1,2), max_df = .6, min_df = .01, max_features = 10000)
X = tfidf_vectorizer.fit_transform(corpus)
feature_names = tfidf_vectorizer.get_feature_names()
dense = X.todense()
denselist = dense.tolist()
tfidf_df = pd.DataFrame(denselist, columns=feature_names)
tfidf_df.head()


# #### Processing TFIDF dataframe

# In[242]:


unique_year_from_fname = list(set(year_from_fname))
year_list = [int(year) for year in unique_year_from_fname]
year_list.sort()

tfidf_data = tfidf_df.transpose()
tfidf_data.columns = year_list
tfidf_data = tfidf_data[~tfidf_data.index.isin(year_list_str)]

## filter
s_l = []
for s in tfidf_data.index:
    if 'http' in s or 'jstor' in s:
        s_l.append(s)
tfidf_data = tfidf_data[~tfidf_data.index.isin(s_l)]

print(tfidf_data.shape)
tfidf_data.head()


# #### Top 15 Salient words by year from 1968 to 2020
# ##### Using TFIDF

# In[33]:


# Find the top 30 words mentioned each year
top_dict = {}
for c in range(53):
    top = tfidf_data.iloc[:,c].sort_values(ascending=False).head(30)
    top_dict[tfidf_data.columns[c]]= list(zip(top.index, top.values))

# Print the top 15 words mentioned each year
for year, top_words in top_dict.items():
    print(year)
    print(', '.join([word for word, count in top_words[0:14]]))
    print('---')


# In[345]:


d1 = pd.DataFrame()
word_list = ['229','nl63', 'c43','hku1', 'sars','mers']
for word in word_list:
    wl = []
    for idx in tfidf_data.index: 
        if word in idx:
            wl.append(idx)
    l = len(wl)
    d1[word] = tfidf_data[tfidf_data.index.isin(wl)].sum().values/l

d1.index = tfidf_data.columns
d1.columns = ['229E','NL63', 'OC43','HKU1', 'SARS','MERS']

plt.figure(figsize= (16,4))
plt.xticks(rotation = 60, color= 'black')
for word in d1.columns: 
    sns.lineplot(d1.index, word, data = d1, label = word)
plt.title('salient mentions of 7 varieties of Coronavirus', fontsize = 30)
plt.xlabel('')
plt.ylabel('')
plt.legend(loc = "best", fontsize = 15)


# In[351]:


d1 = pd.DataFrame()
word_list = ['bats','bromelain','mrna', 'endorphin', 'pheasant', 'amanitin', 'bluecomb']
for word in word_list:
    wl = []
    for idx in tfidf_data.index: 
        if word in idx:
            wl.append(idx)
    l = len(wl)
    d1[word] = tfidf_data[tfidf_data.index.isin(wl)].sum().values/l

d1.index = tfidf_data.columns

plt.figure(figsize= (16,4))
plt.xticks(rotation = 60, color= 'black')
for word in d1.columns: 
    sns.lineplot(d1.index, word, data = d1, label = word)
plt.title('salient mentions of few other words', fontsize = 30)
plt.xlabel('')
plt.ylabel('')
plt.legend(loc = "best", fontsize = 15)


# In[152]:


'''lst = []
for year, top_words in top_dict.items():
    lst = lst + [word for word, count in top_words[0:14]]
    
word_list = lst
total_mentions = 0
word_mention_dict2 = {}

for word in word_list:
    for y in range(1968,2021):
        total_mentions = total_mentions + word_mentions(y,word)
    word_mention_dict2[word] = [total_mentions]
    total_mentions = 0'''


# In[330]:


data_dic = {}
for idx in tf_data.index:
    data_dic[idx] = [tf_data.loc[idx,:].sum()]

d1 = pd.DataFrame(data_dic).transpose()
d1.columns = ['term_freq']
d1.sort_values('term_freq', ascending = False, inplace = True )
d1 = d1.head(20)
plt.figure(figsize= (16,4))
plt.xticks(rotation = 60, color= 'black', fontsize = 15)
sns.barplot(d1.index, 'term_freq', data = d1)
plt.title(' words by most mentions', fontsize = 30)
plt.xlabel('')
plt.ylabel('# mentions')


# In[335]:


data_dic = {}
for idx in tf_data.index:
    if len(idx.split(" ")) > 1:
        data_dic[idx] = [tf_data.loc[idx,:].sum()]

d1 = pd.DataFrame(data_dic).transpose()
d1.columns = ['bigram_term_freq']
d1.sort_values('bigram_term_freq', ascending = False, inplace = True )
d1 = d1.head(20)
plt.figure(figsize= (16,4))
plt.xticks(rotation = 60, color= 'black', fontsize = 15)
sns.barplot(d1.index, 'bigram_term_freq', data = d1)
plt.title('bigrams by most mentions', fontsize = 30 )
plt.xlabel('')
plt.ylabel('# mentions')


# # Wordcloud. One at the end of each decade

# # 1970: 
# 

# In[344]:


plot_wordcloud(year = 1970)


# # 1980

# In[346]:


plot_wordcloud(year = 1980)


# # 1990

# In[347]:


plot_wordcloud(year = 1990)


# # 2000

# In[348]:


plot_wordcloud(year = 2000)


# # 2010

# In[349]:


plot_wordcloud(year = 2010)


# # 2020

# In[350]:


plot_wordcloud(year = 2020)

