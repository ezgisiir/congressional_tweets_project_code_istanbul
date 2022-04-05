# This is the code to preprocess the data

import numpy as np
import pandas as pd
import os

# Import the necessary packages

import nltk #language processing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import re
import emoji

from sklearn.feature_extraction.text import CountVectorizer #for bag of words

#import gensim
#from gensim.models import Word2Vec

from sklearn import preprocessing
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from scipy import interpolate
from scipy.spatial import ConvexHull

from sklearn.manifold import SpectralEmbedding

from sklearn.manifold import TSNE

from sklearn.linear_model import LogisticRegressionCV

path = '/scratch/ierez/BlackHoleforNE/kaggle_istanbul/'
os.chdir(path)

import random
random.seed(465) #seed

# Read the data

data_train = pd.read_csv('congressional_tweet_training_data.csv')
data_test = pd.read_csv('congressional_tweet_test_data.csv')

data_train=data_train.dropna()

data_test=data_test.dropna()

data_train.reset_index(drop=True, inplace=True)

data_test.reset_index(drop=True, inplace=True)

# Add a numerical column for the class
data_train['party_class']=[0] * len(data_train)

for i in np.arange(len(data_train)):
    #Because D is the default
    if data_train['party_id'][i]=='D':
        data_train['party_class'][i]=0
    else:
        data_train['party_class'][i]=1

# Start by cleaning the full text of the tweets 
data_train.dropna(subset=['full_text'],inplace=True)

data_train=data_train.reset_index(drop=True)

#Tokenize
tokens_list=list()
for i in np.arange(len(data_train)):
    tokens_list.append(nltk.word_tokenize(str(data_train['full_text'][i])))

tagged_tokens=list()
for i in np.arange(len(tokens_list)):
    tagged_tokens.append(nltk.pos_tag(tokens_list[i]))

def get_wordnet_pos(treebank_tag): #Following from [16]

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement 

lemmatizer = WordNetLemmatizer()

lemma_list=list()

for i in np.arange(len(tagged_tokens)):
    lemma=list()
    for j in np.arange(len(tagged_tokens[i])):
        
        token=tagged_tokens[i][j][0] 
        tag=tagged_tokens[i][j][1]
        
        wntag = get_wordnet_pos(tag)
        if wntag is None:           #do not supply tag in case of None
            lemma.append(lemmatizer.lemmatize(token))
        else:
            lemma.append(lemmatizer.lemmatize(token, pos=wntag))
    lemma_list.append(lemma)
        

# Before removing elements from the string, make all of them lowercase
filtered_lower=list()
for i in np.arange(len(lemma_list)):
    lowered=list()
    for j in np.arange(len(lemma_list[i])):
        lowered.append(lemma_list[i][j].lower())
    filtered_lower.append(lowered)  

# Now to remove stop words like 'the', 'a', 'and' 
stop_words = set(stopwords.words('english'))

filtered_list=list()
for i in np.arange(len(filtered_lower)):
    filtered_list.append([w for w in filtered_lower[i] if not w.lower() in stop_words])

# Remove numbers, words <2 characters, punctuation, links and emojis 

def emoji_free_text(text): # From [9] 
    return emoji.get_emoji_regexp().sub(r'', text)

filtered_elements=list()

for i in np.arange(len(filtered_list)):
    filter_element=list()
    for j in np.arange(len(filtered_list[i])):
        
        element=re.sub(r'\d+', '',filtered_list[i][j])          # Remove numbers
        element=re.sub(r'\b\w{1}\b', '', element)               # Remove <2 characters
        element=re.sub(r'[^\w\s]', '', element)                 # Remove punctuation
        element=re.sub(r'http\S+', '', element)                 # Remove links
        #element=re.sub('/[\u{1f300}-\u{1f5ff}]/', '', element) # Remove symbols
        #element=re.sub('/[\u{1f600}-\u{1f64f}]/','', element)  # Remove emoticons
        emoji_free_text(element)
        
        filter_element.append(element)
    filtered_elements.append(filter_element)   

# Combine the elements back to form sentences
text_clean_list=list()
s=' '

for i in np.arange(len(filtered_elements)):
    text_clean_list.append(s.join(filtered_elements[i]))

# Add a new column of 'text_clean'
data_train.insert(7, "text_clean", text_clean_list)

# Repeat the same for the test data

# Start by cleaning the full text of the tweets 
data_test.dropna(subset=['full_text'],inplace=True) #First, I drop the rows with NA in the text column
data_test

data_test=data_test.reset_index(drop=True) #to fix the indices

#Tokenize
tokens_list=list()
for i in np.arange(len(data_test)):
    tokens_list.append(nltk.word_tokenize(str(data_test['full_text'][i])))

tagged_tokens=list()
for i in np.arange(len(tokens_list)):
    tagged_tokens.append(nltk.pos_tag(tokens_list[i]))

def get_wordnet_pos(treebank_tag): #Following from [16]

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None # for easy if-statement 

lemmatizer = WordNetLemmatizer()

lemma_list=list()

for i in np.arange(len(tagged_tokens)):
    lemma=list()
    for j in np.arange(len(tagged_tokens[i])):
        
        token=tagged_tokens[i][j][0] 
        tag=tagged_tokens[i][j][1]
        
        wntag = get_wordnet_pos(tag)
        if wntag is None:           #do not supply tag in case of None
            lemma.append(lemmatizer.lemmatize(token))
        else:
            lemma.append(lemmatizer.lemmatize(token, pos=wntag))
    lemma_list.append(lemma)
        
# Before removing elements from the string, make all of them lowercase
filtered_lower=list()
for i in np.arange(len(lemma_list)):
    lowered=list()
    for j in np.arange(len(lemma_list[i])):
        lowered.append(lemma_list[i][j].lower())
    filtered_lower.append(lowered)  

# Now to remove stop words like 'the', 'a', 'and' 
stop_words = set(stopwords.words('english'))

filtered_list=list()
for i in np.arange(len(filtered_lower)):
    filtered_list.append([w for w in filtered_lower[i] if not w.lower() in stop_words])

# Remove numbers, words <2 characters, punctuation, links and emojis 

def emoji_free_text(text): # From [9]
    return emoji.get_emoji_regexp().sub(r'', text)

filtered_elements=list()

for i in np.arange(len(filtered_list)):
    filter_element=list()
    for j in np.arange(len(filtered_list[i])):
        
        element=re.sub(r'\d+', '',filtered_list[i][j])          # Remove numbers
        element=re.sub(r'\b\w{1}\b', '', element)               # Remove <2 characters
        element=re.sub(r'[^\w\s]', '', element)                 # Remove punctuation
        element=re.sub(r'http\S+', '', element)                 # Remove links
        #element=re.sub('/[\u{1f300}-\u{1f5ff}]/', '', element) # Remove symbols
        #element=re.sub('/[\u{1f600}-\u{1f64f}]/','', element)  # Remove emoticons
        emoji_free_text(element)
        
        filter_element.append(element)
    filtered_elements.append(filter_element)   

# Combine the elements back to form sentences
text_clean_list=list()
s=' '

for i in np.arange(len(filtered_elements)):
    text_clean_list.append(s.join(filtered_elements[i]))

# Add a new column of 'text_clean'
data_test.insert(7, "text_clean", text_clean_list)

data=pd.concat([data_train,data_test], ignore_index=True)

# Bag of words
# Following from [1]
vectorizer = CountVectorizer()

text_bow = vectorizer.fit_transform(data['text_clean'])
text_bow_df = pd.DataFrame(text_bow.toarray(),columns=vectorizer.get_feature_names())

#Now, include these vectors as a new column named bow_vector in your datasets 

data_train['bow_vector']=text_bow[0:len(data_train)].toarray().tolist()

data_train=pd.concat([data_train,text_bow_df[0:len(data_train)]], axis=1)

#Now, include these vectors as a new column named bow_vector in your datasets 

data_test['bow_vector']=text_bow[len(data_train):len(data)].toarray().tolist()

# To allow indexing to help the concatenate
data_test=pd.concat([data_test,pd.DataFrame(data=text_bow_df[len(data_train):len(data)]).set_index(np.arange(len(data_train)))], axis=1)

# Now apply bag of words to hashtags

#We do not apply preprocessing to that column because there exists a certain form for hashtags

# Bag of words
# Following from [1]
vectorizer = CountVectorizer()

hash_bow = vectorizer.fit_transform(data['hashtags'])
hash_bow_df = pd.DataFrame(hash_bow.toarray(),columns=vectorizer.get_feature_names())


data_train=pd.concat([data_train,hash_bow_df[0:len(data_train)]], axis=1)

#data_test=pd.concat([data_test,hash_bow_df[len(data_train):len(data)]], axis=1)
# To allow indexing to help the concatenate
data_test=pd.concat([data_test,pd.DataFrame(data=hash_bow_df[len(data_train):len(data)]).set_index(np.arange(len(data_train)))], axis=1)

# Dimensionality reduction

# From HW6
#Following from [5] to use sklearn's PCA 

#Standardize before calling PCA
scaler=preprocessing.StandardScaler().fit(text_bow_df)
text_bow_df_scaled=scaler.transform(text_bow_df) 

sklearn_pca = PCA(n_components=3)
pcs = sklearn_pca.fit_transform(text_bow_df_scaled)

# Add pcs to the original data
data_train['bow_pcs']=pcs[0:len(data_train)].tolist()

data_test['bow_pcs']=pcs[len(data_train):len(data)].tolist()

bow_pc1=np.zeros(len(data_train))
bow_pc2=np.zeros(len(data_train))
bow_pc3=np.zeros(len(data_train))

for i in np.arange(len(data_train)):
    bow_pc1[i]=data_train['bow_pcs'][i][0]
    bow_pc2[i]=data_train['bow_pcs'][i][1]
    bow_pc3[i]=data_train['bow_pcs'][i][2]

data_train['bow_pc1']=bow_pc1
data_train['bow_pc2']=bow_pc2
data_train['bow_pc3']=bow_pc3

#Add these columns to the test data

bow_pc1=np.zeros(len(data_test))
bow_pc2=np.zeros(len(data_test))
bow_pc3=np.zeros(len(data_test))

for i in np.arange(len(data_test)):
    bow_pc1[i]=data_test['bow_pcs'][i][0]
    bow_pc2[i]=data_test['bow_pcs'][i][1]
    bow_pc3[i]=data_test['bow_pcs'][i][2]

data_test['bow_pc1']=bow_pc1
data_test['bow_pc2']=bow_pc2
data_test['bow_pc3']=bow_pc3

# visualization_code.py
# Directly copy-pasted from the provided file

mpl.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 10

df = data_train
pal = sns.color_palette("Paired")[:len(set(data_train['party_class']))]
p1 = sns.scatterplot(x="bow_pc1", y='bow_pc2', hue='party_class', palette = pal, data=df, s=250, alpha=0.7, legend=False)

#For each point, we add a text inside the bubble
for line in range(0,df.shape[0]):
     p1.text(df.bow_pc1[line], df.bow_pc2[line], df.party_class[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.suptitle('Two-Dimensional Map (PCA)', fontsize=36)
plt.xlabel('Dimension 1', fontsize=24)
plt.ylabel('Dimension 2', fontsize=24)


for i in df.party_class.unique():
    # get the convex hull
    points = df[df.party_class == i][['bow_pc1', 'bow_pc2']].values
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    # interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], 
                                    u=dist_along, s=0)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    plt.fill(interp_x, interp_y, '--', c=pal[i], alpha=0.2)
    

plt.grid()
plt.savefig(path+'pca.png')


# bow_vector dimensionality reduction with HW6 spectral embedding
#Following from HW6

#Following from sklearn's guides [4]

embedding = SpectralEmbedding(n_components=3)
pcs_embedded = embedding.fit_transform(text_bow_df_scaled) #Use the scaled version of bow_vec column for all of the data

data_train['bow_spec']=pcs_embedded[0:len(data_train)].tolist()

data_test['bow_spec']=pcs_embedded[len(data_train):len(data)].tolist()

bow_pc1=np.zeros(len(data_train))
bow_pc2=np.zeros(len(data_train))
bow_pc3=np.zeros(len(data_train))

for i in np.arange(len(data_train)):
    bow_pc1[i]=data_train['bow_spec'][i][0]
    bow_pc2[i]=data_train['bow_spec'][i][1]
    bow_pc3[i]=data_train['bow_spec'][i][2]

data_train['bow_spec1']=bow_pc1
data_train['bow_spec2']=bow_pc2
data_train['bow_spec3']=bow_pc3

#Add these columns to the test data

bow_pc1=np.zeros(len(data_test))
bow_pc2=np.zeros(len(data_test))
bow_pc3=np.zeros(len(data_test))

for i in np.arange(len(data_test)):
    bow_pc1[i]=data_test['bow_spec'][i][0]
    bow_pc2[i]=data_test['bow_spec'][i][1]
    bow_pc3[i]=data_test['bow_spec'][i][2]

data_test['bow_spec1']=bow_pc1
data_test['bow_spec2']=bow_pc2
data_test['bow_spec3']=bow_pc3

# visualization_code.py
# Directly copy-pasted from the provided file

mpl.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 10

df = data_train
pal = sns.color_palette("Paired")[:len(set(data_train['party_class']))]
p1 = sns.scatterplot(x="bow_spec1", y='bow_spec2', hue='party_class', palette = pal, data=df, s=250, alpha=0.7, legend=False)

#For each point, we add a text inside the bubble
for line in range(0,df.shape[0]):
     p1.text(df.bow_spec1[line], df.bow_spec2[line], df.party_class[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.suptitle('Two-Dimensional Map (Spectral Embedding)', fontsize=36)
plt.xlabel('Dimension 1', fontsize=24)
plt.ylabel('Dimension 2', fontsize=24)


for i in df.party_class.unique():
    # get the convex hull
    points = df[df.party_class == i][['bow_spec1', 'bow_spec2']].values
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    # interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], 
                                    u=dist_along, s=0)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    plt.fill(interp_x, interp_y, '--', c=pal[i], alpha=0.2)
    

plt.grid()
plt.savefig(path+'spec.png')

# bow_vector dimensionality reduction with HW6 t-SNE
#Following from HW6

pcs_tsne = TSNE(n_components=3).fit_transform(text_bow_df_scaled)

data_train['bow_tsne']=pcs_tsne[0:len(data_train)].tolist()

data_test['bow_tsne']=pcs_tsne[len(data_train):len(data)].tolist()

bow_pc1=np.zeros(len(data_train))
bow_pc2=np.zeros(len(data_train))
bow_pc3=np.zeros(len(data_train))

for i in np.arange(len(data_train)):
    bow_pc1[i]=data_train['bow_tsne'][i][0]
    bow_pc2[i]=data_train['bow_tsne'][i][1]
    bow_pc3[i]=data_train['bow_tsne'][i][2]

data_train['bow_tsne1']=bow_pc1
data_train['bow_tsne2']=bow_pc2
data_train['bow_tsne3']=bow_pc3

#Add these columns to the test data

bow_pc1=np.zeros(len(data_test))
bow_pc2=np.zeros(len(data_test))
bow_pc3=np.zeros(len(data_test))

for i in np.arange(len(data_test)):
    bow_pc1[i]=data_test['bow_tsne'][i][0]
    bow_pc2[i]=data_test['bow_tsne'][i][1]
    bow_pc3[i]=data_test['bow_tsne'][i][2]

data_test['bow_tsne1']=bow_pc1
data_test['bow_tsne2']=bow_pc2
data_test['bow_tsne3']=bow_pc3

# visualization_code.py
# Directly copy-pasted from the provided file

mpl.rcParams['figure.dpi'] = 600
plt.rcParams['figure.figsize'] = (20.0, 10.0)
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 10

df = data_train
pal = sns.color_palette("Paired")[:len(set(data_train['party_class']))]
p1 = sns.scatterplot(x="bow_tsne1", y='bow_tsne2', hue='party_class', palette = pal, data=df, s=250, alpha=0.7, legend=False)

#For each point, we add a text inside the bubble
for line in range(0,df.shape[0]):
     p1.text(df.bow_tsne1[line], df.bow_tsne2[line], df.party_class[line], horizontalalignment='left', size='medium', color='black', weight='semibold')

plt.suptitle('Two-Dimensional Map (t-SNE)', fontsize=36)
plt.xlabel('Dimension 1', fontsize=24)
plt.ylabel('Dimension 2', fontsize=24)


for i in df.party_class.unique():
    # get the convex hull
    points = df[df.party_class == i][['bow_tsne1', 'bow_tsne2']].values
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    # interpolate
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    spline, u = interpolate.splprep([x_hull, y_hull], 
                                    u=dist_along, s=0)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    # plot shape
    plt.fill(interp_x, interp_y, '--', c=pal[i], alpha=0.2)
    

plt.grid()
plt.savefig(path+'tsne.png')

# X_train, X_test, y_train to be prepared

X_train=data_train.loc[:, data_train.columns != 'party_id']

X_train=X_train.loc[:, X_train.columns != 'full_text']

X_train=X_train.loc[:, X_train.columns != 'party_class']

X_train=X_train.loc[:, X_train.columns != 'text_clean']

X_train=X_train.loc[:, X_train.columns != 'hashtags']

X_train=X_train.loc[:, X_train.columns != 'bow_vector']

X_train=X_train.loc[:, X_train.columns != 'bow_spec']

X_train=X_train.loc[:, X_train.columns != 'bow_tsne']

X_train=X_train.loc[:, X_train.columns != 'bow_pcs']

y_train=data_train['party_class']

y_train= pd.DataFrame(y_train)

# Two columns with the same name

X_train['Year']=X_train.iloc[:,2]

#Remove the columns named 'year'

del X_train["year"]

X_test=data_test.loc[:, data_test.columns != 'party_id']

X_test=X_test.loc[:, X_test.columns != 'party']

X_test=X_test.loc[:, X_test.columns != 'full_text']

X_test=X_test.loc[:, X_test.columns != 'party_class']

X_test=X_test.loc[:, X_test.columns != 'text_clean']

X_test=X_test.loc[:, X_test.columns != 'hashtags']

X_test=X_test.loc[:, X_test.columns != 'bow_vector']

X_test=X_test.loc[:, X_test.columns != 'bow_spec']

X_test=X_test.loc[:, X_test.columns != 'bow_tsne']

X_test=X_test.loc[:, X_test.columns != 'bow_pcs']

X_test['Year']=X_train.iloc[:,3]

#Remove the columns named 'year'

del X_test["year"]

Id=X_test['Id']

del X_test['Id']

X_train.to_csv('X_train.csv')

X_test.to_csv('X_test.csv')

y_train.to_csv('y_train.csv')

y_test=pd.DataFrame(data=Id,column='Id')

y_test.to_csv('y_test.csv')
