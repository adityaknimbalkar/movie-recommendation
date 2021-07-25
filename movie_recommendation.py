#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer


# In[3]:


df=pd.read_csv(r"C:\Users\Adi\Downloads\movie_data.csv")


# In[10]:


df.head(10)


# In[4]:


df['Movie_id'] = range(0, 0+len(df))
df


# In[5]:


#Get a count of the number of rows /Movies in the data set and the number of columns
df.shape


# In[6]:


#Create a list of important columns fo the recommendation engine
columns=['actor_1_name','actor_2_name','actor_3_name','director_name','genres','movie_title']


# In[7]:


#Show the Data
df[columns].head(3)


# In[8]:


#Check for any missing values in the important columns
df[columns].isnull().values.any()


# In[9]:


#Create a Function to combine the values of the important columns into a single string
def get_important_features(data):
  important_features = []
  for i in range(0, data.shape[0]):
    important_features.append(data['actor_1_name'][i]+','+data['actor_2_name'][i]+','+data['actor_3_name'][i]+','+data['director_name'][i]+' '+data['genres'][i]+' '+data['movie_title'][i])

  return important_features


# In[10]:


#Create a column to hold the combined strings
df['important_features'] = get_important_features(df)


# In[28]:


#Show the data
df.head(70)


# In[34]:


#Convert the text to a matrix of token counts
#CountVectorizer in Python
#In order to use textual data for predictive modeling, the text must be parsed to remove certain words – this process is called tokenization. These words need to then be encoded as integers, or floating-point values, for use as inputs in machine learning algorithms. This process is called feature extraction (or vectorization).

#Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts. It also enables the ​pre-processing of text data prior to generating the vector representation. This functionality makes it a highly flexible feature representation module for text.
cm=CountVectorizer().fit_transform(df['important_features'])
print(cm)


# In[17]:


#Get the shap of the cosine similarity matrix
cs.shape


# In[16]:


#Get the cosine similarity matrix from the count matrix

#Ex: Compute cosine similarity between samples in X and Y.
#Cosine similarity, or the cosine kernel, computes similarity as the normalized dot product of X and Y

cs =cosine_similarity(cm)
#Print the cosine similarity matrix 
print(cs)


# In[21]:


#Get the title of the movie that the user likes
title = input()


# In[22]:


#title = df.get_value(title,'movie_title')
#movie_id = df[df.movie_title == title]['Movie_id'].values[0]
movie_id = df.loc[df['movie_title'].str.contains(title,case=False)]
movie_id=movie_id.Movie_id.values[0]
movie_id


# In[23]:


#movie_id=df['Movie_id'][df['movie_title']==title].to_numpy()[0]

#movie_id = 2

#Find the movies id
#movie_id  = df.get_value(4, 15, takeable = True)

#Create a list of enumerations for the similarity score (movie_id,similarity score)
#The enumerate() method adds counter to an iterable and returns it (the enumerate object).
scores = list(enumerate(cs[movie_id]))


# In[36]:


#Sort the list
sorted_scores=sorted(scores, key =lambda x:x[1],reverse= True)
sorted_scores =sorted_scores[1:]


# In[27]:


#Print the sorted scores
print(sorted_scores)


# In[25]:


#Create a loop to print the first 7 similar movies
j=0
print('The 5 most recommended movies to ', title,'are:\n')
for item in sorted_scores :
  movie_title =df[df.Movie_id ==item[0]]['movie_title'].values[0]
  print(j+1,movie_title)
  j=j+1
  if j>4:
    break


# In[ ]:




