#!/usr/bin/env python
# coding: utf-8

# # CS/INFO 5304 Assignment 2: Recommender Systems
# Credit: 35 points + possible bonus (10 points)
# 
# Due date: April 19th, 11:00PM
#  
# The goal of the assignment is to get familiar with different types of recommender systems. Specifically, we are going to build a system that can recommend Yelp businesses to users.

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# In[2]:


pd.set_option('display.max_columns', None)


# In[3]:


users = pd.read_csv("user-business.csv", header=None) # The columns are separated by a space.
users.head() # Rij = 1 if the user has visited&rated that business. Otherwise  Rij = 0.


# In[4]:


users.info() # ratings matrix R, where each row corresponds to a user and each column corresponds to a business


# In[5]:


users.shape # 14397 active users and 1000 popular businesses on Yelp (column name = index #)


# In[6]:


users.describe()


# In[7]:


busi = pd.read_csv("business.csv", header=None)
busi.head() # businesses, in the same order as the columns of matrix R (user df)


# In[8]:


busi.shape # row number corresponds to index in user matrix


# In[9]:


busi.columns = ['Business']


# In[10]:


busi.head()


# ## Part A: user – user recommender system [10 points]
# In this assignment we are going to implement three types of recommender systems. We are then going to compare the results of these systems for the 4th user (index starting from 1) of the dataset. Let’s call him Alex (row index 3). Based on Alex’s behavior on the other businesses, you need to give Alex recommendations on the first 100 businesses. We will then see if our recommendations match what Alex had in fact visited.
# 
# Let S denote the set of the first 100 businesses (the first 100 columns of the matrix). From all the businesses in S, which are the five that have the highest similarity scores (rAlex,b) for Alex? What are their similarity scores? In case of ties between two businesses, choose the one with a smaller index. Do not write the index of the businesses, write their names using the file business.csv.
# 

# In[11]:


S = users.iloc[:,:100]
S.head()


# In[12]:


S.shape


# In[13]:


# we have erased the first 100 entries of Alex’s row in the matrix, and replaced them by 0s
S.iloc[3].value_counts()


# In[14]:


R = users.iloc[:,100:]
R.head()


# In[15]:


# Alex
ALEX = R.iloc[3].copy()


# In[16]:


ALEX


# In[17]:


ALEX.value_counts()


# In[18]:


R = R.drop(3)


# In[19]:


R.shape


# In[20]:


R.head()


# In[21]:


from numpy import dot
from numpy.linalg import norm


# In[22]:


def cos_sim(a,b):
    return dot(a, b)/(norm(a)*norm(b))


# In[23]:


# find similarity between users (cosine similarity of other users with Alex excluding the first 100 businesses)
cosims = []
for user in R.index.values:
    cosims += [cos_sim(ALEX, R.loc[user])]


# In[24]:


# NOTE TO SELF
# .loc --> index name (R.loc[14396] works)
# .iloc --> index (R.iloc[14396] IndexError: single positional indexer is out-of-bounds)


# In[25]:


cosims


# In[26]:


len(cosims)


# In[27]:


# ratings matrix R without Alex
compare = users.drop(3)
compare.shape


# In[28]:


# take transpose of original ratings matrix (with first 100)
# and multiply (not dot) it with the similarities 
ratings = compare.T * cosims
ratings.shape


# In[30]:


ratings.head()


# In[31]:


# sum over rows to get vector of predictions for Alex's ratings (only care about first 100 rows)
preds = ratings.sum(axis=1)


# In[32]:


preds.shape


# In[33]:


preds.max()


# In[34]:


unknown = preds[:100]
unknown.max()


# In[35]:


unknown.sort_values(ascending=False)[:10] # inplace=False by default


# In[36]:


uu_top5 = unknown.sort_values(ascending=False)[:5].index
uu_top5


# In[37]:


busi.loc[uu_top5]


# From all the businesses in S, the five that have the highest similarity scores for Alex in order from highest to lowest score are Papi's Cuban & Caribbean Grill, Seven Lamps, Loca Luna, Farm Burger, and Piece of Cake.

# In[38]:


print("Their similarity scores are: ", unknown[uu_top5].values)


# ## Part B: item – item recommender system [10 points]
# From all the businesses in S (first 100 businesses), which are the five that have the highest similarity scores for Alex?  In case of ties between two businesses, choose the one with a smaller index. Again, hand in the names of the businesses and their similarity score.

# In[39]:


# find items (businesses) with similar ratings
compare.shape # exclude the entries of Alex


# In[40]:


# Alex's ratings from the original matrix (1000)
origALEX = users.iloc[3].copy()


# In[41]:


origALEX.value_counts()


# In[42]:


origALEX.shape


# In[43]:


from sklearn.metrics.pairwise import cosine_similarity


# In[44]:


matrix = pd.DataFrame(cosine_similarity(users.T))


# In[45]:


matrix.shape


# In[46]:


# take dot product of the similarities (1000,1000) with Alex's ratings from the original matrix (1000)
# to get predictions for Alex's ratings of the businesses
item_preds = dot(matrix, origALEX)


# In[47]:


item_preds.shape


# In[48]:


item_preds


# In[49]:


ipreds = pd.Series(item_preds)


# In[50]:


ipreds.max()


# In[51]:


# select first 100
ipreds = ipreds[:100]
ipreds


# In[52]:


ipreds.max()


# In[53]:


# sort and look at top (check for ties) 
ipreds.sort_values(ascending=False)[:10]


# In[54]:


# select the top 5 and get their indices corresponding to which businesses
ii_top5 = ipreds.sort_values(ascending=False)[:5].index
ii_top5


# In[55]:


busi.loc[ii_top5] # top 5 recommended businesses for Alex in S


# In[56]:


print("Their similarity scores are: ", ipreds[ii_top5].values)


# ## Part C: Latent hidden model recommender system [15 points]
# Latent model recommender system is the most popular type of recommender system in the market today. Here we perform a matrix factorization of the ratings matrix R into two matrices U and V where U is considered as the user features matrix and V is the movie features matrix. Note that the features are ‘hidden’ and need not be understandable to users. Hence the name latent hidden model. (refer slides for more information)
# 
# The latent model can be implemented by performing a singular value decomposition (SVD) that factors the matrix into three matrices
#  
# R = U Σ VT
#  
# where R is user ratings matrix, U is the user “features” matrix, Σ is the diagonal matrix of singular values (essentially weights), and VT is the movie “features” matrix. U and VT are orthogonal, and represent different things. U represents how much users “like” each feature and VT represents how relevant each feature is to each business.
# To get the lower rank approximation, we take these matrices and keep only the top k features (k factors), which we think of as the k most important underlying taste and preference vectors.
#  
# With k set to 10, perform SVD to identify the U and V matrices. You can then multiply the matrices to estimate the following
#  
# R* = U Σ VT
# 
# From the R* matrix, select the top 5 businesses for Alex in S (first 100 businesses). In case of ties between two businesses, choose the one with a smaller index. Again, hand in the names of the businesses and their similarity score.
#  
# Hint: You can use SVD in surprise package, or numpy, scipy

# In[57]:


from numpy.linalg import svd


# In[58]:


u,s,vh = svd(users) # user factors matrix, singular values/weights, movie factors matrix


# In[59]:


u.shape


# In[60]:


s.shape


# In[61]:


s[:10] # descending order


# In[62]:


diag = np.diag(s)
diag.shape


# In[63]:


vh.shape


# In[64]:


# keep only the top k=10 features (10 factors)
ku = u[:, :10]
ks = diag[:10, :10]
kvh = vh[:10,:]
ku.shape, ks.shape, kvh.shape


# In[65]:


recon = ku @ ks @ kvh


# In[66]:


# reconstructed ratings matrix should be (14397,1000) with Alex's 100 missing entries filled in
recon.shape


# In[67]:


recon[3][:100]


# In[68]:


recon[3][:100].max()


# In[69]:


rec = pd.Series(recon[3][:100])


# In[70]:


rec.max()


# In[71]:


# sort and look at top (check for ties) 
rec.sort_values(ascending=False)[:10]


# In[72]:


# select the top 5 and get their indices corresponding to which businesses
svd_top5 = rec.sort_values(ascending=False)[:5].index
svd_top5


# In[73]:


busi.loc[svd_top5] # names of the businesses


# In[74]:


# similarity scores
print("Their similarity scores are: ", rec[svd_top5].values)

