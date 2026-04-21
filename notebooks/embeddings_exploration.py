# In[1]:
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
import contractions
import pickle

# In[2]:
with open('ModelA_cls_embeddings_1', 'rb') as handle:
    a = pickle.load(handle)

# In[3]:
with open('ModelA_cls_embeddings_2', 'rb') as handle:
    b = pickle.load(handle)

# In[4]:
#meam of all cls embeddings

data= np.empty((0,1024))

for i in range(len(a)):
    c=np.asarray(a[i])
    d=np.mean(c,axis=0)
    d = np.expand_dims(d, axis=0)
   
    
    data = np.append(data, d, axis=0)
    
for i in range(len(b)):
    c=np.asarray(b[i])
    d=np.mean(c,axis=0)
    d = np.expand_dims(d, axis=0)
   
    
    data = np.append(data, d, axis=0)

# In[5]:
#longitudinal cls embeddings

print(len(a[0]))
pad = 200-len(a[0])
padding=np.zeros((pad,1024),dtype=float)
print(padding.shape)
d=np.asarray(a[0])
print(d.shape)
d=np.append(d,padding,axis=0)
d = np.expand_dims(d, axis=0)
print(d.shape)

for i in range(1,len(a)):
    pad = 200-len(a[i])
    padding=np.zeros((pad,1024),dtype=float)
    temp=np.asarray(a[i])
    temp=np.append(temp,padding,axis=0)
    temp = np.expand_dims(temp, axis=0)
    d=np.append(d,temp,axis=0)

for i in range(0,len(b)):
    pad = 200-len(b[i])
    padding=np.zeros((pad,1024),dtype=float)
    temp=np.asarray(b[i])
    temp=np.append(temp,padding,axis=0)
    temp = np.expand_dims(temp, axis=0)
    d=np.append(d,temp,axis=0)

print(d.shape)

# In[6]:
# Here, we read in the training data, preprocess it, and prepare it for training.
# This involves cleaning, merging datasets if necessary, and scaling numerical labels.

# Read in training data
train_data = pd.read_json('./data/train_data_JO.json')

# Preprocess and standardize labels
# Optional: Merge with sentence data if needed and drop redundant columns (removed)
# Example of preprocessing (adjust according to your dataset specifics)
train_data = train_data[train_data['PNEOA_scaled'] != '.']
scaler = StandardScaler()
train_data['PNEOA_scaled_new'] = scaler.fit_transform(train_data['PNEOA_scaled'].to_numpy().reshape(-1, 1))
train_data = train_data[train_data['text'] != '']
train_data['text'] = train_data['text'].apply(lambda x: contractions.fix(x))

# Prepare DataFrame for training
train_texts = train_data['text'].astype(str).tolist()
train_labels = train_data['PNEOA_scaled_new'].astype(float).tolist()
train_data_df = pd.DataFrame({'text': train_texts, 'labels': train_labels})

# In[7]:
train_data_df.shape

# In[8]:
train_data_df.head()

# In[9]:
train_data_df['class']=0
train_data_df['class2']=0

# In[10]:
mean_score=train_data_df['labels'].mean()

# In[11]:
train_data_df[train_data_df['labels']>=mean_score].shape

# In[12]:
train_data_df[train_data_df['labels']<mean_score].shape

# In[13]:
train_data_df.loc[train_data_df['labels']>=mean_score,'class']=1

# In[14]:
mean2=train_data_df[train_data_df['labels']>=mean_score]['labels'].mean()
mean3=train_data_df[train_data_df['labels']<mean_score]['labels'].mean()

# In[15]:
train_data_df.loc[(train_data_df['labels']>=mean_score) & (train_data_df['labels']>=mean2),'class2']=3
train_data_df.loc[(train_data_df['labels']>=mean_score) & (train_data_df['labels']<mean2),'class2']=2

train_data_df.loc[(train_data_df['labels']<mean_score) & (train_data_df['labels']>=mean3),'class2']=1
train_data_df.loc[(train_data_df['labels']<mean_score) & (train_data_df['labels']<mean3),'class2']=0

# In[16]:
train_data_df[train_data_df['class']==1].shape

# In[17]:
data = np.load('./data/largeN_embeddings.npy')

# In[18]:
data.shape

# In[19]:
data

# In[20]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
#fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['labels']
print(label)
#for i in range(0,visit_emb.shape[0]):
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label,cmap='hot')
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[21]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
#fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['labels']
print(label)
#for i in range(0,visit_emb.shape[0]):
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label,cmap='hot')
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[22]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
#fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['labels']
print(label)
#for i in range(0,visit_emb.shape[0]):
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label,cmap='hot')
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[23]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['class2']
print(label)
#for i in range(0,visit_emb.shape[0]):
ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label,cmap='Dark2')
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[24]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
#fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['class2']
print(label)
#for i in range(0,visit_emb.shape[0]):
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,cmap='jet')
plt.legend(loc="lower right")
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[25]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
#fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['class2']
print(label)
#for i in range(0,visit_emb.shape[0]):
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,cmap='jet')
plt.legend(loc="lower right")
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[26]:
print("A")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
#fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['class2']
print(label)
#for i in range(0,visit_emb.shape[0]):
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,cmap='jet')
plt.legend(loc="lower right")
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()

# In[27]:
train_data_df.head()

# In[28]:
from sklearn import cluster
from sklearn import metrics
from nltk.cluster import KMeansClusterer
import nltk


kclusterer = KMeansClusterer(4, distance=nltk.cluster.util.cosine_distance, repeats=25)
assigned_clusters = kclusterer.cluster(data, assign_clusters=True)

print(len(assigned_clusters))
plt.scatter(tsne_proj[:,0],tsne_proj[:,1], c = assigned_clusters,cmap='jet')
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.legend()
plt.show()

# In[29]:
print("C")

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(data)
print(tsne_proj.shape)

# Plot those points as a scatter plot and label them based on the pred labels
#cmap = cm.get_cmap('tab20')
fig, ax = plt.subplots(figsize=(28,28))
#print("unique",doc_emb_data['readmission'].unique())
#     inv_featVocab = {v: k for k, v in featVocab.items()}
label = train_data_df['class2']
print(label)
#for i in range(0,visit_emb.shape[0]):
ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label,cmap='jet')
#ax.scatter(tsne_proj[:,0],tsne_proj[:,1], c = label,label = label)
    #ax.annotate(str(words[i]), (tsne_proj[i,0],tsne_proj[i,1]))
plt.show()
