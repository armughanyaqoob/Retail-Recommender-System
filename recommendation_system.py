#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# DATA LOADING, ANALYSIS AND VISUALIZATION
articles = pd.read_csv("D:/New folder/FYP/articles.csv")


articles.columns

f, ax = plt.subplots(figsize=(15, 7))
ax = sns.histplot(data=articles, y='index_name', color='orange')
ax.set_xlabel('count by index name')
ax.set_ylabel('index name')
plt.show()

fig = px.sunburst(articles, path=['index_group_name', 'index_name'],width=800,
    height=800,color_discrete_sequence=px.colors.cyclical.Edge)
fig.show()

fig = px.sunburst(articles, path=['product_group_name', 'product_type_name'],width=800,
    height=800,color_discrete_sequence=px.colors.cyclical.Edge)
fig.show()

f, ax = plt.subplots(figsize=(15, 7))
ax = sns.histplot(data=articles, y='garment_group_name', color='orange', hue='index_group_name', multiple="stack")
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
plt.show()

customers = pd.read_csv("D:/New folder/FYP/customers.csv")
customers.head(3)

customers = customers.replace("NONE", "None")

f, ax = plt.subplots(figsize=(15, 7))
plt.title('Distribution of members and non-members') 
ax = sns.histplot(data=customers, y='club_member_status', color='green')
ax.set_xlabel('count by member status')
ax.set_ylabel('membership status')
plt.show()


plt.figure(figsize=(10,5))
plt.title('Age distribution of customers') 
plt.xlim(0,100)
plt.xlabel('Number of customers')
plt.ylabel('Age')
sns.distplot(customers['age'],bins=10,kde=False)
plt.show()

f, ax = plt.subplots(figsize=(15, 7))
plt.title('Fashion news frequency') 
ax = sns.histplot(data=customers, y='fashion_news_frequency', color='green')
ax.set_xlabel('number of customers')
plt.show()


from skimage import io
ic = io.ImageCollection("D:/New folder/FYP/images_128_128/*/*.jpg")
ic = np.array(ic)
ic_flat = ic.reshape((len(ic), -1))
ic.shape 
number, m, n, weird = ic.shape  
# VISUALIZING THE PRODUCTS
import ipywidgets as widgets
from ipywidgets import interact

def view_image(n=0):
    plt.imshow(ic[n], cmap='gray', interpolation='nearest')
    plt.show()

w = interact(view_image, n=(0, len(ic)-1))

articles["image_exists"] = 0

for iRow in range(len(articles['article_id'])): 
    article_id = str(articles.iloc[iRow, 0])
    path = "D:/New folder/FYP/images_128_128/0" + str(article_id[:2]) + "/0" +article_id + ".jpg"
    if os.path.isfile(path) == True: 
        articles.iloc[iRow, 25] = 1
articles.head()

#RECOMMENDER SYSTEM USING PRODUCT DESCRIPTION
articles = articles.loc[articles['image_exists'] == 1] 
articles = articles.drop('image_exists', 1)

articles = articles.head(1000)
articles.shape 

articles['detail_desc'].head(3)

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



# Create the TfidfVectorizer object with stop words
tfidf = TfidfVectorizer(stop_words='english')

# Replace NaN values with an empty string in the 'detail_desc' column
articles['detail_desc'] = articles['detail_desc'].fillna('')

# Compute the TF-IDF matrix by fitting and transforming the 'detail_desc' column
tfidf_matrix = tfidf.fit_transform(articles['detail_desc'])

# Output the shape of the tfidf_matrix
print(tfidf_matrix.shape)


import sklearn

# Check the version of scikit-learn
sklearn_version = sklearn.__version__

if sklearn_version < "1.0":
    # Use get_feature_names()
    print(tfidf.get_feature_names()[100:106])
else:
    # Use get_feature_names_out()
    print(tfidf.get_feature_names_out()[100:106])
    
    
subset_tfidf_matrix = tfidf_matrix[:1487, :1487]  
cosine_sim = linear_kernel(subset_tfidf_matrix, subset_tfidf_matrix)


cosine_sim.shape

indices = pd.Series(articles.index, index=articles['article_id']).drop_duplicates()

def get_recommendations(article_id, cosine_sim=cosine_sim):
    # Get the index of the article that matches the ID
    idx = indices[article_id]

    # Get the pairwsie similarity scores of all articles with that article
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the articles based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar articles
    sim_scores = sim_scores[11:21]

    # Get the articles indices
    article_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar articles
    return articles['article_id'].iloc[article_indices]


get_recommendations(110065001)


import cv2
BASE = "D:/New folder/FYP/images_128_128/"
item = 116379047 # Test a random item
name1 = BASE+'0'+str(item)[:2]+'/0'+str(item)+'.jpg'
plt.figure(figsize=(20,5))
img1 = cv2.imread(name1)[:,:,::-1]
plt.title('So, the customer bought this item:',size=18)
plt.axis('off')
plt.imshow(img1)


name2 = BASE+'0'+str(recommendations.iloc[0])[:2]+'/0'+str(recommendations.iloc[0])+'.jpg'
name3 = BASE+'0'+str(recommendations.iloc[1])[:2]+'/0'+str(recommendations.iloc[1])+'.jpg'
name4 = BASE+'0'+str(recommendations.iloc[2])[:2]+'/0'+str(recommendations.iloc[2])+'.jpg'

plt.figure(figsize=(20,5))
img2 = cv2.imread(name2)[:,:,::-1]
img3 = cv2.imread(name3)[:,:,::-1]
img4 = cv2.imread(name4)[:,:,::-1]

plt.subplot(1,4,2)
plt.title('Recommendation 1',size=18)
plt.axis('off')
plt.imshow(img2)
plt.subplot(1,4,3)
plt.title('Recommendation 2',size=18)
plt.axis('off')
plt.imshow(img3)
plt.subplot(1,4,4)
plt.title('Recommendation 3',size=18)
plt.axis('off')
plt.imshow(img4)
plt.show() 


# Using deep learning to recommend items
transactions = pd.read_csv("D:/New folder/FYP/Retailers_Dataset.csv")
transactions = transactions.head(2000)
transactions.head()


transactions['bought'] = 1 #the interaction matrix will be binary
df=transactions[['customer_id', 'article_id', 'bought']]
df.head()


df = df.drop_duplicates()

# Creating a sparse pivot table with customers in rows and items in columns
customer_items_matrix_df = df.pivot(index   = 'customer_id', 
                                    columns = 'article_id', 
                                    values  = 'bought').fillna(0)
customer_items_matrix_df.head(10)


customer_items_matrix_df.shape

customer_items_matrix_df_train, customer_items_matrix_df_test = train_test_split(customer_items_matrix_df,test_size=0.33, random_state=42)
print(customer_items_matrix_df_train.shape, customer_items_matrix_df_test.shape)


customer_items_matrix_df_train.values.mean()*100



def autoEncoder(X):
   

    # Input
    input_layer = Input(shape=(X.shape[1],), name='UserScore')
    
    # Encoder
    # -----------------------------
    enc = Dense(512, activation='selu', name='EncLayer1')(input_layer)

    # Latent Space
    # -----------------------------
    lat_space = Dense(256, activation='selu', name='LatentSpace')(enc)
    lat_space = Dropout(0.8, name='Dropout')(lat_space) # Dropout

    # Decoder
    # -----------------------------
    dec = Dense(512, activation='selu', name='DecLayer1')(lat_space)
    
    # Output
    output_layer = Dense(X.shape[1], activation='linear', name='UserScorePred')(dec)

    # this model maps an input to its reconstruction
    model = Model(input_layer, output_layer)    
    
    return model


X = customer_items_matrix_df_train.values
X.shape[1]
model = autoEncoder(X)

model.compile(optimizer = Adam(lr=0.0001), loss='mse')
    
model.summary()


hist = model.fit(x=X, y=X,
                  epochs=50,# Using 50 here instead of 10 or 20 improved the performance very much!
                  batch_size=64,
                  shuffle=True,
                  validation_split=0.1)





def plot_hist(hist):
    # summarize history for loss
    fig, ax = plt.subplots()  # create figure & 1 axis

    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])

plot_hist(hist)


new_matrix = model.predict(X) * (X[0] == 0)
# converting the reconstructed matrix back to a Pandas dataframe
new_customer_items_matrix_df  = pd.DataFrame(new_matrix, 
                                            columns = customer_items_matrix_df_train.columns, 
                                            index   = customer_items_matrix_df_train.index)
new_customer_items_matrix_df.head()


print(new_customer_items_matrix_df.values.min(), new_customer_items_matrix_df.values.max())


def recommender_for_customer(customer_id, interact_matrix, df_content, topn = 10):
    '''
    Recommender Articles for Customers
    '''
    pred_scores = interact_matrix.loc[customer_id].values

    df_scores   = pd.DataFrame({'article_id': list(customer_items_matrix_df.columns), 
                               'score': pred_scores})

    df_rec      = df_scores.set_index('article_id')\
                    .join(df_content.set_index('article_id'))\
                    .sort_values('score', ascending=False)\
                    .head(topn)[['score', 'prod_name']]
    
    return df_rec[df_rec.score > 0]



articles = pd.read_csv("D:/New folder/FYP/articles.csv") 

# EXAMPLE: ARTICLE previously purchased by the customer
recommender_for_customer(customer_id     = '029ceb992cb63df03c109790046e3fdebfce0b63c968823dd461b7f18ecc6b30', 
                         interact_matrix = customer_items_matrix_df, 
                         df_content      = articles)



recommender_for_customer(customer_id     = '0008968c0d451dbc5a9968da03196fe20051965edde7413775c4eb3be9abe9c2', 
                         interact_matrix = new_customer_items_matrix_df, 
                         df_content      = articles)


X_test = customer_items_matrix_df_test.values
X_test.shape


new_matrix_test = model.predict(X_test) * (X_test[0] == 0)
new_customer_items_matrix_df_test  = pd.DataFrame(new_matrix_test, 
                                            columns = customer_items_matrix_df_test.columns, 
                                            index   = customer_items_matrix_df_test.index)
new_customer_items_matrix_df_test.head()



recommendations = recommender_for_customer(customer_id     = '01f597f5eba83f9709eceb5a70a99f3a4009a6e827bf7293afa4211030a29fa4', 
                                           interact_matrix = customer_items_matrix_df_test, 
                                           df_content      = articles)
print(recommendations)



recommendation_indices = recommendations.index.tolist()
print(recommendation_indices)



counter = 1

for iRec in recommendation_indices:
    name = BASE+'0'+str(iRec)[:2]+'/0'+str(iRec)+'.jpg'
    plt.figure(figsize=(20,5))
    img = cv2.imread(name)[:,:,::-1]
    plt.subplot(1,5,counter)
    plt.title('Recommendation ' + str(counter),size=18)
    plt.axis('off')
    plt.imshow(img)
    counter = counter + 1
    



# In[ ]:




