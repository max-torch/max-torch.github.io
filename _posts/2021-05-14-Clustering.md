---
layout: post
title: "Dating Pools using K-Means Clustering"
subtitle: ""
background: '/img/posts/Clustering/Cover.jpg'
---

Previously, [we explored a large dataset](https://max-torch.github.io/2021/05/15/OKCupid.html) containing 60,000 anonymized OKCupid users from the year 2012.

You also might want to check out [Gender Classification with OKCupid Data](https://max-torch.github.io/2021/05/14/ml_revisited.html), which also uses our OKCupid data.

 In this article, we apply K-Means clustering, a machine learning algorithm, to group our OKCupid users into dating pools, as a means to narrow down their potential matches.

Clustering, is a technique which groups similar data points together. Let's use this to group similar people together and recommend who you should date. People in the same cluster as you are the people who we will recommend.

## Work Skills showcased in this article:
* Application of K-Means Clustering using scikit-learn

## Feature Selection

We start by reviewing the features of our users. Which, among the features, would you want your date to have in common with you?


```python
expanded_df_backup.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 59811 entries, 0 to 59945
    Data columns (total 48 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   age                 59811 non-null  int64  
     1   body_type           54538 non-null  object 
     2   diet                35481 non-null  object 
     3   drinks              56846 non-null  object 
     4   drugs               45755 non-null  object 
     5   education           53211 non-null  object 
     6   essay0              54351 non-null  object 
     7   essay1              52275 non-null  object 
     8   essay2              50210 non-null  object 
     9   essay3              48375 non-null  object 
     10  essay4              49318 non-null  object 
     11  essay5              49010 non-null  object 
     12  essay6              46085 non-null  object 
     13  essay7              47404 non-null  object 
     14  essay8              40636 non-null  object 
     15  essay9              47245 non-null  object 
     16  ethnicity           54159 non-null  object 
     17  height              59811 non-null  float64
     18  income              11456 non-null  float64
     19  job                 51641 non-null  object 
     20  last_online         59811 non-null  object 
     21  location            59811 non-null  object 
     22  offspring           24334 non-null  object 
     23  orientation         59811 non-null  object 
     24  pets                39931 non-null  object 
     25  religion            39631 non-null  object 
     26  sex                 59811 non-null  object 
     27  sign                48787 non-null  object 
     28  smokes              54320 non-null  object 
     29  speaks              59761 non-null  object 
     30  status              59811 non-null  object 
     31  diet_adherence      35481 non-null  object 
     32  diet_type           35481 non-null  object 
     33  city                59811 non-null  object 
     34  state/country       59811 non-null  object 
     35  offspring_want      24334 non-null  object 
     36  offspring_attitude  9711 non-null   object 
     37  religion_type       39631 non-null  object 
     38  religion_attitude   39631 non-null  object 
     39  sign_type           48787 non-null  object 
     40  sign_attitude       48787 non-null  object 
     41  dog_preference      28880 non-null  object 
     42  cat_preference      21293 non-null  object 
     43  has_dogs            39931 non-null  float64
     44  has_cats            39931 non-null  float64
     45  num_ethnicities     54159 non-null  float64
     46  optional_%unfilled  59811 non-null  float64
     47  num_languages       59811 non-null  int64  
    dtypes: float64(6), int64(2), object(40)
    memory usage: 22.4+ MB
    

After asking someone which features they would want to have in common with them, the chosen features are:
* Drugs
* Diet
* Pets
* Orientation
* Religion (attitude only)

Let's isolate that subset of features.


```python
clustering_df = sparse_essay_df.copy()
cat_selection = ['drugs', 'orientation',
                'diet_adherence', 'diet_type','religion_attitude', 'dog_preference', 'cat_preference', 'has_dogs',
                'has_cats']
numeric_selection = []
feature_selection = feature_selection_to_list(clustering_df, cat_selection, numeric_selection)
clustering_df = clustering_df[feature_selection]

#Conversion to Scipy csr_matrix
clustering_coo = clustering_df.sparse.to_coo()
clustering_csr = clustering_coo.tocsr()
clustering_csr.get_shape()
```




    (59811, 37)



## Choosing a value of k

Now that we have our subset let's search for the best number of k clusters to use in our model.


```python
#Recommended setting for training Kmeans on Windows
import os
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.cluster import KMeans

execute = False
if execute:
    num_clusters = list(range(1,200))
    inertias= []
    for i in num_clusters:
      model = KMeans(n_clusters = i)
      model.fit(clustering_csr)
      inertias.append(model.inertia_)

    plt.plot(num_clusters, inertias, '-o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.show()
```

The Inertia vs k graph below took 5 hours to produce (hence the default setting above to skip execution of the code snippet). Inertia, is a metric which represents how spaced out the points of a cluster are relative to its centroid. A line has been drawn over the tail end of the graph so that we can clearly mark where the graph becomes linear. The specific point where the graph becomes linear is known as the elbow point, and is the number of clusters that we should use.

<img alt="K_Clusters_evaluation_annotated" src="\img\posts\Clustering\K_Clusters_evaluation_annotated.png">

k = 100 is where the linearity begins. Let's create the model we will use for clustering with k = 100.


```python
model = KMeans(n_clusters = 100)
model.fit(clustering_csr)
labels = model.predict(clustering_csr)
```

Let's investigate the uniformity of our cluster distribution.


```python
ser_labels  = pd.Series(labels)
ser_labels_props = ser_labels.value_counts()
plt.figure(figsize = (16,9))
plt.pie(x=ser_labels_props.values, labels = ser_labels_props.index,
                wedgeprops=dict(width=0.10,
                                edgecolor="k",
                                linewidth=0.7))
plt.text(0, 0, 'Cluster Distribution', 
                 horizontalalignment = 'center',
                 verticalalignment = 'center',
                 fontsize = 20)
plt.show()
```


    
![png](\img\posts\Clustering\Cluster_Distribution.png)
    


The clusters are somewhat distributed across users. 

## Date Match Recommendations

The code below produces an interface which allows you to enter information and, upon pressing Run Interact, output the number of the cluster you belong to, the number of people in your cluster, and a random user profile from the same cluster. People who are in the same cluster are similar to each other.

*Note that as an experiment, null values have been included as their own category. The model also groups people who leave similar fields blank. The values of 'nan' or 'None' or 'No Prefix' means that you are not sharing information for that particular field.*

The actual interface is not available here because it requires an active python kernel to run. However, you can still see the interface in action via a GIF Image Preview. If you would like to use the interface yourself, open and run 'Report_stable.ipynb' from this project's Github Repository.


```python
from ipywidgets import interact_manual
import random

def make_profile(sex, drugs, orientation, diet_adherence, diet_type, religion_attitude,
                 dog_preference, cat_preference, has_dogs, has_cats, show_same_gender):
    
    user_info = [drugs, orientation, diet_adherence, diet_type, religion_attitude,
                 dog_preference, cat_preference, has_dogs, has_cats]
    user_dict = {}
    for each in clustering_df.columns.to_list():
         user_dict[each] = [0]
    for each in user_info:
        user_dict[each] = [1]
    user_df = pd.DataFrame(data=user_dict, index = ['You'])
    sparse_user_df = user_df.copy()
    for each in sparse_user_df.columns.to_list():
        sparse_user_df[each] = pd.arrays.SparseArray(sparse_user_df[each].values, dtype='uint8')
        
    sparse_user_df = sparse_user_df.sparse.to_coo()
    sparse_user_df = sparse_user_df.tocsr()
    
    user_cluster_label = model.predict(sparse_user_df)
    user_cluster = np.where(labels == user_cluster_label)[0]
    print("Your cluster is {}.\nThere are {} users in your cluster.".format(user_cluster_label, len(user_cluster)))
    
    if show_same_gender == False:
        user_sex_dict = {'Male':'m', "Female":'f'}
        user_sex = user_sex_dict[sex]
        #Look for an opposite gender
        while user_sex == user_sex_dict[sex]:
            rand_index = random.randint(0, len(user_cluster))
            rand_user = user_cluster[rand_index]
            user_sex = expanded_df_backup.iloc[rand_user]['sex']
    else:
        rand_index = random.randint(0, len(user_cluster))
        rand_user = user_cluster[rand_index]       

    print("Is this person dateable? Press Run again to see a new recommendation.\n")
    show_user_data(expanded_df_backup.iloc[:,0:31], rand_user)

interact_manual(
    make_profile,
    sex = ['Male', 'Female'],
    drugs = feature_selection_to_list(clustering_df, ['drugs'], []),
    orientation = feature_selection_to_list(clustering_df, ['orientation'], []),
    diet_adherence = feature_selection_to_list(clustering_df, ['diet_adherence'], []),
    diet_type = feature_selection_to_list(clustering_df, ['diet_type'], []),
    religion_attitude = feature_selection_to_list(clustering_df, ['religion_attitude'], []),
    dog_preference = feature_selection_to_list(clustering_df, ['dog_preference'], []),
    cat_preference = feature_selection_to_list(clustering_df, ['cat_preference'], []),
    has_dogs = feature_selection_to_list(clustering_df, ['has_dogs'], []),
    has_cats = feature_selection_to_list(clustering_df, ['has_cats'], []),
    show_same_gender = True
)
```
### GIF Image Preview

![Animation.gif](\img\posts\Clustering\Animation.gif)

## Dating Pools using K-Means Clustering Recap

In this article, we accomplished the following:
* Selected features for our model, on the basis of wanting a potential date to be similar for those particular features
* Used the 'elbow method' for selecting a value of k for our k-means clustering model
* Created an interactive interface for presenting profiles of suggested users to date based on input user data
