# Section 1
Exploratory Analysis of Keywords in Forums on Dissociation
================
2022-03-25

##### Description of raw data:

-   Webscraped using PRAW, Reddit’s API
-   From forum about depersonalization/derealization.
-   Contains post title, post content, post date
-   Data date range: 2022-03-01 to 2022-03-27

#### Most commonly used words:

![](unnamed-chunk-2-1.png)<!-- -->![](unnamed-chunk-2-2.png)<!-- -->

#### Most common positive and negative words:

![](unnamed-chunk-3-1.png)<!-- -->

#### Relationships between words: n-grams and correlations

##### Visualizing a network of bigrams:

![](unnamed-chunk-4-1.png)<!-- -->

##### Centrality of Words

    ## Warning in graph_from_data_frame(x, directed = directed): In `d' `NA' elements
    ## were replaced with string "NA"

    ## Warning: ggrepel: 140 unlabeled data points (too many overlaps). Consider
    ## increasing max.overlaps

![](unnamed-chunk-5-1.png)<!-- -->


# Section 2
Topic Modeling with BERT and TF-IDF
================
2022-07-20

```python
#!pip install bertopic
```


```python
import pandas as pd

# Reading file from local storage
file_location = r'/home/jls/JN/dpdr.csv'
file_type = "csv"

dfs = pd.read_csv(file_location)
doc = dfs.loc[:,"post"]
print(doc)
#df.count()
```

    0       LolThere is no way out of this , I have offici...
    1       I’m afraid I’ve been dealing with a 5 day long...
    2       If you have tension in your head, does it feel...
    3       Parnate and lamotrigineHey everyone,\n\nI just...
    4       Short discussion of the film "Numb" which is t...
                                  ...                        
    3631                          r/dpdr Subdirect Statistics
    3632    COVID and DPDR(not here to argue about vaccina...
    3633    Relief after 5 months. Ask me anything:)Finall...
    3634    It's so intense that I feel like I don't even ...
    3635    Anyone tried l tyrosine?I’ve heard that’s a go...
    Name: post, Length: 3636, dtype: object



```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')
embeddings = model.encode(doc, show_progress_bar=True)
```


    Batches:   0%|          | 0/114 [00:00<?, ?it/s]



```python
import umap
umap_embeddings = umap.UMAP(n_neighbors=15, 
                            n_components=5, 
                            metric='cosine').fit_transform(embeddings)
```


```python
import hdbscan
cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                          metric='euclidean',                      
                          cluster_selection_method='eom').fit(umap_embeddings)
```


```python
import matplotlib.pyplot as plt

# Prepare data
umap_data = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
result = pd.DataFrame(umap_data, columns=['x', 'y'])
result['labels'] = cluster.labels_

# Visualize clusters
fig, ax = plt.subplots(figsize=(20, 10))
outliers = result.loc[result.labels == -1, :]
clustered = result.loc[result.labels != -1, :]
plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=1)
plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=1, cmap='hsv_r')
plt.colorbar()
```




    <matplotlib.colorbar.Colorbar at 0x7f92ee57c7c0>




    
![png](TopicModeling_BERT_TF-IDF_files/TopicModeling_BERT_TF-IDF_6_1.png)
    



```python
docs_df = doc.to_frame().rename(columns = {'post':'Doc'})
docs_df['Topic'] = cluster.labels_
docs_df['Doc_ID'] = range(len(docs_df))
docs_df
docs_per_topic = docs_df.groupby(['Topic'], as_index = False).agg({'Doc': ' '.join})
```


```python
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count
  
tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(doc))
```


```python
def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes

top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
topic_sizes = extract_topic_sizes(docs_df); topic_sizes.head(10)
```

    /home/jls/anaconda3/lib/python3.9/site-packages/sklearn/utils/deprecation.py:87: FutureWarning: Function get_feature_names is deprecated; get_feature_names is deprecated in 1.0 and will be removed in 1.2. Please use get_feature_names_out instead.
      warnings.warn(msg, category=FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Topic</th>
      <th>Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>2663</td>
    </tr>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>601</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>127</td>
    </tr>
    <tr>
      <th>6</th>
      <td>5</td>
      <td>86</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>31</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>22</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>19</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>17</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
</div>




```python
top_n_words[0][:10]
```




    [('really', 0.007758313612847956),
     ('feeling', 0.007749180482272903),
     ('know', 0.007713640753396208),
     ('ve', 0.007476064080397414),
     ('think', 0.007466554247109187),
     ('time', 0.007440581170049944),
     ('life', 0.0074352045216576805),
     ('felt', 0.007325839413153062),
     ('feels', 0.007263618785323988),
     ('don', 0.007243585572851497)]




```python
top_n_words[1][:10]
```




    [('poll', 0.27884752064544444),
     ('www', 0.17575169562801965),
     ('view', 0.16821173441824555),
     ('com', 0.16627149629544336),
     ('reddit', 0.16545962703344716),
     ('https', 0.16530972633780874),
     ('donate', 0.01729449223848627),
     ('nlm', 0.016815957968685864),
     ('ncbi', 0.016815957968685864),
     ('nih', 0.016411363008509334)]



# Section 3
Topic Modeling with SparkNLP tf-idf LDA
================
2022-07-26


