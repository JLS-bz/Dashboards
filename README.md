# Dashboards
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


# Topic Modeling with BERT and TF-IDF
================
2022-07-20




# Topic Modeling with SparkNLP tf-idf LDA
================
2022-07-26

