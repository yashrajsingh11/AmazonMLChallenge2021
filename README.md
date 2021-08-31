# AmazonMLChallenge2021

### Prediction Model -

[Stochastic Gradient Descent Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html) with [Count Vectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) and [TF-IDF Transformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).

### Libraries used - 

1. Pandas (for data handling and pre-processing)
2. NLTK (for Text Processing)
3. Scikit-Learn (for Machine Learning Model and Text Processing) 

### Problem Statement -

The products which are displayed on their platform are divided into different browse nodes with their respective browse node ids. For e.g. T-shirts and badmintons are different browse nodes. We were provided with nearly 2.8 million training dataset and 100k testing dataset which was classified into nearly 10k browse node ids. Training data had columns - "TITLE","DESCRIPTION","BULLET_POINTS","BRANDS" which had all information about the product.

### Approach - 

Since the total data provided was very large and it was our first time working with such a large dataset, we decided to take a part of the data to train our model on it and see what kind or result we will get. This was our first time working on an NLP problem so we took some help from [the Internet](https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568) to get started. 

1. Data-Preprocessing - 

Filled all NA's with an empty string.
Combined the columns "TITLE", "DESCRIPTION", "BULLET_POINTS" into a new column "description" and dropped them (easier to process data).
Dropped the column "BRAND" since it was very ambiguous as one brand can have different type of products falling in different browse nodes.

2. Text-Preprocessing - 

Removed all stopwords, special characters, numbers and punctuations from the dataset.

### Score -

The first submission with training dataset size 35k and cross-validation dataset size 15k was given an accuracy score of 47% which was quite good since we neither used a very sohpisticated model like BERT etc. for natural language processing neither we used a large amount of data to train out model on. Some can call it beginner's luck though.

Later we increased the training dataset size to nearly 50k (going above this crashed the google collab) which gave an accuracy score of 49% which was also our final score  that surprisingly landed us at 185th position out of 3000+ teams. 

