import csv
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import zipfile
nltk.download('stopwords')
import pickle

!wget https://s3-ap-southeast-1.amazonaws.com/he-public-data/dataset52a7b21.zip
path_to_zip_file = "/content/dataset52a7b21.zip"
directory_to_extract_to = "/content/dataset"

with zipfile.ZipFile(path_to_zip_file, 'r') as zip_ref:
    zip_ref.extractall(directory_to_extract_to)

train_data = "/content/dataset/dataset/train.csv"
test_data = "/content/dataset/dataset/test.csv"

df = pd.read_csv(train_data, escapechar = "\\", quoting = csv.QUOTE_NONE)

test_df = pd.read_csv(test_data, escapechar = "\\", quoting = csv.QUOTE_NONE)

df['TITLE'] = df['TITLE'].fillna(" ")
df['DESCRIPTION'] = df['DESCRIPTION'].fillna(" ")
df["BULLET_POINTS"] = df["BULLET_POINTS"].fillna(" ")

df["description"] = df["TITLE"] + df["DESCRIPTION"] + df["BULLET_POINTS"]
df.drop('TITLE', inplace = True, axis = 1)
df.drop('DESCRIPTION', inplace = True, axis = 1)
df.drop('BULLET_POINTS', inplace = True, axis = 1)
df.drop('BRAND', inplace = True, axis = 1)
nodes = df["BROWSE_NODE_ID"]
df.drop('BROWSE_NODE_ID', inplace = True, axis = 1)

def process_tweet(words_):
  stopwords_english = stopwords.words('english')
  words_ = re.sub(r'\$\w*', '', words_)
  words_ = re.sub(r'^RT[\s]+', '', words_)
  words_ = re.sub(r'https?:\/\/.[\r\n]', '', words_)  
  words_ = re.sub(r'<.*>', '', words_)
  words_ = re.sub(r'\d+', '', words_) 
  words_ = re.sub(r'#', '', words_)
  tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
  words_tokens = tokenizer.tokenize(words_)

  words_clean = []
  for word in words_tokens:
    if (word not in stopwords_english and   
        word not in string.punctuation): 
      words_clean.append(word)
  
  words_clean = " ".join(words_clean)
  return words_clean

myprocessed = pd.DataFrame()

processed = []
for i in range(0,50000):
  raw = df['description'][i]
  processed.append(process_tweet(raw))
myprocessed['description'] = processed

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

sgd = Pipeline([('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
               ])
 
sgd.fit(myprocessed['description'], nodes[:50000])
filename = 'SGDFiftyK.sav'
pickle.dump(sgd, open(filename, 'wb'))

test_df['TITLE'] = test_df['TITLE'].fillna(" ")
test_df['DESCRIPTION'] = test_df['DESCRIPTION'].fillna(" ")
test_df["BULLET_POINTS"] = test_df["BULLET_POINTS"].fillna(" ")

predictions_df = pd.DataFrame()
myprocessed1 = pd.DataFrame()

test_df["description"] = test_df["TITLE"] + test_df["DESCRIPTION"] + test_df["BULLET_POINTS"]
test_df.drop('TITLE', inplace = True, axis = 1)
test_df.drop('DESCRIPTION', inplace = True, axis = 1)
test_df.drop('BULLET_POINTS', inplace = True, axis = 1)
test_df.drop('BRAND', inplace = True, axis = 1)
predictions_df['PRODUCT_ID'] = test_df['PRODUCT_ID']
test_df.drop('PRODUCT_ID', inplace = True, axis = 1)

processed = []
for i in range(0,len(test_df)):
  raw = test_df['description'][i]
  processed.append(process_tweet(raw))
myprocessed1['description'] = processed

filename = 'SGDFiftyK.sav'
loaded_model = pickle.load(open(filename, 'rb'))

predictions = loaded_model.predict(myprocessed1['description'])

predictions_df['BROWSE_NODE_ID'] = predictions

predictions_df.to_csv('PredictionsFiftyk.csv', index = False)