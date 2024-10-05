import pandas as pd 
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re

df = pd.read_csv('C:/Users/ASHOK BALYAN/AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Anaconda3 (64-bit)/LLM/NLP/SpamClassifier/SMSSpamCollection.txt',sep='\t',names=['label','message'])
               

ps =PorterStemmer()

corpus=[]
for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]',' ',df['message'][i])
    sentance = review.lower()
    review = review.split()
    review =[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import CountVectorizer
tf = CountVectorizer(max_features=2500)
X = tf.fit_transform(corpus).toarray()

y= pd.get_dummies(df['label'])
y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix,accuracy_score
cf =confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)



    