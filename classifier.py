import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder,StandardScaler
from wordcloud import WordCloud

# print(os.listdir('../input/storypointsfull'))
filename = "appceleratorstudio.csv"
data = pd.read_csv(filename)
print(data.shape)
data = data.drop(['issuekey'], axis = 1)
# print(data.describe())
# data = data[(data.storypoint == 1) | (data.storypoint == 2) | (data.storypoint == 13)]

data['titDescription'] = data[['title', 'description']].apply(lambda x: ' '.join(x.map(str)), axis=1)
data['lenTitDescription'] = data['titDescription'].str.len()

data = data.drop(['title'], axis = 1)
data = data.drop(['description'], axis = 1)
# print(data.columns)
# print(data.head(10))
plt.rcParams['figure.figsize'] = (10, 7)
sns.boxenplot(x = data['storypoint'], y = data['lenTitDescription'])
plt.title('Relation between Story Points and Title Length', fontsize = 20)
# plt.show()
wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['titDescription']))

plt.rcParams['figure.figsize'] = (10, 10)
plt.title('Most Common words in the dataset', fontsize = 20)
plt.axis('off')
plt.imshow(wordcloud)
# plt.show()

cv = CountVectorizer()
words = cv.fit_transform(data['titDescription'].values.astype('U'))

sum_words = words.sum(axis=0)

words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)

frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])

frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'orange')
plt.title("Most Frequently Occuring Words - Top 30")
# plt.show()
corpusTitDescription = []

for i in range(0, 2919):

  review = re.sub('[^a-zA-Z]', ' ', data['titDescription'].values.astype('U')[i])

  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  # joining them back with space
  review = ' '.join(review)

  corpusTitDescription.append(review)

# print(corpusTitDescription)
print(data.columns)
# print(data.head(10))

# creating bag of words

cv = CountVectorizer()

x = cv.fit_transform(corpusTitDescription).toarray()
y = data.iloc[:, 0]
print(x)
print(y)
print(x.shape)
print(y.shape)

#Title
# splitting the training data into train and valid sets


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.20,)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# standardization
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# models = []
# models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr',max_iter=4000,dual=False)))
# # models.append(('LDA', LinearDiscriminantAnalysis()))
# models.append(('KNN', KNeighborsClassifier(n_neighbors=8,max_iter=4000)))
# models.append(('CART', DecisionTreeClassifier(max_depth=5,)))
# models.append(('NB', GaussianNB()))
# models.append(('SVM', SVC(kernel="linear", C=0.025,max_iter=4000,)))
# models.append(('SVM', SVC()))
# models.append(('CNN', MLPClassifier(max_iter=4000)))
# models.append(('AdaBoost', AdaBoostClassifier()))
# # models.append(('Bagging', BaggingClassifier(base_estimator=SVC(),n_estimators=10, random_state=0)))
# # models.append(('QDA', QuadraticDiscriminantAnalysis()))
# models.append(('RF', RandomForestClassifier()))
# # evaluate each model in turn
# results = []
# names = []
# scoring = 'accuracy'
# for name, model in models:
#     kfold = model_selection.KFold(n_splits=10)
#     cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring)
#     results.append(cv_results)
#     names.append(name)
#     msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#     print(msg)
#
# # boxplot algorithm comparison
# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.savefig('Algorithm Comparison')
# plt.show()

model = RandomForestClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

print("Training Accuracy :", model.score(x_train, y_train))
print("Testing Accuracy :", model.score(x_test, y_test))

# classification report
cr = classification_report(y_test, y_pred)
print(cr)
