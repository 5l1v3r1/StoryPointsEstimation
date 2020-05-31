import os,glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,roc_curve, auc, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

filename = "appcelerator.csv"
data = pd.read_csv(filename)
print(data.isnull().sum())
# print(data.describe())
data = data.dropna(how='any')
data = data.drop(['issuekey'], axis = 1)

print(data.shape)
print(data.describe())
print(data.groupby('storypoint').size())

# data = data[(data.storypoint == 5) | (data.storypoint == 3)|(data.storypoint == 8)]

data.loc[data.storypoint <= 2, 'storypoint'] = 0 #small
data.loc[(data.storypoint > 2) & (data.storypoint <= 5), 'storypoint'] = 1 #medium
data.loc[data.storypoint > 5, 'storypoint'] = 2 #big

print(data.groupby('storypoint').size())

data['titDescription'] = data[['title', 'description']].apply(lambda x: ' '.join(x.map(str)), axis=1)
data['lenTitDescription'] = data['titDescription'].str.len()

data = data.drop(['title'], axis = 1)
data = data.drop(['description'], axis = 1)
print(data.shape)


plt.rcParams['figure.figsize'] = (18, 10)
sns.boxenplot(x = data['storypoint'], y = data['lenTitDescription'])
plt.title('Relation between Story Points and Title Length', fontsize = 20)
plt.savefig('classes representation with sampling after new segmentation')
# plt.show()
# plt.hist(data.storypoint, bins=20, alpha=0.6, color='y')
# plt.title("#Items per Point")
# plt.xlabel("Points")
# plt.ylabel("Count")
# plt.savefig('items per point')
# wordcloud = WordCloud(background_color = 'gray', width = 1000, height = 1000, max_words = 50).generate(str(data['titDescription']))
# plt.rcParams['figure.figsize'] = (10, 10)
# plt.title('Most Common words in the dataset', fontsize = 20)
# plt.axis('off')
# plt.imshow(wordcloud)
# plt.savefig('common words')
# plt.show()

cv = CountVectorizer()
tfidf = TfidfVectorizer()
words = cv.fit_transform(data['titDescription'].values.astype('U'))
sum_words = words.sum(axis=0)
#
# words_freq = [(word, sum_words[0, i]) for word, i in cv.vocabulary_.items()]
# words_freq = sorted(words_freq, key = lambda x: x[1], reverse = True)
#
# frequency = pd.DataFrame(words_freq, columns=['word', 'freq'])
#
# frequency.head(30).plot(x='word', y='freq', kind='bar', figsize=(15, 7), color = 'orange')
# plt.title("Most Frequently Occuring Words - Top 30")
# plt.savefig('top common words')
# plt.show()


corpusTitDescription = []
n=data.shape[0]
for i in range(0, n):

  review = re.sub('[^a-zA-Z]', ' ', data['titDescription'].values.astype('U')[i])
  review = review.lower()
  review = review.split()
  ps = PorterStemmer()
  # stemming
  review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]

  # joining them back with space
  review = ' '.join(review)

  corpusTitDescription.append(review)


# creating bag of words
x = cv.fit_transform(corpusTitDescription).toarray()
y = data.iloc[:, 0]

print('x:',x)
print('y:',y)
print('x_shape',x.shape)
print('y_shape',y.shape)



#Title
# splitting the training data into train and valid sets
seed=50


x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = 0.20 ,random_state=seed,shuffle=True)

print('x_train_shape',x_train.shape)
print('x_test_shape',x_test.shape)
print('y_train_shape',y_train.shape)
print('x_test_shape',y_test.shape)

# standardization
sc = StandardScaler()

# x_train = sc.fit_transform(x_train)
# x_test = sc.transform(x_test)
print('Y_train_value_counts :','\n',y_train.value_counts())


# smt = SMOTE()
# x_train, y_train = smt.fit_sample(x_train, y_train)
# print('Y_train_classes_counts',np.bincount(y_train))
# param_grid = {'penalty':['l1','l2'],'C':[0.01,0.025,0.1,1,5,10,100],'loss':['squared_hinge','hinge']}
# model = model_selection.GridSearchCV(LinearSVC(), param_grid, cv=10,scoring='accuracy')
# model.fit(x_train, y_train)
# print('best parameters',model.best_params_)

models = []
models.append(('SVC', SVC(C=0.025,kernel='linear', class_weight='balanced', probability=True)))
# models.append(('SVCLw', LinearSVC(C=0.025,class_weight='balanced')))
models.append(('SVCLp', LinearSVC(C=0.025,penalty='l1', loss='squared_hinge', dual=False,random_state=seed,)))
models.append(('SVCLw', LinearSVC(C=0.025,penalty='l1', loss='squared_hinge', dual=False,random_state=seed,class_weight='balanced')))
models.append(('SVCd', LinearSVC()))
# models.append(('SVCLp2', LinearSVC(C=100,penalty='l2', loss='hinge',random_state=seed,class_weight='balanced')))
# models.append(('SVCL', LinearSVC(C=0.025,random_state=seed,)))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr',class_weight='balanced')))
models.append(('KNN', KNeighborsClassifier(n_neighbors=5)))
# models.append(('CART', DecisionTreeClassifier(max_depth=5)))
# models.append(('NB', GaussianNB()))
models.append(('xGB', XGBClassifier(n_estimators=100, random_state=seed,scale_pos_weight=99)))
# models.append(('CNN', MLPClassifier(alpha=1, )))
# models.append(('AdaBoost', AdaBoostClassifier(n_estimators=100)))
# models.append(('RF', RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)))


# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'

for name, model in models:
    kfold = model_selection.KFold(n_splits=10,random_state=seed,shuffle=True)
    cv_results = model_selection.cross_val_score(model, x, y, cv=kfold, scoring=scoring,)
    results.append(cv_results)
    names.append(name)
    accuracy = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm=confusion_matrix(y_test,y_pred)
    cr = classification_report(y_test, y_pred)

    print(accuracy)
    print('Error Rate',1-cv_results.mean())
    print(cm)
    print(cr)


# fig = plt.figure()
# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# plt.boxplot(results)
# ax.set_xticklabels(names)
# plt.savefig('Algorithm Comparison with sampling before new segmentation')# boxplot algorithm comparison
# plt.show()

# model = LinearSVC(C=0.025,penalty='l1', loss='squared_hinge', dual=False,random_state=seed)
# # model = AdaBoostClassifier()
# # model = XGBClassifier()
# # model = LinearSVC(C=5,penalty='l1', loss='squared_hinge', dual=False,=4000)
# # # model = model_selection.GridSearchCV(svm, parameters, cv=10,scoring='accuracy')
# model.fit(x_train, y_train)
# # # print('best parameters',model.best_params_)
#
# y_pred = model.predict(x_test)
# print("Training Accuracy :", model.score(x_train, y_train))
# print("Testing Accuracy :", model.score(x_test, y_test))
# cr = classification_report(y_test, y_pred,)
# cm=confusion_matrix(y_test,y_pred)
# print(cm)
# print(cr)

