import os,glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk import ConfusionMatrix
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,roc_curve, auc, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,BaggingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
warnings.filterwarnings('always')
import uuid
import subprocess

filename = "appceleratorstudio.csv"
df = pd.read_csv(filename,usecols=['issuekey', 'title', 'description', 'storypoint'])
print(df.isnull().sum())

df = df.dropna(how='any')
print(df.head(10))
print(df.storypoint.describe())

plt.hist(df.storypoint, bins=20, alpha=0.6, color='y')
plt.title("#Items per Point")
plt.xlabel("Points")
plt.ylabel("Count")

# plt.show()
print(df.groupby('storypoint').size())


df.loc[df.storypoint <= 2, 'storypoint'] = 0 #small
df.loc[(df.storypoint > 2) & (df.storypoint <= 5), 'storypoint'] = 1 #medium
df.loc[df.storypoint > 5, 'storypoint'] = 2 #big

print(df.groupby('storypoint').size())

# Define some known html tokens that appear in the data to be removed later
htmltokens = ['{html}', '<div>', '<pre>', '<p>', '</div>', '</pre>', '</p>']


# Clean operation
# Remove english stop words and html tokens
def cleanData(text):
    result = ''

    for w in htmltokens:
        text = text.replace(w, '')

    text_words = text.split()

    resultwords = [word for word in text_words if word not in stopwords.words('english')]

    if len(resultwords) > 0:
        result = ' '.join(resultwords)
    else:
        print('Empty transformation for: ' + text)

    return result


def formatFastTextClassifier(label):
    return "__label__" + str(label) + " "

df['title_desc'] = df['title'].str.lower() + ' - ' + df['description'].str.lower()
df['label_title_desc'] = df['storypoint'].apply(lambda x: formatFastTextClassifier(x)) + df['title_desc'].apply(lambda x: cleanData(str(x)))

df = df.reset_index(drop=True)




def SimpleOverSample(_xtrain, _ytrain):
    xtrain = list(_xtrain)
    ytrain = list(_ytrain)

    samples_counter = Counter(ytrain)
    max_samples = sorted(samples_counter.values(), reverse=True)[0]
    for sc in samples_counter:
        init_samples = samples_counter[sc]
        samples_to_add = max_samples - init_samples
        if samples_to_add > 0:
            # collect indices to oversample for the current class
            index = list()
            for i in range(len(ytrain)):
                if (ytrain[i] == sc):
                    index.append(i)
            # select samples to copy for the current class
            copy_from = [xtrain[i] for i in index]
            index_copy = 0
            for i in range(samples_to_add):
                xtrain.append(copy_from[index_copy % len(copy_from)])
                ytrain.append(sc)
                index_copy += 1
    return xtrain, ytrain





# class FastTextClassifier:
#     rand = ""
#     inputFileName = ""
#     outputFileName = ""
#     testFileName = ""
#
#     def __init__(self):
#         self.rand = str(uuid.uuid4())
#         self.inputFileName = "issues_train_" + self.rand + ".txt"
#         self.outputFileName = "supervised_classifier_model_" + self.rand
#         self.testFileName = "issues_test_" + self.rand + ".txt"
#
#     def fit(self, xtrain, ytrain):
#         outfile = open(self.inputFileName, mode="w", encoding="utf-8")
#         for i in range(len(xtrain)):
#             # line = "__label__" + str(ytrain[i]) + " " + xtrain[i]
#             line = xtrain[i]
#             outfile.write(line + '\n')
#         outfile.close()
#         p1 = subprocess.Popen(["cd", "/C",
#                                "fasttext supervised -input " + self.inputFileName + " -output " + self.outputFileName + " -epoch 500 -wordNgrams 4 -dim 300 -minn 4 -maxn 6 -pretrainedVectors pretrain_model.vec"],
#                               stdout=subprocess.PIPE)
#         p1.communicate()[0].decode("utf-8").split("\r\n")
#
#     def predict(self, xtest):
#         # save test file
#         outfile = open(self.testFileName, mode="w", encoding="utf-8")
#         for i in range(len(xtest)):
#             outfile.write(xtest[i] + '\n')
#         outfile.close()
#         # get predictions
#         p1 = subprocess.Popen(["cd", "/C", "fasttext predict " + self.outputFileName + ".bin " + self.testFileName],
#                               stdout=subprocess.PIPE)
#         output_lines = p1.communicate()[0].decode("utf-8").split("\r\n")
#         test_pred = [int(p.replace('__label__', '')) for p in output_lines if p != '']
#         return test_pred


pretrain_files = ['apache_pretrain.csv',
                  'jira_pretrain.csv',
                  'spring_pretrain.csv',
                  'talendforge_pretrain.csv',
                  'moodle_pretrain.csv',
                  'appcelerator_pretrain.csv',
                  'duraspace_pretrain.csv',
                  'mulesoft_pretrain.csv',
                  'lsstcorp_pretrain.csv']

pretrained = None

for file in pretrain_files:
    df_pretrain = pd.read_csv('PretrainData/' + file, usecols=['issuekey', 'title', 'description'])
    if (pretrained is not None):
        pretrained = pd.concat([pretrained, df_pretrain])
    else:
        pretrained = df_pretrain

pretrained = pretrained.dropna(how='any')

pretrained['title_desc'] = (pretrained['title'].str.lower() + ' - ' + pretrained['description'].str.lower()).apply(lambda x: cleanData(str(x)))

outfile = open("issues_pretrain.txt", mode="w", encoding="utf-8")
for line in pretrained.title_desc.values:
    outfile.write(line + '\n')
outfile.close()

#Selecting Training and Testing sets
def rebuild_kfold_sets(folds, k, i):
    training_set = None
    testing_set = None

    for j in range(k):
        if (i == j):
            testing_set = folds[i]
        elif (training_set is not None):
            training_set = pd.concat([training_set, folds[j]])
        else:
            training_set = folds[j]

    return training_set, testing_set

#Defining the evaluation criteria

import itertools


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_confusion_matrix_with_accuracy(classes, y_true, y_pred, title, sum_overall_accuracy, total_predictions):
    cm = ConfusionMatrix(y_true, y_pred)
    print('Current Overall accuracy: ' + str(cm.stats()['overall']['Accuracy']))
    if total_predictions != 0:
        print('Total Overall Accuracy: ' + str(sum_overall_accuracy / total_predictions))
    else:
        print('Total Overall Accuracy: ' + str(cm.stats()['overall']['Accuracy']))

    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure()
    plot_confusion_matrix(conf_matrix, classes=classes, title=title)
    plt.show()


from sklearn.metrics import confusion_matrix
# from pandas_ml import ConfusionMatrix

# K-folds cross validationdf
# K=5 or K=10 are generally used.
# Note that the overall execution time increases linearly with k
k = 5

# Define the classes for the classifier
classes = ['0', '1', '2']

# Make Dataset random before start
df_rand = df.sample(df.storypoint.count(), random_state=99)

# Number of examples in each fold
fsamples = int(df_rand.storypoint.count() / k)

# Fill folds (obs: last folder could contain less than fsamples datapoints)
folds = list()
for i in range(k):
    folds.append(df_rand.iloc[i * fsamples: (i + 1) * fsamples])

# Init
sum_overall_accuracy = 0
total_predictions = 0

# Repeat k times and average results
for i in range(k):

    # 1 - Build new training and testing set for iteration i
    training_set, testing_set = rebuild_kfold_sets(folds, k, i)
    y_true = testing_set.storypoint.tolist()

    # 2 - Oversample (ONLY TRAINING DATA)
    X_resampled, y_resampled = SimpleOverSample(training_set.label_title_desc.values.tolist(),
                                                training_set.storypoint.values.tolist())

    # 3 - train
    clf = LinearSVC(C=0.025,penalty='l1', loss='squared_hinge', dual=False,max_iter=4000,random_state=50)
    clf.fit(X_resampled, y_resampled)

    # 4 - Predict
    y_pred = clf.predict(testing_set.label_title_desc.values.tolist())

    # 3 - Update Overall Accuracy
    for num_pred in range(len(y_pred)):
        if (y_pred[num_pred] == y_true[num_pred]):
            sum_overall_accuracy += 1
        total_predictions += 1

    # 4 - Plot Confusion Matrix and accuracy
    plot_confusion_matrix_with_accuracy(classes, y_true, y_pred,    
                                        'Confusion matrix (testing-set folder = ' + str(i) + ')', sum_overall_accuracy,
                                        total_predictions)






