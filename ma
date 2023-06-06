import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler

from scipy.stats import boxcox

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

import math
from collections import Counter
data = pd.read_csv(r'/Users/manalalnamani/Desktop/riceClassification.csv')
data.head()
data.shape
data.columns
data.nunique(axis=0)
data.describe().apply(lambda s: s.apply(lambda x: format(x, 'f')))
print("From all {all} rows, {num} of them are unique.".format(all = len(data),num = len(data.id.unique())))
data.info()
data = data.drop(columns = 'id', axis = 1)
list_of_num_features = data.loc[:, data.columns != 'Class']
palette_features = ['#E68753', '#409996']
sns.set(rc={'axes.facecolor':'#ECECEC'}) 

for feature in list_of_num_features:
    plt.figure(figsize=(12,6.5)) 
    plt.title(feature, fontsize=15, fontweight='bold', fontname='Helvetica', ha='center')
    ax = sns.boxplot(x = data['Class'], y = list_of_num_features[feature], data = data, palette=palette_features)
    for container in ax.containers:
        ax.bar_label(container)
    plt.show()
    columns = data.columns
columns = [c for c in columns if c not in ['Extent', 'Class']]

for col in columns:
    data[col] = boxcox(x=data[col])[0]
    
    sns.set(rc={'axes.facecolor':'#ECECEC'}) #background color of plot
plt.figure(figsize=(12,6))
plt.title("Target variable", fontsize=15, fontweight='bold', fontname='Helvetica', ha='center')
ax = sns.countplot(x=data['Class'], data=data, palette=palette_features)

abs_values = data['Class'].value_counts(ascending=True).values
ax.bar_label(container=ax.containers[0], labels=abs_values) 

plt.show()

corr = data.corr()

plt.figure(figsize = (20, 12))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, linewidths = 4, annot = True, fmt = ".2f", cmap="BrBG")
plt.show()

data = data.drop(['ConvexArea','EquivDiameter'], axis = 1)

columns = data.columns
columns = [c for c in columns if c not in ['Class']]
y = data['Class'] 
X = data[columns]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1) 
X_train.shape, X_test.shape, y_train.shape, y_test.shape

#implementing Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

#plotting Confusion Matrix
cf_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print(cf_matrix_lr)

ax = sns.heatmap(cf_matrix_lr/np.sum(cf_matrix_lr), annot=True, fmt='.2%', cmap='binary')

ax.set_title('Logistic Regression Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

plt.show()

print(classification_report(y_test, y_pred_lr))

print('Accuracy Score : ' + str(round(accuracy_score(y_test,y_pred_lr),3)))
print('Precision Score : ' + str(round(precision_score(y_test,y_pred_lr),3)))
print('Recall Score : ' + str(round(recall_score(y_test,y_pred_lr),3)))
print('F-Score : ' + str(round(f1_score(y_test,y_pred_lr),3)))

params = [{'penalty' : ['l2'], 'solver': ['lbfgs', 'liblinear'],
    'max_iter' : [1000, 5000, 10000], 'C': [20, 5,1,0.1,0.5]}]
lr_before_tuning = LogisticRegression()
lr_model_tuning = GridSearchCV(lr_before_tuning, param_grid = params, verbose=True, n_jobs=-1)
grid_lr_metrics = lr_model_tuning.fit(X_train, y_train)

y_lrc_pred_metrics = grid_lr_metrics.predict(X_test)
lr_tuned_accuracy = accuracy_score(y_test,y_lrc_pred_metrics)
lr_tuned_precision = precision_score(y_test,y_lrc_pred_metrics)
lr_tuned_recall = recall_score(y_test,y_lrc_pred_metrics)
lr_tuned_f1_score = f1_score(y_test,y_lrc_pred_metrics)

print('Most suitable parameters for Logistic Regression: ' + str(grid_lr_metrics.best_params_) + '\n')

confusion_matrix(y_test, y_lrc_pred_metrics)

cf_matrix_lr = confusion_matrix(y_test, y_lrc_pred_metrics)
print(cf_matrix_lr)

ax = sns.heatmap(cf_matrix_lr/np.sum(cf_matrix_lr), annot=True, fmt='.2%', cmap='binary')
ax.set_title('Logistic Regression Confusion Matrix\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

ax.xaxis.set_ticklabels(['0','1'])
ax.yaxis.set_ticklabels(['0','1'])

plt.show()

print(classification_report(y_test, y_lrc_pred_metrics))

print('Accuracy Score : ' + str(round(accuracy_score(y_test,y_lrc_pred_metrics),3)))
print('Precision Score : ' + str(round(precision_score(y_test,y_lrc_pred_metrics),3)))
print('Recall Score : ' + str(round(recall_score(y_test,y_lrc_pred_metrics),3)))
print('F-Score : ' + str(round(f1_score(y_test,y_lrc_pred_metrics),3)))
