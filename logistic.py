'''
Data from: http://www.statsci.org/data/general/titanic.html
ideally all three features Age, Sex, and Class are independent of one another.
'''

'''imports'''
import pandas as pd
import numpy as np
import csv


def read_data(file):
    f = open(file, 'r')
    data = {}
    arr, all_data = [], []
    lines = f.readlines()
    head = lines[0].split("\t")
    head[-1] = head[-1][0:-1]
    arr.append(head)
    all_data.append(head)
    for key in head:
        data[key] = []

    def break_off_name(str):
        c = 0
        name = ''
        rest = ''
        for i, char in enumerate(str):
            c += 1 if char == '"' else 0
            if c == 2:
                name = str[0:i+1]
                rest = str[i+2:]
                break
        c = 0
        for i, char in enumerate(str):
            c += 1 if char == '"' else 0
        if c == 2:
            return name, rest
        else:
            return 'SAMPLE NOT FORMATTED CORRECTLY', None

    for i, line in enumerate(lines):
        if i == 0:
            continue
        name, rest = break_off_name(line)
        if rest == None:
            continue
        entry = rest.split("\t")
        entry.insert(0, name)
        entry[-1] = entry[-1][0:-1] if len(entry[-1]) > 1 else entry[-1] #cut off the new line character when its there in the data
        entry[1] = entry[1][0:1] #make 3rd into '3' or '2nd' into '2'
        if entry[2] == 'NA': #filter out non-existent data
            entry[2] = 'NA'
            for idx, feature in enumerate(entry): #convert entries to numbers (where applicable) clean data
                if feature == 'male':
                    entry[idx] = 0
                elif feature == 'female':
                    entry[idx] = 1
                elif feature.isdigit() or feature == '-1':
                    entry[idx] = int(feature)
            all_data.append(entry)
        else:
            for idx, feature in enumerate(entry): #convert entries to numbers (where applicable) clean data
                if feature == 'male':
                    entry[idx] = 0
                elif feature == 'female':
                    entry[idx] = 1
                elif feature.isdigit():
                    entry[idx] = int(feature)
            for idx, feature in enumerate(entry):
                data[head[idx]].append(feature)

            arr.append(entry)
            all_data.append(entry)

    return data, arr, all_data

def write_csv(arr, path):
    f = open(path, 'w+')
    writer = csv.writer(f)
    for row in arr:
        writer.writerow(row)

    f.close()
def write_csv_no_name(arr, path):
    f = open(path, 'w+')
    writer = csv.writer(f)
    for row in arr:
        writer.writerow(row[1:])
    f.close()

if __name__ == '__main__':
    dict, arr, all = read_data("titanic.txt");
    print(len(all))
    print(len(arr))
    write_csv(arr, 'age_known_titatic.csv')
    # for entry in all:
    #     print(entry)
    # print('\n\n\n\nARR\n\n\n\n', arr)
    # df = pd.DataFrame(dict)
    # df[['Age', 'PClass', 'Survived', 'Sex']] = df[['Age', 'PClass', 'Survived', 'Sex']].apply(pd.to_numeric)
    # print(df)
    # X = df[['Age', 'PClass', 'Sex']]
    # y = df.Survived
    #
    # import sklearn
    # from sklearn.model_selection import train_test_split
    # X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
    # print('X_TRAIN',X_train)
    # print('Y_TRAIN', y_train)
    #
    # # import the class
    # from sklearn.linear_model import LogisticRegression
    #
    # # instantiate the model (using the default parameters)
    # logreg = LogisticRegression()
    # # logreg = LogisticRegressionCV(cv=10, scoring='roc_auc', n_jobs=-1)
    #
    # # fit the model with data (training)
    # logreg.fit(X_train,y_train)
    #
    # print('equation:', logreg.intercept_, logreg.coef_)
    # print('classes', logreg.feature_names_in_)
    #
    # y_pred=logreg.predict(X_test)
    #
    # from sklearn import metrics
    # cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
    # cnf_matrix
    #
    # # import required modules
    # import numpy as np
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # # %matplotlib inline
    # class_names=[0,1] # name  of classes
    # fig, ax = plt.subplots()
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(tick_marks, class_names)
    # plt.yticks(tick_marks, class_names)
    # # create heatmap
    # sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
    # ax.xaxis.set_label_position("top")
    # plt.tight_layout()
    # plt.title('Confusion matrix', y=1.1)
    # plt.ylabel('Actual label')
    # plt.xlabel('Predicted label')
    # plt.show()
    #
    # print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # print("Precision:",metrics.precision_score(y_test, y_pred))
    # print("Recall:",metrics.recall_score(y_test, y_pred))
    #
    # y_pred_proba = logreg.predict_proba(X_test)[::,1]
    # fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    # auc = metrics.roc_auc_score(y_test, y_pred_proba)
    # plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
    # plt.legend(loc=4)
    # plt.show()
