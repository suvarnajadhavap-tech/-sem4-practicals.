# Sem-4-all-practicals-
All practicals sem 4

AML PRACTICALS 
PRAC-01 

import pandas as pd 

import matplotlib.pyplot as plt 

from sklearn.datasets import load_iris 

from sklearn.model_selection import train_test_split 

from sklearn.tree import DecisionTreeClassifier, plot_tree 

from sklearn.metrics import accuracy_score, classification_report 


iris = load_iris() 

X = iris.data 

y = iris.target 


X_train, X_test, y_train, y_test = train_test_split( 

    X, y, 

    test_size=0.2, 

    random_state=42 

)

classifier = DecisionTreeClassifier( 

    criterion='entropy', 

    random_state=42 

) 

 

classifier.fit(X_train, y_train) 

 

y_pred = classifier.predict(X_test) 



accuracy = accuracy_score(y_test, y_pred) 

print("Accuracy:", accuracy) 

 
 

print("\nClassification Report:") 

print(classification_report(y_test, y_pred, 

                            target_names=iris.target_names)) 

 

plt.figure(figsize=(15, 10)) 

 

plot_tree( 

    classifier, 

    feature_names=iris.feature_names, 

    class_names=iris.target_names, 

    filled=True, 

    rounded=True, 

    fontsize=10 

) 

 

plt.title("Decision Tree Classifier - Iris Dataset") 

plt.show()


|DAV PRACTICALS|
•PRAC-01

import pandas as pd
import numpy as np

data = pd.read_csv('C:\\Users\\Admin\\Documents\\Book1.csv')
print(data)

for i in data.columns:
    if i != 'student_name':
        mean = np.mean(data[i])
        median = np.median(data[i])
        std = np.std(data[i])

        print(f'''For subject '{i}'
            mean = {mean}
            median={median}
            std_dev = {std}''')

print('Students who scored above average in FML')
print(data['student_name'][data['fml']>=np.mean(data['fml'])])


