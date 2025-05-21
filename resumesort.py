import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ai=pd.read_excel(r"C:\Users\UGANDA\Downloads\Book2.xlsx")
x = pd.get_dummies(ai[['degree', 'applied_role']])
y=ai['is_fit']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
model=LogisticRegression()
model.fit(x_train, y_train)
y_pred=model.predict(x_test)
print("Model Score: ",model.score(x_test, y_test))
print("Accuracy Score: ",accuracy_score(y_test, y_pred))
print("Precision Score: ",precision_score(y_test, y_pred))
print("Recall Score: ",recall_score(y_test, y_pred))
print("F1 Score: ",f1_score(y_test, y_pred))
print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
new_Data = pd.DataFrame([{
    'degree_B.Sc': 0,
    'degree_B.Tech': 0,
    'degree_M.Sc': 0,
    'degree_M.Tech': 0,
    'degree_MBA': 1,
    'degree_PhD': 0,
    'applied_role_Analyst': 0,
    'applied_role_Developer': 0,
    'applied_role_Manager': 1
}])
predection=model.predict(new_Data)
print(predection)

print(ai)