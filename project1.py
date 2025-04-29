#1. Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix

#2. Loading and inspecting the dataset
df = pd.read_csv("KaggleV2-May-2016.csv")
print(df.head())
print(df.info())

#3. Data Cleaning

#From the information regarding the data, it is seen that, there are 110527 entries, 
#and all of the columns have 110527 non-null elements. Which implies that there are no missing values in the dataset.


# Renaming columns
df.columns = [col.strip().lower().replace('-', '_') for col in df.columns]

# Convert 'scheduledday' and 'appointmentday' to datetime
df['scheduledday'] = pd.to_datetime(df['scheduledday'])
df['appointmentday'] = pd.to_datetime(df['appointmentday'])

# Create new features
df['waiting_days'] = (df['appointmentday'] - df['scheduledday']).dt.days
df['appointment_weekday'] = df['appointmentday'].dt.day_name()

# Convert 'no_show' to binary (1 = no-show, 0 = show)
df['no_show'] = df['no_show'].map({'Yes': 1, 'No': 0})

# Remove unrealistic values
df = df[df['age'] >= 0]

#4. EDA - Analyzing Trends
sns.barplot(data=df, x='appointment_weekday', y='no_show',color='pink')
plt.title('No-Show Rate by Day of the Week')
plt.show()

sns.boxplot(data=df, x='no_show', y='age',color='green')
plt.title('Age Distribution vs No-Show')
plt.show()

sns.barplot(data=df, x='sms_received', y='no_show',color='purple')
plt.title('SMS Received vs No-Show Rate')
plt.show()

#5.Prepare Data for Modeling
features = ['age', 'scholarship', 'hipertension', 'diabetes', 
            'alcoholism', 'handcap', 'sms_received', 'waiting_days']
X = df[features]
y = df['no_show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#6.Train Decision Tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, y_train)

#7.Evaluate Model
y_pred = clf.predict(X_test)
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Tree
plot_tree(clf, feature_names=features, class_names=["Show", "No-Show"], filled=True)
plt.show()

#CONCLUSION:
#Decision Tree model is predicting that everyone will show up (class 0), which is why it's doing well on accuracy (80%), 
# but terribly on identifying actual no-shows (class 1).
#The model is highly biased toward class 0, likely due to class imbalance (26560 vs. 6598 is roughly a 4:1 ratio).
#While the accuracy is high, the recall for class 1 is 0, meaning the model fails to identify the minority class almost entirely.
#The F1-score for class 1 is only 0.01, which is extremely poor.
#So, trying other methods so as to increase the accuracy of predicting no-show cases.

#XGBoost Method:

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Apply SMOTE to balance the training set
sm = SMOTE(random_state=42)
X_resampled, y_resampled = sm.fit_resample(X_train_scaled, y_train)

#Train XGBoost classifier
xgb = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, eval_metric='logloss', random_state=42)
xgb.fit(X_resampled, y_resampled)

#Make predictions
y_pred = xgb.predict(X_test_scaled)
y_probs = xgb.predict_proba(X_test_scaled)[:, 1]  # probability of class 1

#Evaluation
print("Confusion Matrix: ")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#XGBoost

#Confusion Matrix: Recall for No-Show increased significantly (80%). However, the precision for No-Show dropped to 30%.
#XGBoost is one of the best models in terms of recall for class 1, which is what we need to detect no-shows.
#Precision is still low, meaning the model is over-predicting no-shows (false positives).
#After doing the undersampling, SMOTE oversampling, Random Forest Method and XGBoost method, by far 80% is the highest accuracy. 
# Hence this model with XGBoost method is the most suitable to predict whether patients will miss their appointments and 
# optimize scheduling. 

