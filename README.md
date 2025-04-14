# 🩺 Stroke Prediction using Healthcare Dataset
# 📁 Dataset: healthcare-dataset-stroke-data.csv
# 📄 Project Overview:
This project uses a stroke prediction dataset from the healthcare domain. 
The goal is to analyze patient records to identify key risk factors and predict the likelihood of a stroke using various classification models.

# 📊 Dataset Description:
Source: /kaggle/input/healthcare-dataset-stroke-data.csv

Records: 5110 entries

Columns: 12 features including:

id: Unique identifier

gender: Male, Female, or Other

age: Patient age

hypertension: 0 = No, 1 = Yes

heart_disease: 0 = No, 1 = Yes

ever_married: Yes/No

work_type: Type of employment

Residence_type: Urban or Rural

avg_glucose_level: Average glucose level

bmi: Body Mass Index (some missing values)

smoking_status: smoking habits

stroke: 0 = No stroke, 1 = Stroke

# 🧹 Preprocessing Steps:
Loaded and explored the dataset.

Identified and handled missing values in the bmi column.

Encoded categorical features using LabelEncoder.

Used StandardScaler for normalization of numerical features.

Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

# 📈 Exploratory Data Analysis:
Explored distributions of continuous variables like age, avg_glucose_level, and bmi using Seaborn KDE plots.

Visualizations gave insight into how numeric features vary and their potential impact on stroke risk.

# 🧠 Models Used:
Logistic Regression

Support Vector Classifier (SVC)

Decision Tree Classifier

Evaluation metrics included:

Accuracy

Precision

Recall

F1 Score

ROC AUC

# 🔧 Libraries:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# 📌 Key Questions Explored:
Does age significantly impact stroke risk?

How do average glucose level and BMI influence stroke probability?

Are there gender-based differences in stroke occurrence?

How does smoking status correlate with stroke?

# 🧾 Conclusion:
This project demonstrates how exploratory data analysis, preprocessing, and machine learning can be combined to tackle real-world healthcare problems. 
The models help understand and predict the likelihood of stroke, aiding preventive health strategies.
