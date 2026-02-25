##### NSL-KDD DATASET #####

### IMPORTS ###
#Basic Imports
import pandas as pd #Handling dataframes
import numpy as np #Handling mathematics
import time #Calculate time elapsed for each model

#EDA imports
import matplotlib.pyplot as plt #Handling plots
import seaborn as sns #Handling performance metric graph

#Data Preprocessing Imports
from sklearn.preprocessing import LabelEncoder #Handle label encoding
from sklearn.preprocessing import StandardScaler #Scale dataset
from sklearn.model_selection import train_test_split #Split dataset into training and test sets

#Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier #Random Forest classifier model
from sklearn.tree import DecisionTreeClassifier #Decision Tree classifier model
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosting classifier model
from xgboost import XGBClassifier #XGBoost classifier model

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score #Evaluate models on different metrics

#Deep Learning Imports
import tensorflow as tf

### LOAD DATASETS ###
kddtest = pd.read_csv('Datasets/KDDTest.csv')
kddtrain = pd.read_csv('Datasets/KDDTrain.csv')

### INITIAL DATA PREPROCESSING ###
#Check for null values in dataset
print(kddtest.isnull().sum())    
print(kddtrain.isnull().sum()) 

#Rename 'attack_class' column to 'label' in both datasets
kddtest.rename(columns={'attack_class': 'label'}, inplace=True)
kddtrain.rename(columns={'attack_class': 'label'}, inplace=True)

#Changing 'label' contents to 'normal' or 'abnormal'
kddtest['label'] = kddtest['label'].apply(lambda x: 'normal' if x == 'normal' else 'abnormal')
kddtrain['label'] = kddtrain['label'].apply(lambda x: 'normal' if x == 'normal' else 'abnormal')

### EXPLORATORY DATA ANALYSIS ###
#Tuple containing number of rows and columns in dataset
print("===== NSL-KDD Initial Analysis =====")
rows, columns = kddtest.shape    
print(f"Test: Rows: {rows}, Columns: {columns}")
rows, columns = kddtrain.shape    
print(f"Train: Rows: {rows}, Columns: {columns}")

#Display first 5 rows of dataset
print("=== Test ===")
print(kddtest.head(5))
print("=== Train ===")
print(kddtrain.head(5))

#Categorical data analysis
print("===== Categorical Data Analysis =====")
plt.figure(figsize=(10, 5))
sns.countplot(data=kddtest, x="label", order=kddtest["label"].value_counts().index)
plt.title('Distribution')
plt.tight_layout()
plt.show()

#Show distribution of anomaly vs normal
print("===== Attack Distribution =====")
print(kddtest['label'].value_counts())   

### CONTINUED DATA PREPROCESSING ###
#Removing unnecessary columns
kddtest.drop('num_outbound_cmds', axis=1, inplace=True)
kddtrain.drop('num_outbound_cmds', axis=1, inplace=True)

#One-hot encoding
kddtest = pd.get_dummies(kddtest, columns = ['protocol_type', 'service', 'flag'])
kddtrain = pd.get_dummies(kddtrain, columns = ['protocol_type', 'service', 'flag'])

#Label encoding
le = LabelEncoder()
kddtest['label'] = le.fit_transform(kddtest['label'])
kddtrain['label'] = le.fit_transform(kddtrain['label'])

#Ensure both datasets have the same columns
kddtrain, kddtest = kddtrain.align(kddtest, join='left', axis=1)
kddtest.fillna(0, inplace=True)

#Setting y_train to column to be predicted
y_train = kddtrain['label']

#Setting X_train to dataframe without column being predicted 
X_train = kddtrain.drop('label', axis=1)

#Setting y_test to column to be predicted
y_test = kddtest['label']

#Setting X_test to dataframe without column being predicted
X_test = kddtest.drop('label', axis=1)

#Initiate scaler
scaler = StandardScaler()

#Scale X train data
X_train_scaled = scaler.fit_transform(X_train)

#Scale X test data
X_test_scaled = scaler.transform(X_test)

#Split data for validation set
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X_train_scaled, y_train, test_size=0.2, random_state=42)

### MACHINE LEARNING MODELS ###
print("\n===== MACHINE LEARNING MODELS =====")

#Dictionary for classifiers
classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=50),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier()
    }

#Dictionary to store classifiers metric scores for graph
performance = {
    'Classifier': [],
    'Accuracy': [],
    'F1 Score': [],
    'Precision': [],
    'Recall': []
}

#Loop through each classifier in dictionary
for name, model in classifiers.items():
    print(f"\n{name} Score: ")
    
    #Start timer for timing model training
    start_time = time.time()
    
    #Train model on dataset 
    model.fit(X_train_split, y_train_split)
    y_pred = model.predict(X_test_split)
    
    #End timer
    end_time = time.time()
    
    #Calculate performance metrics (* 100 for percentages)
    accuracy = accuracy_score(y_test_split, y_pred) * 100
    precision = precision_score(y_test_split, y_pred) * 100
    recall = recall_score(y_test_split, y_pred) * 100
    f1 = f1_score(y_test_split, y_pred) * 100
    elapsed_time = end_time - start_time
    
    #Add metric score to performance dictionary for graph
    performance['Classifier'].append(name)
    performance['Accuracy'].append(accuracy)
    performance['F1 Score'].append(f1)
    performance['Precision'].append(precision)
    performance['Recall'].append(recall)
    
    #Output metrics to console for classifer
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    
    #Output confusion matrix
    cm = confusion_matrix(y_test_split, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    #Separator for output formatting
    print("="*70, "\n")

### DEEP LEARNING MODEL ###
print("\n===== DEEP LEARNING MODEL =====")

#Create DL model
model = tf.keras.models.Sequential([ #Sequential = linear, output becomes input
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train_split.shape[1],)),  # 256 neurons, relu for learning complex patterns, input shape specifies number of input features
    tf.keras.layers.BatchNormalization(),    # Normalises output of prev layer, helps with overfitting
    tf.keras.layers.Dense(128, activation='relu'),   # 128 neurons, relu for complex patterns
    tf.keras.layers.Dense(64, activation='relu'),    # 64 neurons - gradual complexity reduction, relu for complex patterns
    tf.keras.layers.Dense(1, activation='sigmoid')   # 1 neuron - outputs value between 0 and 1, sigmoid for binary classification
])

#Compile model with optimiser and metrics
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) # adam adapts learning rate for each parameter, loss for binary classification, metrics to display

#Start timer for timing model training
start_time = time.time()

#Train model on training sets
model.fit(X_train_split, y_train_split, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test_split, y_test_split))  

#End timer
end_time = time.time()

# Make predictions
y_pred_prob = model.predict(X_test_split)

# Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

#Output the prediction results
print("Predictions:", y_pred[:8].flatten())
print("Actual Values:", np.array(y_test_split[:8].values))

#Calculate performance metrics
test_loss, test_acc = model.evaluate(X_test_split, y_test_split)
precision = precision_score(y_test_split, y_pred) * 100
recall = recall_score(y_test_split, y_pred) * 100
f1 = f1_score(y_test_split, y_pred) * 100
elapsed_time = end_time - start_time

#Add DL model metrics to performance dictionary for graph
performance['Classifier'].append('DNN')
performance['Accuracy'].append(test_acc * 100)
performance['F1 Score'].append(f1)
performance['Precision'].append(precision)
performance['Recall'].append(recall)

#Output metrics to console for DNN
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print(f"Time Elapsed: {elapsed_time:.2f} seconds")

#Output confusion matrix
cm = confusion_matrix(y_test_split, y_pred)
print(f"Confusion Matrix:\n{cm}")

### PERFORMANCE VISUALISATION ###
# Convert dictionary to dataframe
df_perf = pd.DataFrame(performance)

# Melt the dataframe for seaborn
df_melted = df_perf.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

# Plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Classifier', y='Score', hue='Metric')
plt.title('Classifier Performance Metrics - NSL-KDD')
plt.ylabel('Score (%)')
plt.ylim(96, 100) #Start y-axis at 96 for clearer visualisation
plt.legend(title='Metric')
plt.tight_layout()
plt.show()