##### TON IOT #####

### IMPORTS ###
#Basic Imports
import pandas as pd #Handling dataframes
import numpy as np #Handling mathematics
import time #Timing model training time

#EDA Imports
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
ton = pd.read_csv('Datasets/TON_IoT_Train_Test_Network.csv')

### EXPLORATORY DATA ANALYSIS ###
#Tuple containing number of rows and columns in dataset
print("===== TON IoT Initial Analysis =====")
rows, columns = ton.shape    
print(f"Rows: {rows}, Columns: {columns}")
#Display first 5 rows of dataset
print(ton.head(5))

#Categorical data analysis
print("===== Categorical Data Analysis =====")
plt.figure(figsize=(10, 5))
sns.countplot(data=ton, x="label", order=ton["label"].value_counts().index)
plt.title('Distribution')
plt.tight_layout()
plt.show()

### DATA PREPROCESSING ###
#Check for null values in dataset
print(ton.isnull().sum())

#Removing unnecessary columns
ton.drop(columns=['dns_query', 'dns_qclass', 'dns_qtype', 'dns_rcode', 'dns_AA', 'dns_RD', 'dns_RA', 'dns_rejected',
    'ssl_version', 'ssl_cipher', 'ssl_resumed', 'ssl_established', 'ssl_subject', 'ssl_issuer',
    'http_trans_depth', 'http_method', 'http_uri', 'http_version', 'http_user_agent',
    'http_orig_mime_types', 'http_resp_mime_types',
    'weird_name', 'weird_addl', 'weird_notice', 'type'], axis=1, inplace=True)

#Label encoding
le = LabelEncoder()
ton['src_ip'] = le.fit_transform(ton['src_ip'])
ton['dst_ip'] = le.fit_transform(ton['dst_ip'])
ton['proto'] = le.fit_transform(ton['proto'])
ton['service'] = le.fit_transform(ton['service'])
ton['conn_state'] = le.fit_transform(ton['conn_state'])

#Show distribution of anomaly vs normal
print("===== Attack Distribution =====")
print(ton['label'].value_counts())   

#Setting Y to dataframe the column to be predicted
y = ton['label']

#Setting X to dataframe without column to be predicted
X = ton.drop('label', axis=1)

#Initiate scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

#Creating X and Y test and train sets with an 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

### MACHINE LEARNING MODELS ###
print("\n===== MACHINE LEARNING MODELS =====")

#Dictionary for Classifiers
classifiers = {
        'Random Forest': RandomForestClassifier(n_estimators=50),
        'Decision Tree': DecisionTreeClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'XGBoost': XGBClassifier()
    }

#Dictionary to store Classifiers metric scores for graph
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
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #End timer
    end_time = time.time()
    
    #Calculate performance metrics (* 100 for percentages)
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    elapsed_time = end_time - start_time  # Time elapsed
    
    #Add metric score to performance dictionary for graph
    performance['Classifier'].append(name)
    performance['Accuracy'].append(accuracy)
    performance['F1 Score'].append(f1)
    performance['Precision'].append(precision)
    performance['Recall'].append(recall)
    
    #Output metrics to console for classifier
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Time Elapsed: {elapsed_time:.2f} seconds")
    
    #Output confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{cm}")
    
    #Separator for output formatting
    print("="*70, "\n")

### DEEP LEARNING MODEL ###
print("\n===== DEEP LEARNING MODEL =====")

#Create DL model
model = tf.keras.models.Sequential([ #Sequential = linear, output becomes input
    tf.keras.layers.Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # 256 neurons, relu for learning complex patterns, input shape specifies number of input features
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
model.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=True, validation_data=(X_test, y_test))  

#End timer
end_time = time.time()

#Make predictions
y_pred_prob = model.predict(X_test)

#Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

#Have model make predictions and output the results
print("Predictions:", y_pred[:8].flatten())
print("Actual Values:", np.array(y_test[:8].values))

#Calculate performance metrics
test_loss, test_acc = model.evaluate(X_test, y_test)
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
elapsed_time = end_time - start_time

#Add metric score to Performance dictionary
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

### PERFORMANCE VISUALISATION ###
# Convert dictionary to DataFrame
df_perf = pd.DataFrame(performance)

# Melt the DataFrame for seaborn
df_melted = df_perf.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

# Plot using seaborn
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Classifier', y='Score', hue='Metric')
plt.title('Classifier Performance Metrics - TON IoT')
plt.ylabel('Score (%)')
plt.ylim(96, 100) #Start y-axis at 96 for clearer visualisation
plt.legend(title='Metric')
plt.tight_layout()
plt.show()