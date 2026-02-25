##### CICIDS2017 DATASET #####

### IMPORTS ###
#Basic Imports
import pandas as pd #Handling dataframes
import numpy as np #Handling mathematics
import time #Timing models training and testing

#Data Preprocessing Imports
from sklearn.preprocessing import LabelEncoder #Handle label encoding
from sklearn.preprocessing import StandardScaler #Scale dataset
from sklearn.model_selection import train_test_split #Split dataset into training and test sets

#EDA Imports
import matplotlib.pyplot as plt #Handling plots
import seaborn as sns #Handling performance metric graph

#Machine Learning Imports
from sklearn.ensemble import RandomForestClassifier #Random Forest classifier model
from sklearn.tree import DecisionTreeClassifier #Decision Tree classifier model
from sklearn.ensemble import GradientBoostingClassifier #Gradient Boosting classifier model
from xgboost import XGBClassifier #XGBoost classifier model
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score #Evaluate models on different metrics

#Deep Learning Imports
import tensorflow as tf

### LOAD DATASET ###
cicids2017 = pd.read_csv('Datasets/IoTNetworkIntrusionDataset.csv')

### INITIAL DATA PREPROCESSING ###
#Check for null values in dataset
print(cicids2017.isnull().sum())

#Replace infinite and nan values in the dataset   
cicids2017 = cicids2017.replace([np.inf, -np.inf], np.nan)
cicids2017 = cicids2017.fillna(0)

### EXPLORATORY DATA ANALYSIS ###
#Tuple containing number of rows and columns in dataset
print("===== CICIDS2017 Initial Analysis =====")
rows, columns = cicids2017.shape    
print(f"Rows: {rows}, Columns: {columns}")

#Display first 5 rows of dataset
print(cicids2017.head(5))

#Numerical data analysis
print("===== Numerical Data Analysis =====")
#Find numerical data
num_cols = cicids2017.select_dtypes(include=["int64", "float64"]).columns
#Create histograms to display data
cicids2017[num_cols].hist(figsize=(20,30))
plt.show()

#Categorical data analysis
print("===== Categorical Data Analysis =====")
#Create Bar plot for attack class
plt.figure(figsize=(10, 5))
sns.countplot(data=cicids2017, x="Label", order=cicids2017["Label"].value_counts().index)
plt.title('Distribution')
plt.tight_layout()
plt.show()

#Create Box plot for different numerical features vs label
num_feats = ["Flow_Duration", "Tot_Fwd_Pkts", "Tot_Bwd_Pkts", "Fwd_Pkt_Len_Mean"]
for col in num_feats:
    sns.boxplot(data=cicids2017, x='Label', y=col)
    plt.title(f"{col} by Label")
    plt.yscale('log')  # Use log scale for skewed data
    plt.show()

#Show distribution of anomaly vs normal
print("===== Attack Distribution =====")
print(cicids2017['Label'].value_counts())

### CONTINUED DATA PREPROCESSING ###
#Label encoding
le = LabelEncoder()
cicids2017['Label'] = le.fit_transform(cicids2017['Label'])

#Editing column names to lowercase and no white space
cicids2017.columns = (
    cicids2017.columns
    .str.strip()    #Remove leading and trailing whitespace
    .str.lower()    #Change all characters to lowercase
    .str.replace(' ', '_', regex=False) #Replace spaces with underscores
    .str.replace('(', '', regex=False)  #Remove open brackets (replace with nothing)
    .str.replace(')', '', regex=False)  #Remove closed brackets (replace with nothing)
)

#Dropping unnecessary columns
cicids2017 = cicids2017.drop(columns=['flow_id', 'src_ip', 'dst_ip', 'timestamp', 'cat', 'sub_cat'])

#Setting y to the column to be predicted
y = cicids2017['label']

#Setting X to dataframe without column to be predicted
X = cicids2017.drop('label', axis=1)

#Initiate scaler
scaler = StandardScaler()

# Fit and transform the features
X_scaled = scaler.fit_transform(X)

#Creating X and Y test and train sets with an 80:20 split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

### MACHINE LEARNING MODELS ###
print("\n===== MACHINE LEARNING MODELS =====")

#Dictionary for Machine Learning Classifiers
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

#Loop through each classifier in dictionary and output performance metrics
for name, model in classifiers.items():
    print(f"\n{name} Score: ")
    
    #Start timer to time model training
    start_time = time.time()
    
    #Train model on dataset
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    #End time
    end_time = time.time()
    
    #Calculate performance metrics (* 100 for percentages)
    accuracy = accuracy_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    elapsed_time = end_time - start_time
    
    #Add metric score to performance dictionary for graph
    performance['Classifier'].append(name)
    performance['Accuracy'].append(accuracy)
    performance['F1 Score'].append(f1)
    performance['Precision'].append(precision)
    performance['Recall'].append(recall)
    
    #Output metrics to console for classifier
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"F1 Score: {f1:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
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

#Start timer to time model training time
start_time = time.time()

#Train model on training sets
model.fit(X_train, y_train, epochs=10, batch_size=64, shuffle=True, validation_data=(X_test, y_test))  

#End timer
end_time = time.time()

# Make predictions
y_pred_prob = model.predict(X_test)

# Convert probabilities to binary predictions (0 or 1)
y_pred = (y_pred_prob > 0.5).astype(int)

#Output the prediction results
print("Predictions:", y_pred[:8].flatten())
print("Actual Values:", np.array(y_test[:8].values))

#Calculate performance metrics
test_loss, test_acc = model.evaluate(X_test, y_test)
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100
elapsed_time = end_time - start_time

#Add DL model metrics to performance dictionary for graph
performance['Classifier'].append('DNN')
performance['Accuracy'].append(test_acc * 100)
performance['F1 Score'].append(f1)
performance['Precision'].append(precision)
performance['Recall'].append(recall)

#Output Metrics to console for DNN
print(f"Test Accuracy: {test_acc * 100:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")
print(f"Time Elapsed: {elapsed_time:.2f} seconds")

#Output Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

### PERFORMANCE VISUALISATION ###
#Convert performance dictionary to dataframe
df_perf = pd.DataFrame(performance)

#Melt the DataFrame for seaborn
df_melted = df_perf.melt(id_vars='Classifier', var_name='Metric', value_name='Score')

#Plot using seaborn using melted performance dataframe
plt.figure(figsize=(12, 6))
sns.barplot(data=df_melted, x='Classifier', y='Score', hue='Metric')
plt.title('Classifier Performance Metrics - CICIDS2017')
plt.ylabel('Score (%)')
plt.ylim(96, 100)   #Start y-axis at 96 for clearer visualisation
plt.legend(title='Metric')
plt.tight_layout()
plt.show()
