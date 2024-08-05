# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


# Load the dataset from the CSV file
df = pd.read_csv("C:/Users/Asus/OneDrive - Universiti Teknikal Malaysia Melaka/SEM 4/NN/brake_Modifieds.csv")


# In[3]:


df.head()


# In[4]:


df['km'] = df['km'].str.replace(',', '').astype(float)


# In[5]:


# Split the data into features (X) and target variable (y)
X = df.drop('worn', axis=1)
y = df['worn']


# In[6]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Normalize the input features
X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()


# In[8]:


# Build the ANN model
model = Sequential()
model.add(Dense(16, activation='relu', input_shape=(X_train.shape[1],)))
model.add(Dense(1, activation='sigmoid'))


# In[9]:


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[10]:


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)


# In[11]:


# Print training data
print("Training Data:")
print(X_train)
print(y_train)


# In[12]:


# Print testing data
print("Testing Data:")
print(X_test)
print(y_test)


# In[13]:


# Test the model and print prediction output
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()


# In[14]:


# Print the predicted output
print("Prediction Output:")
print(y_pred)


# In[15]:


# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[16]:


# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


# In[17]:


# Classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

