import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
 
def load_and_prepare_data(filepath, columns_to_keep, chunksize=10000):
    normal_sessions = []
    abnormal_sessions = []
 
    # Load the dataset in chunks
    for chunk in pd.read_csv(filepath, usecols=columns_to_keep, chunksize=chunksize, dtype=str):
        # Separate normal and abnormal sessions
        normal_chunk = chunk[chunk['session_label'] == '0']
        abnormal_chunk = chunk[chunk['session_label'] == '1']
 
        # Append to respective lists
        normal_sessions.append(normal_chunk)
        abnormal_sessions.append(abnormal_chunk)
 
        # If we have enough sessions, break the loop
        if sum(len(c) for c in normal_sessions) >= 88271 and sum(len(c) for c in abnormal_sessions) >= 88271:
            break
 
    # Concatenate the collected chunks
    normal_sessions = pd.concat(normal_sessions, ignore_index=True).head(88271)
    abnormal_sessions = pd.concat(abnormal_sessions, ignore_index=True).head(88271)
 
    # Combine normal and abnormal sessions
    combined_sessions = pd.concat([normal_sessions, abnormal_sessions], ignore_index=True)
 
    # Shuffle the combined sessions
    combined_sessions = combined_sessions.sample(frac=1, random_state=42).reset_index(drop=True)
 
    return combined_sessions
# Define columns to keep
testing_columns = ['session_id', 'date', 'user', 'pc', 'activity', 'session_label']
 
# Load and prepare the dataset with reduced size
reduced_set = load_and_prepare_data('C:/Users/bayad/Downloads/labeled_test_session.csv',
                                    testing_columns)
 
# Initialize LabelEncoders and encode the data
def encode_data(df, columns_to_encode):
    encoders = {}
    for column in columns_to_encode:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        encoders[column] = le
    return df, encoders
 
 
# Initialize LabelEncoders and encode the data
columns_to_encode = ['user', 'pc']  # Add other columns if needed
encoded_set, encoders = encode_data(reduced_set, columns_to_encode)
 
'''# Split into 70% training and 30% testing
train_set = encoded_set.sample(frac=0.7, random_state=42)
test_set = encoded_set.drop(train_set.index)'''
 
# Split into 70% training and 30% testing
train_set, test_set = train_test_split(encoded_set, test_size=0.3, random_state=42,
                                       stratify=encoded_set['session_label'])
# Print lengths of the splits
print("Length of training set:", len(train_set))
print("Length of testing set:", len(test_set))
print(train_set.head())
print(test_set.head())
# Convert date column to timestamp
train_set['date'] = pd.to_datetime(train_set['date'], errors='coerce').astype(np.int64) // 10**9
test_set['date'] = pd.to_datetime(test_set['date'], errors='coerce').astype(np.int64) // 10**9
 
# Convert the 'user', 'pc', and 'activity' columns to strings to ensure uniformity
train_set[['activity']] = train_set[['activity']].astype(str)
test_set[['activity']] = test_set[['activity']].astype(str)
 
print(train_set.head())
print(test_set.head())
 
# Tokenize characters in the activity strings
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(pd.concat([train_set['activity'],test_set['activity']]))
 
# Convert text to sequences of character indices
X_train_sequences = tokenizer.texts_to_sequences(train_set['activity'])
X_test_sequences = tokenizer.texts_to_sequences(test_set['activity'])
 
# Pad sequences
max_length = max(max(len(seq) for seq in X_train_sequences), max(len(seq) for seq in X_test_sequences))
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_length, padding='post')
 
 
print(X_train_sequences[:5])
print(X_test_sequences[:5])
print(X_train_padded.shape)
print(X_test_padded.shape)
 
# Prepare X and y for training and testing
X_train_features = train_set[['date', 'user', 'pc']].values
X_test_features = test_set[['date', 'user', 'pc']].values
y_train = train_set['session_label'].values
y_test = test_set['session_label'].values
 
# Ensure that y_train and y_test are in numeric format
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
 
# Scaling the features
scaler = MinMaxScaler()
X_train_features = scaler.fit_transform(X_train_features)
X_test_features = scaler.transform(X_test_features)
 
# Combine padded activity sequences with other features
X_train_combined = np.concatenate([X_train_padded, X_train_features], axis=1).reshape(X_train_padded.shape[0], max_length + X_train_features.shape[1], 1)
X_test_combined = np.concatenate([X_test_padded, X_test_features], axis=1).reshape(X_test_padded.shape[0], max_length + X_test_features.shape[1], 1)
 
 
# Define the model
model = Sequential()
# Add an LSTM layer
model.add(LSTM(100, input_shape=(X_train_combined.shape[1], 1), return_sequences=False))
# Add a Dense layer for classification
model.add(Dense(1, activation='sigmoid'))
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Print the model summary
model.summary()
# Train the simplified model
history = model.fit(
    X_train_combined, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)
# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_combined, y_test, verbose=1)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')
# Predict on test data
y_pred = model.predict(X_test_combined)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary labels
 
# Evaluate predictions
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
 
print("confusion_matrix")
print(confusion_matrix(y_test, y_pred))
print("classification_report")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
 
# Assuming y_test and y_pred are already defined
conf_matrix = confusion_matrix(y_test, y_pred)
# Assuming your model is named 'model'
model.save('C:/Users/bayad/Downloads/InsiderModel.keras')
labels = ['Normal', 'Abnormal']
 
# Create pie charts for each true label
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
 
for i, label in enumerate(labels):
    axes[i].pie(conf_matrix[i], labels=['Predicted Normal', 'Predicted Abnormal'],
                autopct='%1.1f%%', startangle=90, colors=['#21457d', '#90b1d2'])
    axes[i].set_title(f'True {label}', fontsize=14)
 
plt.suptitle('Confusion Matrix - Pie Charts', fontsize=16)
plt.show()
 
# Plot classification reports with Bar Charts
# Assuming y_test and y_pred are already defined
report_dict = classification_report(y_test, y_pred, output_dict=True)
 
# Define the labels for the bar chart
labels = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
 
# Map original labels to 'Normal' and 'Abnormal'
label_mapping = {'0': 'Normal', '1': 'Abnormal'}
mapped_labels = [label_mapping.get(label, label) for label in labels]
 
# Extract precision, recall, and F1 scores
precision = [report_dict[label]['precision'] for label in labels]
recall = [report_dict[label]['recall'] for label in labels]
f1_scores = [report_dict[label]['f1-score'] for label in labels]
 
# Create the bar chart
x = np.arange(len(labels))  # Label locations
width = 0.2  # Width of bars
 
fig, ax = plt.subplots(figsize=(12, 6))
 
# Define custom colors for bars
precision_color = '#e0e1e3'  # Light Grey
recall_color = '#21457d'  # Dark Blue
f1_color = '#87c7f3'  # Light Blue
 
bars1 = ax.bar(x - width, precision, width, label='Precision', color=precision_color)
bars2 = ax.bar(x, recall, width, label='Recall', color=recall_color)
bars3 = ax.bar(x + width, f1_scores, width, label='F1 Score', color=f1_color)
 
ax.set_xlabel('Classes')
ax.set_ylabel('Scores')
ax.set_title('Classification Report Metrics')
ax.set_xticks(x)
ax.set_xticklabels(mapped_labels)
ax.legend()
 
fig.tight_layout()
plt.show()