import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Load the dataset
df = pd.read_excel("network_logs.xlsx")

# Step 2: Encode the Protocol column manually (TCP=0, UDP=1, ICMP=2)
protocol_mapping = {'TCP': 0, 'UDP': 1, 'ICMP': 2}
df['Protocol'] = df['Protocol'].map(protocol_mapping)

# Step 3: Apply MinMaxScaler to the Length column
scaler = MinMaxScaler()
df['Length'] = scaler.fit_transform(df[['Length']])

# Step 4: Define features and labels
X = df[['Protocol', 'Length']]
y = df['Malicious']

# Step 5: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

# Additional Evaluation Metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Example of prediction vs actual labels
print("\nExample predictions vs actual values:")
for i in range(10):  # Displaying 10 sample predictions
    print(f"Predicted: {y_pred[i]}, Actual: {y_test.iloc[i]}")

# Function to predict based on input length and protocol
def predict_packet(protocol_input, length_input):
    # Ensure that the input protocol is encoded and length is scaled
    protocol_encoded = protocol_mapping.get(protocol_input.upper(), None)  # Encode protocol
    if protocol_encoded is None:
        print("Invalid Protocol input. Use 'TCP', 'UDP', or 'ICMP'.")
        return
    
    length_normalized = scaler.transform([[length_input]])[0][0]  # Normalize length
    
    # Predict using the trained model
    prediction = model.predict([[protocol_encoded, length_normalized]])
    
    # Output the prediction
    print(f"Prediction for Protocol: {protocol_input}, Length: {length_input} is: {'Malicious' if prediction[0] else 'Not Malicious'}")

# Test the prediction function with a sample input
predict_packet("ICP", 5400)  # Replace "TCP" and 54 with any test values you'd like to try


# -----------------------------
# Step 8: Predict on new dataset
# -----------------------------

# Load new dataset for prediction (same format as training set)
new_df = pd.read_excel("network_logs.xlsx")  # <-- Replace with your new file name

# Apply the same protocol mapping
new_df['Protocol'] = new_df['Protocol'].map(protocol_mapping)

# Apply the same scaler to the Length column
new_df['Length'] = scaler.transform(new_df[['Length']])

# Prepare features
X_new = new_df[['Protocol', 'Length']]

# Make predictions
new_predictions = model.predict(X_new)

# Add predictions to the new DataFrame
new_df['Predicted_Malicious'] = new_predictions

# Print malicious entries
malicious_entries = new_df[new_df['Predicted_Malicious'] == 1]
print("\nMalicious Entries Detected:")
print(malicious_entries)

# Optionally save the output to Excel
malicious_entries.to_excel("malicious_output.xlsx", index=False)
