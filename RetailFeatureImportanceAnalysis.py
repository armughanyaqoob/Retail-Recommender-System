

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score




# Load the dataset
articles_df = pd.read_csv("D:/New folder/articles.csv")
customers_df = pd.read_csv('D:/New folder/customers.csv')

retailers_df = pd.read_csv("D:/New folder/Retailers_Dataset.csv")




fraction = 0.001

# Randomly sample a fraction of the dataset
retailers_sampled_df = retailers_df.sample(frac=fraction, random_state=42)

# Print the size of the original and sampled datasets
print("Original dataset size:", retailers_df.shape)
print("Sampled dataset size:", retailers_sampled_df.shape)

merged_df = pd.merge(retailers_sampled_df, articles_df, on='article_id')
merged_df = pd.merge(merged_df, customers_df, on='customer_id')



import numpy as np
import lightgbm as lgb

# Prepare the feature matrix and target variable
X = merged_df.drop(['price'], axis=1)  # Feature matrix
y = merged_df['price']  # Target variable

# Remove columns with incompatible data types
columns_to_remove = ['Data', 'customer_id', 'article_id', 'sales_channel_id', 'prod_name', 'product_group_name',
                     'graphical_appearance_name', 'perceived_colour_value_name', 'perceived_colour_master_name',
                     'index_code', 'index_name', 'index_group_name', 'section_name', 'garment_group_name',
                     'detail_desc', 'club_member_status', 'fashion_news_frequency', 'postal_code']
X_encoded = X.drop(columns_to_remove, axis=1, errors='ignore')

# Convert categorical columns to numeric using one-hot encoding
categorical_cols = ['product_code', 'product_type_name', 'colour_group_name', 'department_name', 'FN']
X_encoded = pd.get_dummies(X_encoded, columns=categorical_cols)

# Drop remaining columns with object data type
X_encoded = X_encoded.select_dtypes(exclude=['object'])

train_ratio = 0.8  # 80% for training, 20% for testing
train_size = int(train_ratio * len(merged_df))
X_train, X_test = X_encoded[:train_size], X_encoded[train_size:]
y_train, y_test = y[:train_size].values, y[train_size:].values  # Convert to NumPy arrays

# Define the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Set the hyperparameters for LightGBM
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
# Train the LightGBM model
model = lgb.train(params, train_data, num_boost_round=100)

# Predict on the test set
y_pred = model.predict(X_test)

# Access valuable information from the trained model
feature_importances = model.feature_importance()
important_features = X_encoded.columns[np.argsort(feature_importances)[::-1]][:10]

# Print the important features and their importances
print("Important Features:")
for feature, importance in zip(important_features, feature_importances):
    print(f"{feature}: {importance}")
from sklearn.metrics import mean_squared_error
# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)




import matplotlib.pyplot as plt

# Get feature importances
feature_importance = model.feature_importance()
feature_names = model.feature_name()

# Sort feature importances in descending order
sorted_indices = np.argsort(feature_importance)[::-1]

# Select the top 5 to 6 features
top_features = sorted_indices[:6]  # Adjust the number as per your preference

# Get the names and importances of the top features
top_feature_names = [feature_names[i] for i in top_features]
top_feature_importance = [feature_importance[i] for i in top_features]

# Plotting feature importances
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(range(len(top_feature_names)), top_feature_importance, align='center')
ax.set_yticks(range(len(top_feature_names)))
ax.set_yticklabels(top_feature_names)
ax.invert_yaxis()
plt.title("Top Feature Importances", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.show()





# Graph showing Trend of 10 values 
plt.figure(figsize=(8, 6))
plt.plot(y_test[:10], label='Actual')
plt.plot(y_pred[:10], label='Predicted')
plt.title("Actual vs Predicted Values", fontsize=16)
plt.xlabel("Sample", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend()
plt.show()




import matplotlib.pyplot as plt
# Plot actual vs. predicted values
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
ax.scatter(range(len(y_test)), y_pred, color='red', label='Predicted')
plt.title("Actual vs. Predicted Values", fontsize=16)
plt.xlabel("Data Point", fontsize=14)
plt.ylabel("Value", fontsize=14)
plt.legend()
plt.show()




