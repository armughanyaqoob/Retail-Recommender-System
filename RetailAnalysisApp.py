

import findspark
findspark.init()




from pyspark.sql import SparkSession

spark= SparkSession.builder.master("local").appName("TestSpark").getOrCreate()

sc =spark.sparkContext




spark






import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score



# Load the dataset
articles_df = pd.read_csv("D:/New folder/articles.csv")
customers_df = pd.read_csv('D:/New folder/customers.csv')





retailers_df = pd.read_csv("D:/New folder/Retailers_Dataset.csv")



from matplotlib import pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(15, 7))
ax = sns.histplot(articles_df, y='garment_group_name', color='orange', hue='index_group_name', multiple="stack")
ax.set_xlabel('count by garment group')
ax.set_ylabel('garment group')
plt.show()





import seaborn as sns
from matplotlib import pyplot as plt
sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=customers_df, x='age', bins=50, color='blue')
ax.set_xlabel('Distribution of the customers age')
plt.show()




sns.set_style("darkgrid")
f, ax = plt.subplots(figsize=(10,5))
ax = sns.histplot(data=customers_df, x='club_member_status', color='orange')
ax.set_xlabel('Distribution of club member status')
plt.show()




retailers_df.head(3)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error





# Step 2: Feature selection
selected_features = ["customer_id", "article_id", "sales_channel_id"]
X = retailers_df[selected_features]
y = retailers_df["price"]  # Use the "price" column as the target variable

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM 
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint




retailers_df.info()





retailers_modified_df = retailers_df.drop(['customer_id', 'article_id','sales_channel_id'], axis=1)





retailers_modified_df.info()





retailers_modified_df = retailers_modified_df[retailers_modified_df['Data'] != 'Date']  # Remove rows with 'Date' string
retailers_modified_df['Data'] = pd.to_datetime(retailers_modified_df['Data'], errors='coerce')


# Convert 'price' column to numeric format
retailers_modified_df['price'] = pd.to_numeric(retailers_modified_df['price'], errors='coerce')

# Remove rows with invalid date or price values
retailers_modified_df = retailers_modified_df.dropna(subset=['Data', 'price'])




retailers_modified_df.info()



# Convert the "Data" column to datetime format
retailers_modified_df['Data'] = pd.to_datetime(retailers_modified_df['Data'], errors='coerce')

# Verify the updated data types
print(retailers_modified_df.dtypes)





import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



retailers_dataset=retailers_modified_df

retailers_dataset.isnull().sum()





#Convert the "Data" column to datetime format
retailers_dataset['Data'] = pd.to_datetime(retailers_dataset['Data'], errors='coerce')

# Check for any remaining missing values (just to be sure)
print(retailers_dataset.isnull().sum())



# Convert dates to ordinal values
retailers_dataset['Data_ordinal'] = retailers_dataset['Data'].apply(lambda x: x.toordinal())


# Split the data into features (X) and target (y)
X = retailers_dataset[['Data_ordinal']]
y = retailers_dataset['price']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




# Initialize the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)



# Make predictions on the test data
X_test_ordinal = X_test['Data_ordinal']
y_pred = model.predict(X_test_ordinal.values.reshape(-1, 1))





# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)




# Print the coefficients and intercept of the linear regression model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)



from datetime import datetime, timedelta
# Define the time frame for future predictions
start_date = max(retailers_dataset['Data']) + timedelta(days=1)  # Start from the next day after the last available date
end_date = start_date + timedelta(days=365)  # Predict for the next year

# Generate a new set of dates for prediction
# Generate a new set of dates for prediction (monthly)
future_dates = pd.date_range(start=start_date, end=end_date, freq='MS')


# Convert future dates to ordinal values
future_dates_ordinal = future_dates.to_series().apply(lambda x: x.toordinal())

# Reshape the future dates for prediction
future_dates_ordinal_reshaped = future_dates_ordinal.values.reshape(-1, 1)

# Make predictions for future sales
future_sales_predictions = model.predict(future_dates_ordinal_reshaped)

# Create a DataFrame to store the predictions
future_sales_df = pd.DataFrame({'Date': future_dates, 'Predicted Sales': future_sales_predictions})





import matplotlib.pyplot as plt
# Plot the graph
plt.figure(figsize=(10, 6))
plt.plot(future_sales_df['Date'], future_sales_df['Predicted Sales'], label='Predicted Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Future Sales Trend Prediction')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





# Convert the "Data" column to datetime format
retailers_dataset['Data'] = pd.to_datetime(retailers_dataset['Data'], errors='coerce')

# Extract the month and year from the "Data" column
retailers_dataset['Month'] = retailers_dataset['Data'].dt.month
retailers_dataset['Year'] = retailers_dataset['Data'].dt.year

# Group the data by month and calculate the average price
monthly_avg_price = retailers_dataset.groupby(['Year', 'Month'])['price'].mean().reset_index()

# Convert the Year and Month columns to a datetime format
monthly_avg_price['Date'] = pd.to_datetime(monthly_avg_price[['Year', 'Month']].assign(day=1))

# Sort the data by date
monthly_avg_price.sort_values('Date', inplace=True)

# Plot the graph
plt.figure(figsize=(10, 6))

# Plot the trend line
x = np.arange(len(monthly_avg_price))
coefficients = np.polyfit(x, monthly_avg_price['price'], 1)
trend_line = np.polyval(coefficients, x)
plt.plot(monthly_avg_price['Date'], trend_line, color='red', label='Trend Line')

plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Price Trend')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()






