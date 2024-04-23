import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# File to graph the original dataset with minimal modifications

# Load data
df = pd.read_csv('data/train.csv')

# Convert log_price to price
df['price'] = np.exp(df['log_price'])

# Ensure directory exists for saving figures
os.makedirs('training_metrics', exist_ok=True)

# Select numerical columns for the heatmap
numerical_columns = ['price', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'latitude', 'longitude', 'number_of_reviews', 'review_scores_rating']
num_df = df[numerical_columns]

# Create correlation heatmap
plt.figure(figsize=(22, 14))
heatmap = sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Numerical Features')
plt.savefig('visualized_dataset/numerical_correlation_heatmap.png')
plt.close()  # Close the plot to avoid displaying it here


# List of categorical columns
categorical_columns = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'cleaning_fee']

# Generate and save bar plots
for column in categorical_columns:
    plt.figure(figsize=(22, 14))
    sns.barplot(x=column, y='price', data=df,  errorbar=None)  # ci=None to avoid drawing error bars
    plt.title(f'Price vs {column}')
    plt.xticks(rotation=45)
    plt.savefig(f'visualized_dataset/price_vs_{column}.png')
    plt.close()  # Close the plot to avoid displaying it here


# Histogram plot
plt.figure(figsize=(10, 5))
plt.hist(df['price'], bins=50, color='skyblue', alpha=0.7)
plt.title(f'Distribution of Predicted Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.grid(True)
plt.savefig("visualized_dataset/distribution_plot.png")
plt.close()

# List of categorical columns
categorical_columns = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'host_has_profile_pic', 'host_identity_verified', 'instant_bookable', 'cleaning_fee']

# Generate and save bar plots
for column in categorical_columns:
    plt.figure(figsize=(22, 14))
    sns.barplot(x=column, y='price', data=df, errorbar=None)  # ci=None to avoid drawing error bars
    plt.title(f'Price vs {column}')
    plt.xticks(rotation=45)
    plt.savefig(f'visualized_dataset/price_vs_{column}.png')
    plt.close()  # Close the plot to avoid displaying it here
