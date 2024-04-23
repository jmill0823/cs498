import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# File to predict the airbnb prices

# Load the Pre-trained Model
def load_model(path):
    return joblib.load(path)

# Load the New Test Data
def load_test_data(path):
    df = pd.read_csv(path)
    ids = df['id']
    X_test = df.drop(columns = ['id'])
    return X_test, ids

# Predict Function
def predict_prices(model, X):
    y_pred_log = model.predict(X)
    y_pred = np.exp(y_pred_log)
    return y_pred_log, y_pred

# Visualization Functions
def plot_price_distribution(prices, title):
    plt.figure(figsize=(10, 5))
    plt.hist(prices, bins=50, color='skyblue', alpha=0.7)
    plt.title(f'Distribution of {title}')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("prediction_metrics/distribution_plot.png")
    plt.close()

if __name__ == "__main__":

    # Model and test data paths
    model_path = 'price_prediction_model.pkl'
    test_data_path = 'prep_data/prep_test_data.csv'
    original_data_path = 'data/test.csv'  # Path to the original data

    # Load the trained model
    model = load_model(model_path)

    # Load new test data and IDs
    X_test, ids = load_test_data(test_data_path)

    # Make predictions
    y_pred_log, y_pred = predict_prices(model, X_test)

    # Visualize the distribution of predicted prices
    plot_price_distribution(y_pred, 'Predicted Prices')

    # Create DataFrame from predictions
    predictions_df = pd.DataFrame({
        'id': ids,
        'Predicted_Log_Price': y_pred_log,
        'Predicted_Price': y_pred
    })

    # Load original data
    original_data = pd.read_csv(original_data_path)

    # Merge original data with predictions based on 'id'
    merged_df = pd.merge(original_data, predictions_df, on='id', how='left')
    merged_df.to_csv('prediction_metrics/predicted_prices.csv', index=False)
    print("--> prediction_metrics/predicted_prices.csv saved!")