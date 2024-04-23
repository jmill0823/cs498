import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math
import joblib
import matplotlib.pyplot as plt

# File to train the model for predicting the airbnb prices

# Predicting and converting log_price to price
def predict_and_convert(model, X):
    log_price_pred = model.predict(X)
    price_pred = np.exp(log_price_pred)  # Convert log_price back to price
    return price_pred

# Evaluate the model
def evaluate_model(model, X, y, dataset_name):
    log_price_pred = model.predict(X)
    price_pred = np.exp(log_price_pred)
    price_actual = np.exp(y)
    mae = mean_absolute_error(price_actual, price_pred)
    mse = mean_squared_error(price_actual, price_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(price_actual, price_pred)
    metrics = f'{dataset_name} - MAE: {mae}\n{dataset_name} - RMSE: {rmse}\n{dataset_name} - R2 Score: {r2}\n'
    # Writing metrics to file
    with open('training_metrics/model_metrics.txt', 'a') as file:
        file.write(metrics)

# Plot the predictions
def plot_predictions(model, X, y_actual_log, filename):
    # Predict log prices
    y_pred_log = model.predict(X)
    # Convert predicted and actual log prices back to regular prices
    y_pred_price = np.exp(y_pred_log)
    y_actual_price = np.exp(y_actual_log)

    # Creating subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot for log prices
    ax[0].scatter(y_actual_log, y_pred_log, alpha=0.5, color='blue')
    ax[0].plot([0, 8], [0, 8], 'k--', lw=4)
    ax[0].set_xlabel('Actual Log Price')
    ax[0].set_ylabel('Predicted Log Price')
    ax[0].set_title('Actual vs. Predicted Log Price')

    # Plot for actual prices
    ax[1].scatter(y_actual_price, y_pred_price, alpha=0.5, color='red')
    ax[1].plot([0, 800], [0, 800], 'k--', lw=4)
    ax[1].set_xlabel('Actual Price')
    ax[1].set_ylabel('Predicted Price')
    ax[1].set_title('Actual vs. Predicted Price')

    plt.tight_layout()  # Adjust layout to not overlap
    plt.savefig(filename)  # Save the figure to a file
    plt.close()  # Close the plot to free up memory

    # Price histogram for the predicted prices
    plt.figure(figsize=(10, 5))
    plt.hist(y_pred_price, bins=50, color='skyblue', alpha=0.7)
    plt.title(f'Distribution of Predicted Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("training_metrics/distribution_plot_pred.png")
    plt.close()

    # Price histogram for the actual prices
    plt.figure(figsize=(10, 5))
    plt.hist(y_actual_price, bins=50, color='skyblue', alpha=0.7)
    plt.title(f'Distribution of Actual Prices')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig("training_metrics/distribution_plot_actual.png")
    plt.close()

# Main
if __name__ == "__main__":

    # Load and preprocess data
    df = pd.read_csv('prep_data/prep_train_data.csv')
    df = df.drop(columns=['id'])

    # Calculate the 1st and 99th percentiles
    upper_bound = df['log_price'].quantile(0.95)
    lower_bound = df['log_price'].quantile(0.05)

    # Trim values below and above the 1st and 99th percentiles
    df = df = df[(df['log_price'] > lower_bound) & (df['log_price'] < upper_bound)]
    X = df.drop('log_price', axis=1)
    y = df['log_price']

    # Split the data into train, test, and validation sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Individual models
    lgb_model = lgb.LGBMRegressor(num_leaves=40, n_estimators=300, max_depth=20, learning_rate=0.05, random_state=256, n_jobs=-1, force_col_wise=True)
    ridge_model = Ridge(alpha=0.1)

    # Stacking ensemble
    best_model = StackingRegressor(
        estimators=[('lgbm', lgb_model), ('ridge', ridge_model)],
        final_estimator=Ridge(alpha=0.1),
        cv=5
    )

    # Training the stacked model
    best_model.fit(X_train, y_train)

    # Clearing the file at the start of the program
    with open('training_metrics/model_metrics.txt', 'w') as file:
        pass

    # Evaluating and plotting the test and validation sets
    evaluate_model(best_model, X_test, y_test, "Test set")  # Evaluate test metrics
    evaluate_model(best_model, X_val, y_val, "Validation set")  # Evaluate validation metrics
    plot_predictions(best_model, X_test, y_test, 'training_metrics/test_plot.png')
    plot_predictions(best_model, X_val, y_val, 'training_metrics/validation_plot.png')

    # Save the best model
    joblib.dump(best_model, 'price_prediction_model.pkl')