import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# File to preprocess the given data

def preprocess_data(file_path):
    # Load data
    df = pd.read_csv(file_path)
    
    # Input missing values for numeric columns
    imputer_num = SimpleImputer(strategy='median')
    df[['bathrooms', 'bedrooms', 'beds']] = imputer_num.fit_transform(df[['bathrooms', 'bedrooms', 'beds']])
    
    # Categorical columns
    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city']] = imputer_cat.fit_transform(df[['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city']])
    
    # Convert booleans to binary and handle NaNs
    boolean_columns = ['host_has_profile_pic', 'host_identity_verified', 'instant_bookable']
    for col in boolean_columns:
        df[col] = df[col].map({'t': 1, 'f': 0})
        df[col] = df[col].fillna(df[col].mode()[0])

    # Convert host_since to number of days
    df['host_since'] = pd.to_datetime(df['host_since'])
    df['host_duration'] = (pd.to_datetime('today') - df['host_since']).dt.days
    df.drop(['host_since'], axis=1, inplace=True)
    
    # One-hot encode categorical variables
    encoder = OneHotEncoder(drop='first', sparse_output=False)
    encoded_features = encoder.fit_transform(df[['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city']])
    encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city']))
    df = pd.concat([df, encoded_features_df], axis=1)
    df.drop(['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city'], axis=1, inplace=True)

    # Remove specified feature columns if they exist
    columns_to_remove = ['property_type_Casa particular', 'property_type_Hut', 'property_type_Island', 'property_type_Lighthouse', 'property_type_Parking Space', 'cancellation_policy_long_term']
    df.drop(columns=[col for col in columns_to_remove if col in df.columns], axis=1, inplace=True)

    # Scaling numerical features
    scaler = StandardScaler()
    numeric_features = ['latitude', 'longitude', 'number_of_reviews', 'accommodates', 'bathrooms', 'bedrooms', 'beds']
    df[numeric_features] = scaler.fit_transform(df[numeric_features])

    # Drop unused columns
    unused_columns = ['amenities', 'name', 'thumbnail_url', 'description', 'first_review', 'last_review', 'host_duration', 'neighbourhood', 'zipcode', 'cleaning_fee', 'host_response_rate', 'review_scores_rating']
    df.drop(unused_columns, axis=1, inplace=True)

    # Replace whitespaces with underscores
    df.columns = [col.replace(' ', '_') for col in df.columns]

    # Check for any remaining NaN values
    # nan_counts = df.isnull().sum()
    # print("Columns with NaN values and their count:")
    # print(nan_counts[nan_counts > 0])

    return df

# Main
if __name__ == "__main__":
    df = preprocess_data('data/train.csv')
    #print(df.head())
    df.to_csv('prep_data/prep_train_data.csv', index=False)
    print("--> prep_data/prep_train_data.csv saved!")

    df = preprocess_data('data/test.csv')
    #print(df.head())
    df.to_csv('prep_data/prep_test_data.csv', index=False)
    print("--> prep_data/prep_test_data.csv saved!")

