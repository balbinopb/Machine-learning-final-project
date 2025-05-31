import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64).reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))
        self.bias = 0

        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.dot(X, self.weights) + self.bias


def encode_categorical(df, columns):
    for col in columns:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df.drop(col, axis=1), dummies], axis=1)
    return df

class SalaryPredictor:
    def __init__(self, data_path='salary_prediction_data.csv'):
        self.categorical_cols = ['Education', 'Location', 'Job_Title', 'Gender']
        self.data = pd.read_csv(data_path)
        
        # Encode whole data
        data_encoded = encode_categorical(self.data.copy(), self.categorical_cols)
        
        # Features and target
        self.X = data_encoded.drop('Salary', axis=1)
        self.y = data_encoded['Salary']

        # Fit scaler on features
        self.scaler_X = StandardScaler()
        X_scaled = self.scaler_X.fit_transform(self.X)

        # Train model
        self.model = LinearRegression(learning_rate=0.01, n_iterations=5000)
        self.model.fit(X_scaled, self.y.values.reshape(-1, 1))

    def predict_salary(self, input_dict):
        user_df = pd.DataFrame([input_dict])
        user_encoded = encode_categorical(user_df, self.categorical_cols)

        # Add missing columns with zeros
        for col in self.X.columns:
            if col not in user_encoded.columns:
                user_encoded[col] = 0

        # Reorder columns to match training data
        user_encoded = user_encoded[self.X.columns]

        # Scale input
        user_scaled = self.scaler_X.transform(user_encoded)

        # Predict
        pred = self.model.predict(user_scaled)
        return pred[0, 0]

    def get_unique_values(self):
        return {col: self.data[col].unique() for col in self.categorical_cols}
