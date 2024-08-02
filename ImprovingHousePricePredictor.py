import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import datetime

class SelfImprovingHousePricePredictor:
    def __init__(self, data_file='house_data.csv', model_file='house_price_model.joblib'):
        self.data_file = data_file
        self.model_file = model_file
        self.model = None
        self.performance_history = []

    def load_data(self):
        if os.path.exists(self.data_file):
            return pd.read_csv(self.data_file)
        else:
            # Create synthetic data if file doesn't exist
            np.random.seed(42)
            n_samples = 1000
            size = np.random.randint(1000, 5000, n_samples)
            bedrooms = np.random.randint(1, 6, n_samples)
            age = np.random.randint(0, 50, n_samples)
            price = 100000 + 100 * size + 25000 * bedrooms - 1000 * age + np.random.normal(0, 50000, n_samples)
            
            data = pd.DataFrame({
                'size': size,
                'bedrooms': bedrooms,
                'age': age,
                'price': price
            })
            data.to_csv(self.data_file, index=False)
            return data

    def preprocess_data(self, data):
        X = data[['size', 'bedrooms', 'age']]
        y = data['price']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train_model(self, X_train, y_train):
        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

    def evaluate_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return mse, r2

    def save_model(self):
        joblib.dump(self.model, self.model_file)

    def load_model(self):
        if os.path.exists(self.model_file):
            self.model = joblib.load(self.model_file)
            return True
        return False

    def update_performance_history(self, mse, r2):
        self.performance_history.append({
            'timestamp': datetime.datetime.now(),
            'mse': mse,
            'r2': r2
        })

    def run(self):
        data = self.load_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(data)

        if not self.load_model():
            print("Training new model...")
            self.train_model(X_train, y_train)
        else:
            print("Loaded existing model.")

        mse, r2 = self.evaluate_model(X_test, y_test)
        print(f"Model Performance - MSE: {mse:.2f}, R2: {r2:.2f}")

        self.update_performance_history(mse, r2)

        # Self-improvement logic
        if len(self.performance_history) > 1:
            prev_mse = self.performance_history[-2]['mse']
            if mse < prev_mse:
                print("Model improved. Saving new model.")
                self.save_model()
            else:
                print("Model did not improve. Retraining with more data.")
                # Add some noise to existing data to simulate new data
                new_data = data.copy()
                new_data['price'] += np.random.normal(0, 1000, len(new_data))
                new_data.to_csv(self.data_file, index=False)
                X_train, X_test, y_train, y_test = self.preprocess_data(new_data)
                self.train_model(X_train, y_train)
                new_mse, new_r2 = self.evaluate_model(X_test, y_test)
                print(f"Retrained Model Performance - MSE: {new_mse:.2f}, R2: {new_r2:.2f}")
                self.update_performance_history(new_mse, new_r2)
                if new_mse < mse:
                    print("Retrained model improved. Saving new model.")
                    self.save_model()
        else:
            self.save_model()

    def predict(self, features):
        if self.model is None:
            self.load_model()
        return self.model.predict(features)

if __name__ == "__main__":
    predictor = SelfImprovingHousePricePredictor()
    for _ in range(5):  # Run the script multiple times to demonstrate self-improvement
        predictor.run()
        print("\n")

    # Make a prediction
    new_house = np.array([[2000, 3, 10]])  # size: 2000 sq ft, 3 bedrooms, 10 years old
    predicted_price = predictor.predict(new_house)
    print(f"Predicted price for new house: ${predicted_price[0]:,.2f}")5