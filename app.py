from flask import Flask, send_file
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend for Matplotlib
import matplotlib.pyplot as plt
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/predict-followers')
def predict_followers():
    # Create dummy data
    data = pd.DataFrame({
        'Posts': [5 * i for i in range(1, 141)],
        'Likes': [100 * i for i in range(1, 141)],
        'Followers': [150 * i for i in range(1, 141)]
    })

    # Prepare data
    X = data[['Posts', 'Likes']]
    y = data['Followers']

    # Polynomial Features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_poly)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and visualize
    y_pred = model.predict(X_test)

    # Create 2x2 grid for graphs
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Graph: Actual vs Predicted Followers (Scatter Plot)
    axs[0, 0].scatter(y_test, y_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
    axs[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
    axs[0, 0].set_xlabel("Actual Followers")
    axs[0, 0].set_ylabel("Predicted Followers")
    axs[0, 0].set_title("Actual vs Predicted Followers")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # 2. Graph: Distribution of Actual Followers (Histogram)
    axs[0, 1].hist(y_test, bins=20, color='green', alpha=0.7)
    axs[0, 1].set_xlabel("Followers")
    axs[0, 1].set_ylabel("Frequency")
    axs[0, 1].set_title("Distribution of Actual Followers")
    axs[0, 1].grid(True)

    # 3. Graph: Residual Plot (Prediction Error)
    error = y_test - y_pred
    axs[1, 0].scatter(y_test, error, color='purple', alpha=0.7)
    axs[1, 0].axhline(0, color='black', linestyle='--')
    axs[1, 0].set_xlabel("Actual Followers")
    axs[1, 0].set_ylabel("Prediction Error")
    axs[1, 0].set_title("Prediction Error vs Actual Followers")
    axs[1, 0].grid(True)

    # 4. Graph: Polynomial Regression Curve
    # Create a range of values for the feature
    x_range = np.linspace(X_test[:, 1].min(), X_test[:, 1].max(), 100).reshape(-1, 1)  # Using the 'Likes' feature
    x_range_poly = poly.transform(np.column_stack([np.zeros_like(x_range), x_range]))  # Polynomial features for 'Likes'
    x_range_scaled = scaler.transform(x_range_poly)  # Scale the new feature range
    y_range_pred = model.predict(x_range_scaled)

    axs[1, 1].plot(x_range, y_range_pred, color='orange', linewidth=2, label="Polynomial Fit")
    axs[1, 1].scatter(X_test[:, 1], y_pred, color='orange', alpha=0.7, label='Predicted Followers')
    axs[1, 1].set_xlabel("Likes")
    axs[1, 1].set_ylabel("Predicted Followers")
    axs[1, 1].set_title("Polynomial Regression Fit")
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the plot to an in-memory file
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()

    # Serve the image
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
