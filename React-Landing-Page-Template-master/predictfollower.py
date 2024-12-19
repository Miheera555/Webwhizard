
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import streamlit as st

# Sample data (replace with your actual data)
data = {'posts': [10, 15, 20, 25, 30], 
        'likes': [100, 150, 200, 250, 300], 
        'followers': [1000, 1200, 1400, 1600, 1800]}
df = pd.DataFrame(data)

# Prepare the data
X = df[['posts', 'likes']]
y = df['followers']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)

# Predict followers for the next four months (replace with your actual data)
future_data = pd.DataFrame({'posts': [35, 40, 45, 50], 'likes': [350, 400, 450, 500]})
future_followers = model.predict(future_data)

# Streamlit webpage
st.title("Followers Prediction")
st.write(f"Mean Squared Error: {mse:.2f}")

# Display the predictions for future posts in columns
st.write("Predicted followers for future data:")

# Create three columns for displaying data
col1, col2, col3 = st.columns(3)

# Populate the columns
with col1:
    st.write("Posts")
    st.write(future_data['posts'])

with col2:
    st.write("Likes")
    st.write(future_data['likes'])

with col3:
    st.write("Predicted Followers")
    st.write(future_followers)

# Plot the data
st.write("Followers Growth Plot:")
fig, ax = plt.subplots()
ax.plot(df['posts'], df['followers'], label='Past Data')
ax.plot(future_data['posts'], future_followers, label='Predictions')
ax.set_xlabel('Posts')
ax.set_ylabel('Followers')
ax.legend()
st.pyplot(fig)

