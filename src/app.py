import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Real Estate Price Prediction Dashboard")

rf = joblib.load("model/random_forest_price_model.pkl")
df = pd.read_csv("data/home_listings.csv")

st.write("Rows in dataset:", df.shape[0])

st.header("Predict Home Sale Price")

beds = st.number_input("Beds", min_value=0, max_value=20, value=3, step=1)
bath = st.number_input("Bath", min_value=0.0, max_value=20.0, value=2.0, step=0.5)
sqft = st.number_input("Sq Ft", min_value=0, value=1800, step=50)
lot_size = st.number_input("Lot Size", min_value=0, value=6000, step=100)
year_built = st.number_input("Year Built", min_value=1800, max_value=2100, value=1995, step=1)
listing_price = st.number_input("Listing Price", min_value=0.0, value=650000.0, step=1000.0)
pool = st.selectbox("Pool", [0, 1], index=0)
zip_code = st.number_input("Zip", min_value=0, max_value=99999, value=33000, step=1)

if st.button("Predict"):
    X = pd.DataFrame([{
        "Beds": beds,
        "Bath": bath,
        "Sq Ft": sqft,
        "Lot Size": lot_size,
        "Year Built": year_built,
        "Listing Price": listing_price,
        "Pool": pool,
        "Zip": zip_code
    }])

    pred = rf.predict(X)[0]
    st.success(f"Predicted Sale Price: ${pred:,.0f}")

st.header("Market Insights")

# Visualization 1: Sale Price Distribution
st.subheader("Sale Price Distribution")
fig1, ax1 = plt.subplots()
sns.histplot(df["Sale Price"], bins=30, kde=True, ax=ax1)
ax1.set_xlabel("Sale Price")
ax1.set_ylabel("Count")
st.pyplot(fig1)

# Visualization 2: Square Footage vs Sale Price
st.subheader("Square Footage vs Sale Price")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=df["Sq Ft"], y=df["Sale Price"], ax=ax2)
ax2.set_xlabel("Square Footage")
ax2.set_ylabel("Sale Price")
st.pyplot(fig2)

# Visualization 3: Feature Importance
st.subheader("Feature Importance (Random Forest)")
importances = rf.feature_importances_
features = ["Beds", "Bath", "Sq Ft", "Lot Size", "Year Built", "Listing Price", "Pool", "Zip"]

fig3, ax3 = plt.subplots()
sns.barplot(x=importances, y=features, ax=ax3)
ax3.set_xlabel("Importance")
ax3.set_ylabel("Feature")
st.pyplot(fig3)