import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression   
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#page config
st.set_page_config("Multiple Linear Regression", layout="centered")   

#load css
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
load_css("style.css")

#title
st.markdown("""  
            <div class="card">
            
            <h1>Multiple Linear Regression </h1>
            <p>Predict <b> Tip Amount </b> from <b> Total Bill </b> and <b> Size </b> using Multiple Linear Regression.</p>
            
            </div>
            """,unsafe_allow_html=True)

#load data
@st.cache_data
def load_data():
    return sns.load_dataset("tips")
df=load_data()

#dataset preview

st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>',unsafe_allow_html=True)



#prepare data
x,y = df[["total_bill","size"]], df["tip"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scalar=StandardScaler()
x_train_scaled=scalar.fit_transform(x_train)
x_test_scaled=scalar.transform(x_test)

#train model
model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_pred=model.predict(x_test_scaled)

#model evaluation(metrics)
mae=mean_absolute_error(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
rmse=np.sqrt(mse)
r2=r2_score(y_test,y_pred)
adjusted_r2=1-(1-r2)*(len(y_test)-1)/(len(y_test)-x.shape[1]-1)


# display metrics (visualization)

st.subheader("Total Bill vs Tip (with multiple regression line)")

fig, ax = plt.subplots()
ax.scatter(df['total_bill'], df['tip'], color='blue', label='Actual Data')
ax.plot(
    df["total_bill"],
    model.predict(scalar.transform(x)),
    color='red',
    label='Multiple Regression Line'
)

ax.set_xlabel('Total Bill ($)')
ax.set_ylabel('Tip Amount ($)')
ax.legend()

st.pyplot(fig) 



#performance metrics
st.markdown('<div class="card">',unsafe_allow_html=True)
st.subheader("Model Performance Metrics")
c1,c2=st.columns(2)
c1.metric("Mean Absolute Error (MAE)",f"{mae:.2f}")
c2.metric("Mean Squared Error (MSE)",f"{mse:.2f}")
c3,c4=st.columns(2)
c3.metric("Root Mean Squared Error (RMSE)",f"{rmse:.2f}")
c4.metric("R-squared (R²)",f"{r2:.2f}")
st.metric("Adjusted R-squared",f"{adjusted_r2:.2f}")


#m & c 
st.markdown(f""" 
            <div class="card">
            <h3>Model Interception</h3>
            <p><b>Intercept :</b> {model.intercept_:.2f}</p>
            <p><b>co-efficient(Total bill):</b> {model.coef_[0]:.2f}</p>
            <p><b>co-efficient(Group Size):</b> {model.coef_[1]:.2f}</p>
            <p><b>Equation:</b> y = {model.coef_[0]:.2f}x + {model.coef_[1]:.2f}z + {model.intercept_:.2f}</p>
            </div>
            """,unsafe_allow_html=True)



# prediction

st.subheader("Predict Tip Amount")
total_bill = st.slider(
    "Enter Total Bill Amount ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)
size=st.slider("Group size",
    int(df["size"].min()),
    int(df["size"].max()),2)
input_scaled=scalar.transform([[total_bill,size]])

tip = model.predict(input_scaled)[0]

st.markdown(
    f"""
    <div class="prediction-box">
        <h3>
            Predicted Tip Amount for Total Bill of ${total_bill:.2f}
            is <b>${tip:.2f}</b>
        </h3>
    </div>
    """,
    unsafe_allow_html=True
)





