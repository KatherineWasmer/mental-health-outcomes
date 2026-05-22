import streamlit as st
import pandas as pd
import joblib
import warnings

warnings.filterwarnings("ignore")

@st.cache_resource
def load_model():
    return joblib.load('./mental-health-pipeline.pkl')

def make_prediction(input_data):
    model = load_model()

    df = pd.DataFrame([input_data], columns=[
        'happiness',
        'gdp_per_capita',
        'social_support',
        'life_expectancy',
        'freedom',
        'generosity',
        'corruption'
    ])

    result = model.predict(df)

    return result[0]

st.title("Mental Health Outcomes from World Happiness Metrics")

st.write("""
This app predicts the mental health prevalence in a country
given a set of metrics from the world happiness index. Data 
were trained on an SVM-RBF (support vector machine, radial basis function) 
model. For more information on SVM-RBF, please refer to the documentation 
here:
https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html

The prevalence levels are:
- Low Prevalence (below the 25th percentile)
- Below Average Prevalence (25th - 50th percentile)
- Above Average Prevalence (50th - 75th percentile)
- High Prevalence (above the 75th percentile)
         
These percentiles were derived from the combined prevalences of addiction, alcoholism, 
anxiety, bipolar disorder, depression, eating disorders, and schizophrenia.  
         
Information on the world happiness index can be found here:
https://www.kaggle.com/datasets/ajaypalsinghlo/world-happiness-report-2021?select=world-happiness-report.csv
         
This project was created by Kate Wasmer and Humaira Nasir for our SI 618 class, 
taught by Dr. Chris Teplovs. 
""")

form = st.form(key='my_form')

happiness = form.number_input(
    "Input the country's happiness score on a scale from 0 to 10",
    min_value=0.0,
    max_value=10.0,
    step=0.1
)

log_gdp = form.number_input(
    "Input the country's log GDP per capita",
    min_value=6.0,
    max_value=12.0,
    step=0.1
)

social_support = form.number_input(
    "Input the rate of perceived social support from 0 - 100%",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)

life_expectancy = form.number_input(
    "Input the country's life expectancy",
    min_value=0.0,
    max_value=120.0,
    step=1.0
)

freedom = form.number_input(
    "Input the level of freedom from 0 - 100%",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)

generosity = form.number_input(
    "Input the level of perceived generosity, where -1 is the least generous and \
    1 is the most generous",
    min_value= -1.0,
    max_value= 1.0, 
    step=0.01
)

corruption = form.number_input(
    "Input the level of perceived government corruption from 0 - 100%",
    min_value=0.0,
    max_value=1.0,
    step=0.01
)

submit = form.form_submit_button('Make Prediction')

if submit:

    input_data = [
        happiness,
        log_gdp,
        social_support,
        life_expectancy,
        freedom,
        generosity,
        corruption
    ]

    result = make_prediction(input_data)

    st.header('Predicted Category')
    st.markdown(
        f"""
        <p style='font-size:30px; color:green; font-weight:bold;'>
            {result}
        </p>
        """,
        unsafe_allow_html=True
    )