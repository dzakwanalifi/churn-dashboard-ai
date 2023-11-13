# Importing necessary libraries for creating a Streamlit app
import streamlit as st
import pandas as pd
import pickle
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
import numpy as np
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.image as plt
import os
import plotly.figure_factory as ff
import plotly.express as px

os.environ['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']

# Load your trained CatBoost model
model_cb = pickle.load(open('Model/model_cb.pkl', 'rb'))

# load scaler model
scaler = pickle.load(open('Model/scaler.pkl', 'rb'))

#load kmeans model
km_model = pickle.load(open('Model/cluster.pkl', 'rb'))

st.set_page_config(page_title="CRAIDS",
                   page_icon="🔍",
                   layout="wide")

st.markdown("<h1 style='text-align: center;'>CRAIDS</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Customer Retention - AI Dynamics Dashboard</h3>", unsafe_allow_html=True)

# Create a horizontal menu with the 'option_menu' custom component
selected2 = option_menu(None, ["Churn Analysis", "Customer Analysis", "Prediction", "Model Evaluation"], 
    icons=['bar-chart-fill', 'people-fill', 'stars', 'gear-fill'], 
    menu_icon="cast", default_index=0, orientation="horizontal")

if selected2 == "Churn Analysis":
    # The HTML content to be embedded
    tableau_html = """
<!DOCTYPE html>
<html>
<head>
    <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
    <style>
        /* CSS untuk mengatur elemen di tengah */
        .center {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        /* CSS untuk membuat elemen responsif */
        #tableauViz {
            max-width: 100%; /* Maksimum lebar elemen */
            max-height: 100%; /* Maksimum tinggi elemen */
        }
    </style>
</head>
<body>
    <!-- Membungkus elemen dengan div yang memiliki kelas "center" -->
    <div class="center">
        <tableau-viz id="tableauViz"
        src='https://public.tableau.com/views/CustomerChurnDashboard_16998481467460/Dashboard1'
        device="desktop"
        toolbar="hidden" hide-tabs>
        </tableau-viz>
    </div>
</body>
</html>

    """

    with st.columns([1, 100, 1])[1]:
        html(tableau_html, height=670)
    
    st.header('Chatbot')
    
    st.session_state.openai_key = os.environ['OPENAI_API_KEY']
    st.session_state.prompt_history = []
    st.session_state.df = None

    if "openai_key" in st.session_state:
        if "df" not in st.session_state or st.session_state.df is None:
            st.session_state.df = pd.read_excel('Data/Telco_customer_churn_adapted_v2.xlsx')

        with st.form("Question"):
            question = st.text_input("Question", value="", type="default")
            submitted = st.form_submit_button("Submit")
            if submitted:
                with st.spinner():
                    llm = OpenAI(api_token=st.session_state.openai_key)
                    pandas_ai = PandasAI(llm)
                    x = pandas_ai.run(st.session_state.df, prompt=question)

                    if os.path.isfile('temp_chart.png'):
                        im = plt.imread('temp_chart.png')
                        st.image(im)
                        os.remove('temp_chart.png')

                    if x is not None:
                        st.write(x)
                    st.session_state.prompt_history.append(question)

    if st.button("Clear"):
        st.session_state.prompt_history = []
        st.session_state.df = None

elif selected2 == "Customer Analysis":
    # The HTML content to be embedded
    tableau_html = """
<!DOCTYPE html>
<html>
<head>
    <script type="module" src="https://public.tableau.com/javascripts/api/tableau.embedding.3.latest.min.js"></script>
    <style>
        /* CSS untuk mengatur elemen di tengah */
        .center {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }

        /* CSS untuk membuat elemen responsif */
        #tableauViz {
            max-width: 100%; /* Maksimum lebar elemen */
            max-height: 100%; /* Maksimum tinggi elemen */
        }
    </style>
</head>
<body>
    <!-- Membungkus elemen dengan div yang memiliki kelas "center" -->
    <div class="center">
        <tableau-viz id="tableauViz"
        src='https://public.tableau.com/views/DashboardCustomerOverview_16996880518420/Dashboard2'
        device="desktop"
        toolbar="hidden" hide-tabs>
        </tableau-viz>
    </div>
</body>
</html>

    """

    with st.columns([1, 100, 1])[1]:
        html(tableau_html, height=670)

    st.header('Chatbot')
    
    st.session_state.openai_key = os.environ['OPENAI_API_KEY']
    st.session_state.prompt_history = []
    st.session_state.df = None

    if "openai_key" in st.session_state:
        if "df" not in st.session_state or st.session_state.df is None:
            st.session_state.df = pd.read_excel('Data/Telco_customer_churn_adapted_v2.xlsx')

        with st.form("Question"):
            question = st.text_input("Question", value="", type="default")
            submitted = st.form_submit_button("Submit")
            if submitted:
                with st.spinner():
                    llm = OpenAI(api_token=st.session_state.openai_key)
                    pandas_ai = PandasAI(llm)
                    x = pandas_ai.run(st.session_state.df, prompt=question)

                    if os.path.isfile('temp_chart.png'):
                        im = plt.imread('temp_chart.png')
                        st.image(im)
                        os.remove('temp_chart.png')

                    if x is not None:
                        st.write(x)
                    st.session_state.prompt_history.append(question)

    if st.button("Clear"):
        st.session_state.prompt_history = []
        st.session_state.df = None

elif selected2 == "Prediction":
    # Tenure Months Prediction Page
    st.header("Tenure Months to Churn Prediction")
    
    df_sample = pd.read_excel("Data/Telco_customer_churn_adapted_v2.xlsx")

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
    # Read the uploaded file into a DataFrame
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                st.error('This file format is not supported. Please upload a .xlsx or .csv file.')
                st.stop()

            df_tenure = df['Tenure Months']

            # Check for 'Latitude' and 'Longitude' columns and drop if exists
            if 'Latitude' in df.columns and 'Longitude' in df.columns:
                df.drop(['Latitude','Longitude'], axis=1, inplace=True)
            
            # Rename columns if they exist
            column_renames = {
                'Monthly Purchase (Thou. IDR)': 'Monthly Purchase',
                'CLTV (Predicted Thou. IDR)': 'CLTV'
            }
            df.rename(columns={k: v for k, v in column_renames.items() if k in df.columns}, inplace=True)

            # ubah cltv dan monthly purchase dengan dikali 1000
            df['CLTV'] = df['CLTV']*1000
            df['Monthly Purchase'] = df['Monthly Purchase']*1000

            # Process the DataFrame as needed for your models
            df['Total Service'] = df[['Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp']].apply(
                lambda row: sum(x == 'Yes' for x in row), axis=1)
            
            df['CLTV per Purchase'] = df['CLTV'] / df['Monthly Purchase']
            
            # Scale the numeric features
            numeric_features = df[['Monthly Purchase', 'CLTV', 'Total Service']].values
            numeric_features_scaled = scaler.transform(numeric_features)
            
            # Predict the cluster
            clusters = km_model.predict(numeric_features_scaled)
            df['Cluster'] = clusters
            
            # Urutkan data sesuai permintaan
            df = df[['Location', 'Device Class', 'Games Product',
                      'Music Product', 'Education Product', 'Call Center', 'Video Product',
                      'Use MyApp', 'Payment Method', 'Monthly Purchase', 'CLTV',
                      'Total Service', 'CLTV per Purchase', 'Cluster']]

            # Predict using the CatBoost model
            predictions = model_cb.predict(df)
            
            # Add predictions to the DataFrame
            df['Predicted Tenure Months to Churn'] = predictions

            df['Predicted Tenure Months to Churn'] = df['Predicted Tenure Months to Churn'].apply(lambda x: round(x))
            
            df = pd.concat([df, df_tenure], axis=1)

            df_disp = df[['Predicted Tenure Months to Churn', 'Tenure Months', 'Location', 'Games Product', 'Music Product', 'Education Product',
                'Call Center', 'Video Product', 'Use MyApp', 'Device Class', 'Payment Method',
                'Monthly Purchase', 'CLTV', 'Total Service', 'CLTV per Purchase', 'Cluster']]

            st.header('Table of Predictions')
            # Display the DataFrame with predictions
            st.dataframe(df_disp)

            # histogram tenure months dan predicted tenure months to churn dengan px histogram
            fig11 = px.histogram(df, x=['Tenure Months', 'Predicted Tenure Months to Churn'],
                                title='Tenure Months vs Predicted Tenure Months to Churn')
            
            fig11.update_layout(legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            ))

            st.plotly_chart(fig11, use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")

    if uploaded_file is None:
        st.download_button("Download Sample Data", data=df_sample.to_csv(index=False), file_name='sample_data.csv')

        # Define the form and its fields
        with st.form(key='prediction_form'):
            # Create columns for the input fields
            col1, col2, col3 = st.columns(3)
            
            with col1:
                location = st.selectbox('Location', ['Jakarta', 'Bandung'])
                games_product = st.radio('Games Product', ['Yes', 'No', 'No internet service'])
                use_myapp = st.radio('Use MyApp', [ 'Yes', 'No', 'No internet service'])
            
            with col2:
                device_class = st.selectbox('Device Class', ['Mid End', 'High End', 'Low End'])
                music_product = st.radio('Music Product', ['Yes', 'No', 'No internet service'])
                education_product = st.radio('Education Product', ['Yes','No', 'No internet service'])
            
            with col3:
                payment_method = st.selectbox('Payment Method', ['Digital Wallet', 'Pulsa', 'Debit', 'Credit'])
                video_product = st.radio('Video Product', ['Yes', 'No','No internet service'])
                call_center = st.radio('Call Center', ['Yes','No'])
            
            # Create a single column for the numeric inputs below the radio buttons
            col4, col5 = st.columns(2)  # The third column is just to take up the remaining space
            with col4:
                monthly_purchase = st.number_input('Monthly Purchase', min_value=0, value=100000)
            
            with col5:
                cltv = st.number_input('CLTV', min_value=0, value=500000)
            
            # Place the submit button in the center below the columns
            submit_button = st.form_submit_button(label='Predict')

        if submit_button:
            # Create a dataframe from the inputs
            input_data = pd.DataFrame({
                'Location': [location],
                'Games Product': [games_product],
                'Music Product': [music_product],
                'Education Product': [education_product],
                'Call Center': [call_center],
                'Video Product': [video_product],
                'Use MyApp': [use_myapp],
                'Device Class': [device_class],
                'Payment Method': [payment_method],
                'Monthly Purchase': [monthly_purchase],
                'CLTV': [cltv]
            })
            
            input_data['Total Service'] = input_data[['Games Product', 'Music Product', 'Education Product', 'Call Center', 'Video Product', 'Use MyApp']].apply(
                lambda row: sum(x == 'Yes' for x in row), axis=1)
            
            input_data['CLTV per Purchase'] = input_data['CLTV'] / input_data['Monthly Purchase']
            
            # Standardize the numeric fields before passing them to the KMeans model
            numeric_features = input_data[['Monthly Purchase', 'CLTV', 'Total Service']].values
            numeric_features_scaled = scaler.transform(numeric_features)
            
            # Predict the cluster
            cluster = km_model.predict(numeric_features_scaled)
            input_data['Cluster'] = cluster
            
            # Prediction with CatBoost
            prediction = model_cb.predict(input_data)

            # Display the prediction
            st.metric(label='Predicted Tenure Months to Churn', value=f"{int(prediction)} Month")    
elif selected2 == "Model Evaluation":
    # load df_churn.csv as df
    df = pd.read_csv('Data/df_churn.csv')
    
    df.sort_values(by=['Tenure Months'], inplace=True, ascending=True)
    
    # rename tenure month names variables to tenure month customer
    df.rename(columns={'Tenure Months': 'Tenure Months Churn Customer'}, inplace=True)

    # create index after sort
    df['Index'] = np.arange(len(df))

    # show mae and r2 score and show using st.metric
    mae = round(np.mean(abs(df['Predicted Tenure Months to Churn'] - df['Tenure Months Churn Customer'])), 2)
    r2 = round(1 - (np.sum((df['Tenure Months Churn Customer'] - df['Predicted Tenure Months to Churn'])**2) / np.sum((df['Tenure Months Churn Customer'] - np.mean(df['Tenure Months Churn Customer']))**2)), 2)
    mse = round(np.mean((df['Tenure Months Churn Customer'] - df['Predicted Tenure Months to Churn'])**2), 2)

    st.header('Model Evaluation')

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label='MAE', value=mae)
    with col2:
        st.metric(label='MSE', value=mse)
    with col3:
        st.metric(label='R2', value=r2)
        
    # buat line chart 'Predicted Tenure Months to Churn', 'Tenure Months' dengan plotly
    fig1 = px.scatter(df,x='Index', y=['Predicted Tenure Months to Churn', 'Tenure Months Churn Customer'],
                        title='Predicted Tenure Months to Churn vs Tenure Months')

    fig1.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01
    ))

    # load feature importance
    feature_importance = pd.read_csv('Data/importance.csv')
    # buat bar chart feature importance dengan px horizontal bar dan sort value
    fig2 = px.bar(feature_importance, x='Importances', y='Feature Id',
                 orientation='h', title='Feature Importance')
    
    fig2.update_layout(yaxis=dict(autorange="reversed"))

    col1, col2 = st.columns([5, 3])
    with col1:
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        st.plotly_chart(fig2, use_container_width=True)