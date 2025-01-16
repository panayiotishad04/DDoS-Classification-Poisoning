import os
import pickle

import numpy as np
import pandas as pd
import shap
import streamlit as st
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
# import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_tree
import plotly.express as px

# Sidebar for view selection
st.set_page_config(layout="wide")
view = st.sidebar.radio("Select View", ['Normal NIDS', 'Malicious Alerts', 'Testing'])

if view == 'Normal NIDS':
    st.title("Live NIDS")
    st.write("The flows are read live as soon as they are being captured")
    # Directory input for Normal NIDS
    monitored_dir = st.text_input("Enter the directory to monitor:",
                                  "nfcapd_dir")  # Default is the current directory

    # Check if the directory exists
    if os.path.exists(monitored_dir):
        st.success(f"Monitoring directory: {monitored_dir}")

        # Get the list of files
        files = [f for f in os.listdir(monitored_dir) if os.path.isfile(os.path.join(monitored_dir, f))]

        if files:
            # Select a file to view
            selected_file = st.selectbox("Select a file to view:", files)

            # Display the selected file
            if selected_file:
                filepath = os.path.join(monitored_dir, selected_file)

    # This is the normal flow where we process the file, make predictions, and display the table
    try:
        # Attempt to read the file as a pandas DataFrame
        df = pd.read_csv(filepath)
        st.write(f"Displaying content of {selected_file}:")
        st.dataframe(df, use_container_width=True)

        # Process the flow data
        selected_features = ['sp', 'dp', 'pr', 'td', 'flg', 'ibyt', 'ipkt', 'opkt', 'obyt']
        filtered_flows = df[selected_features]

        filtered_flows = filtered_flows.rename(columns={
            'sp': 'id.orig_port',
            'dp': 'id.resp_pport',
            'pr': 'proto_enum',
            'td': 'duration_interval',
            'flg': 'conn_state_string',
            'ibyt': 'orig_pkts_count',
            'ipkt': 'orig_ip_bytes_count',
            'opkt': 'resp_pkts_count',
            'obyt': 'resp_bytes'
        })

        # Encoding categorical columns
        categorical_col = ['proto_enum', 'conn_state_string']
        mapping = {}
        for col in categorical_col:
            le = LabelEncoder()
            filtered_flows[col] = le.fit_transform(filtered_flows[col])
            mapping[col] = le

        # Convert all columns to float and then to integer
        for col in filtered_flows.columns:
            if filtered_flows[col].dtype == 'object':  # Check if the column contains strings
                filtered_flows[col] = filtered_flows[col].astype(float)

        # Convert all numeric columns to integers
        filtered_flows = filtered_flows.astype(int)

        # Load the model and make predictions
        # model = tf.keras.models.load_model('nn.h5')
        with open('random_forest_model.pkl', 'rb') as file:
            loaded_model = pickle.load(file)
        model = loaded_model
        predictions = model.predict(filtered_flows)

        malicious_flows = df[predictions > 0.0]
        malicious_file_path = 'malicious_flows.csv'

        # Write malicious flows to a CSV file
        malicious_flows.to_csv(malicious_file_path, index=False)
        # Add predictions to the DataFrame
        df['Prediction'] = (predictions >= 0.0).astype(int)


    except Exception as e:
        st.error(f"Could not read {selected_file} as a DataFrame: {e}")

elif view == 'Malicious Alerts':
    st.title("Malicious Alerts")
    st.write("This view contains only the flows marked as potentially malicious DDoS")
    malicious_filepath = 'malicious_flows.csv'

    try:
        # Attempt to read the malicious alerts file as a pandas DataFrame
        df_malicious = pd.read_csv(malicious_filepath)
        st.write(f"Displaying preprocessed malicious alerts from {malicious_filepath}:")
        st.dataframe(df_malicious)

        if 'pr' in df_malicious.columns:
            protocol_counts = df_malicious['pr'].value_counts(normalize=True) * 100
            protocol_df = protocol_counts.reset_index()
            protocol_df.columns = ['Protocol', 'Percentage']

            # Create the pie chart
            fig = px.pie(protocol_df, names='Protocol', values='Percentage', title='Protocol Distribution')
            st.plotly_chart(fig)

        if 'ibyt' in df_malicious.columns:
            # Create a histogram for byte values
            fig = px.histogram(df_malicious, x='ibyt', nbins=30, title='Histogram of Byte Values (ibyt)',
                               labels={'ibyt': 'Byte Values'},
                               template='plotly_white')
            fig.update_layout(xaxis_title='Byte Values', yaxis_title='Count')
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Could not read {malicious_filepath} as a DataFrame: {e}")

    else:
        st.warning("No files found in the directory.")

elif view == 'Testing':
    st.title("Adversarial Testing")
    st.write("Statistics regarding adversarial testing")
    malicious_filepath = 'df_ben_ddos_shorter.csv'

    # if st.button('Start Attacking'):

    try:
        # Attempt to read the malicious alerts file as a pandas DataFrame
        df_malicious = pd.read_csv(malicious_filepath)
        st.write(f"Displaying preprocessed malicious alerts from {malicious_filepath}:")
        st.dataframe(df_malicious)
        score = [0, 0, 1, 1, 1, 1]
        st.write(f"Adversarial score: {np.average(score)}")

        # most_common_prot = df_malicious['pr'].max
        # st.write(f"Most common protocol: {most_common_prot}")

        # Example data
        df_sampled = pd.read_csv('df_ben_ddos_shorter.csv')
        X = df_sampled.drop(columns=['Category', 'id.orig_addr', 'id.resp_haddr'])
        # feature_names = ['sp', 'dp', 'pr', 'td', 'flg', 'ibyt', 'ipkt', 'opkt', 'obyt']
        y = df_sampled['Category']
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X, y = make_classification(n_samples=10000, n_features=10, n_classes=2, random_state=42)

        feature_names = ['id.orig_port', 'id.resp_pport', 'proto_enum',
                         'duration_interval', 'conn_state_string', 'orig_pkts_count', 'orig_ip_bytes_count',
                         'resp_pkts_count', 'resp_bytes']

        # Train an XGBoost model
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        model = xgb.train({
            'max_depth': 20,
            'eta': 0.1,
            'objective': 'reg:squarederror'
        }, dtrain, 30)

        # Create a SHAP explainer for the XGBoost model
        explainer = shap.TreeExplainer(model)

        # Get SHAP values for the test set
        shap_values = explainer.shap_values(X_test)

        if st.button("Show SHAP Summary Plot"):
            st.subheader("SHAP Summary Plot (Overall Feature Contribution)")
            st.write("""
            The SHAP summary plot shows how each feature contributes to the model's predictions across all samples.
            Positive SHAP values indicate that the feature pushes the prediction towards the positive class (e.g., malicious),
            and negative SHAP values indicate the opposite (e.g., benign).
            """)
            fig = plt.figure()
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            st.pyplot(fig)

        if st.button("Explainer LIME"):
            X_train = pd.DataFrame(X_train, columns=feature_names)
            class_names = ['0', '1']
            explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names,
                                             mode='classification')

        if st.button("Show Decision Tree"):
            fig, ax = plt.subplots(figsize=(10, 10))
            plot_tree(model, num_trees=0, ax=ax)
            st.pyplot(fig)

        if st.button("Compare Hist"):
            df = pd.read_csv("flows_2.csv")
            # Create the histogram
            fig = px.histogram(
                df,
                x="column",  # Column to plot
                title="Feature changed to make the flow benign",
                labels={"column": "Category"},  # Label for x-axis
                text_auto=True  # Show counts on bars
            )

            # Update layout to adjust bar spacing, tilt labels, and set colors
            fig.update_traces(
                marker=dict(color="darkorange", line=dict(width=0)),  # Set bar color to orange
                textfont=dict(color="blue")  # Set text color to blue
            )

            fig.update_xaxes(tickangle=45, linecolor="blue")  # Tilt labels
            fig.update_yaxes(linecolor="blue")  # Adjust y-axis line color
            # Remove background color
            fig.update_layout(
                xaxis_title="Feature changed",
                yaxis_title="Times applied",
                bargap=0.2,  # Ensure bars are joined
                plot_bgcolor="white",  # Background color of the plot area
                paper_bgcolor="white",  # Background color of the entire figure
            )

            st.plotly_chart(fig)  # Correct method to display Plotly figure in Streamlit


    except Exception as e:
        st.error(f"Could not read {malicious_filepath} as a DataFrame: {e.pr()}")

else:
    st.error("The specified directory does not exist.")
