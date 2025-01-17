import ipaddress
import os
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import shap
import streamlit as st
import tensorflow as tf
import xgboost as xgb
from lime.lime_tabular import LimeTabularExplainer
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import plot_tree

# Sidebar for view selection
st.set_page_config(layout="wide")
view = st.sidebar.radio("Select View", ['Normal NIDS', 'Malicious Alerts', 'Thunder NIDS Analyzer'])

if view == 'Normal NIDS':
    st.title("Live NIDS")
    st.write("The flows are read live as soon as they are being captured")

    model_options = ['Random Forest', 'Neural Network', 'GNN']
    selected_model = st.selectbox("Select model: ", model_options)
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
        filtered_flows = filtered_flows[['id.orig_port',
                                         'id.resp_pport',
                                         'proto_enum',
                                         'duration_interval',
                                         'conn_state_string',
                                         'orig_pkts_count',
                                         'orig_ip_bytes_count',
                                         'resp_pkts_count',
                                         'resp_bytes']]
        print(filtered_flows)

        if selected_model == 'Random Forest':
            with open('random_forest_model.pkl', 'rb') as file:
                model = pickle.load(file)
        elif selected_model == 'Neural Network':
            model = tf.keras.models.load_model('nn.h5')
            print(model.summary())
        print(filtered_flows.shape)
        predictions = model.predict(filtered_flows)

        df.to_csv('malicious_flows.csv', index=False)
        df['Prediction'] = predictions


    except Exception as e:
        st.error(f"Could not read {selected_file} as a DataFrame: {e.p()}")

elif view == 'Malicious Alerts':
    st.title("Malicious Alerts")
    st.write("This view contains only the flows marked as potentially malicious DDoS")
    malicious_filepath = 'malicious_flows.csv'

    try:
        # Attempt to read the malicious alerts file as a pandas DataFrame
        df_malicious = pd.read_csv(malicious_filepath)
        st.write(f"Total alerts: {df_malicious.shape[0]}")
        # st.write(f"Displaying preprocessed malicious alerts from {malicious_filepath}:")
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

    # else:
    #     st.warning("No files found in the directory.")

elif view == 'Thunder NIDS Analyzer':
    st.title("Adversarial Testing")
    st.write("Statistics regarding adversarial testing")
    malicious_filepath = 'df_ben_ddos_shorter.csv'

    # Initialize session states for buttons
    if 'start_attack' not in st.session_state:
        st.session_state.start_attack = False
    if 'compare_hist' not in st.session_state:
        st.session_state.compare_hist = False
    if 'show_shap' not in st.session_state:
        st.session_state.show_shap = False
    if 'show_lime' not in st.session_state:
        st.session_state.show_lime = False
    if 'show_tree' not in st.session_state:
        st.session_state.show_tree = False

    # Callback for "Start Adversarial Attack" button
    def start_attack_callback():
        st.session_state.start_attack = True

    # Display the "Start Adversarial Attack" button
    if not st.session_state.start_attack:
        st.button('Start Adversarial Attack', on_click=start_attack_callback)
    else:
        try:
            # Load and display the adversarial data
            df_malicious = pd.read_csv(malicious_filepath)
            st.write(f"Displaying malicious flows which managed to trick the classifier and were labeled as normal")
            df_sampled = pd.read_csv('df_ben_ddos_shorter.csv')
            df_copy = pd.read_csv('df_ben_ddos_shorter.csv')
            df_copy['id.orig_addr'] = df_copy['id.orig_addr'].map(lambda ip: str(ipaddress.IPv4Address(ip)))
            df_copy['id.orig_port'] = df_copy['id.orig_port'].map(lambda ip: str(ip))
            df_copy['id.resp_haddr'] = df_copy['id.resp_haddr'].map(lambda ip: str(ipaddress.IPv4Address(ip)))
            st.dataframe(df_copy)

            X = df_sampled.drop(columns=['Category', 'id.orig_addr', 'id.resp_haddr'])
            y = df_sampled['Category']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

            # Save the model and test data in session state for subsequent use
            st.session_state['xgboost_model'] = model
            st.session_state['X_test'] = X_test
            st.session_state['feature_names'] = feature_names

        except Exception as e:
            st.error(f"Error during attack simulation: {e}")

        # Display the rest of the buttons after the attack is started
        def compare_hist_callback():
            st.session_state.compare_hist = True

        def show_shap_callback():
            st.session_state.show_shap = True

        def show_lime_callback():
            st.session_state.show_lime = True

        def show_tree_callback():
            st.session_state.show_tree = True

        # st.button('Compare Hist', on_click=compare_hist_callback)
        # st.button('Show SHAP Summary Plot', on_click=show_shap_callback)
        # st.button('Explainer LIME', on_click=show_lime_callback)
        # st.button('Show Decision Tree', on_click=show_tree_callback)

        # Actions for the buttons
        # if st.session_state.compare_hist:
        try:
            st.subheader("Feature Manipulation to Evade Detection")
            df = pd.read_csv("flows_2.csv")
            fig = px.histogram(
                df,
                x="column",  # Column to plot
                title="Feature changes to make flows benign",
                labels={"column": "Category"},
                text_auto=True
            )
            fig.update_traces(
                marker=dict(color="darkorange", line=dict(width=0)),
                textfont=dict(color="blue")
            )
            fig.update_xaxes(tickangle=45, linecolor="blue")
            fig.update_yaxes(linecolor="blue")
            fig.update_layout(
                xaxis_title="Feature changed",
                yaxis_title="Times applied",
                bargap=0.2,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"Error generating histogram: {e}")

    # if st.session_state.show_shap:
        try:
            model = st.session_state.get('xgboost_model')
            X_test = st.session_state.get('X_test')
            feature_names = st.session_state.get('feature_names')

            if model and X_test is not None:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                st.subheader("SHAP Summary Plot (Overall Feature Contribution)")
                fig = plt.figure()
                shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
                st.pyplot(fig)
            else:
                st.error("Model or test data not available for SHAP plot.")
        except Exception as e:
            st.error(f"Error generating SHAP plot: {e}")

    # if st.session_state.show_lime:
        try:
            st.subheader("Flow Specific Label Probabilities using LIME")
            X_train = pd.DataFrame(X_train, columns=st.session_state.feature_names)
            model_2 = xgb.XGBClassifier()
            model_2.fit(X_train, y_train)
            class_names = ['0', '1']
            explainer = LimeTabularExplainer(X_train.values, feature_names=feature_names, class_names=class_names,
                                             mode='classification')

            sample_idx = st.slider("Select a Test Instance", 1, len(X_train) - 1, 1)
            if sample_idx == 0:
                st.dataframe(X_train[0])
            else:
                st.dataframe(X_train[sample_idx - 1:sample_idx])
            instance = X_train.iloc[sample_idx]
            explanation = explainer.explain_instance(
                data_row=instance.values,
                predict_fn=model_2.predict_proba
            )
            fig = explanation.as_pyplot_figure()
            st.write("Prediction Probabilities:", model_2.predict_proba([instance.values])[0])
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error generating LIME explanation: {e}")

    # if st.session_state.show_tree:
        try:
            st.subheader("Decision branches of Random Forest")
            model = st.session_state.get('xgboost_model')
            if model:
                fig, ax = plt.subplots(figsize=(10, 10))
                plot_tree(model, num_trees=0, ax=ax)
                st.pyplot(fig)
            else:
                st.error("Model not available for decision tree plot.")
        except Exception as e:
            st.error(f"Error generating decision tree: {e}")

else:
    st.error("The specified directory does not exist.")
