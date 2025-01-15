import streamlit as st
import pandas as pd
import os
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Streamlit UI

# Sidebar for view selection
st.set_page_config(layout="wide")
view = st.sidebar.radio("Select View", ['Normal NIDS', 'Malicious Alerts'])

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
        model = tf.keras.models.load_model('nn.h5')
        predictions = model.predict(filtered_flows)

        malicious_flows = df[predictions > 0.0]
        malicious_file_path = 'malicious_flows.csv'

        # Write malicious flows to a CSV file
        malicious_flows.to_csv(malicious_file_path, index=False)
        # Add predictions to the DataFrame
        df['Prediction'] = (predictions >= 0.0).astype(int)


        # Sidebar radio button for tab selection
        # def highlight_malicious(row):
        #     """Highlight rows with Prediction = 1 (Malicious)."""
        #     return ['background-color: red' if row['Prediction'] == 1 else '' for _ in row]
        #
        #
        # # Filter based on selected mode
        # st.subheader(f"Normal NIDS Flows - {selected_file}")
        # styled_df = df.style.apply(highlight_malicious, axis=1)
        # st.dataframe(styled_df)

    except Exception as e:
        st.error(f"Could not read {selected_file} as a DataFrame: {e}")

elif view == 'Malicious Alerts':
    st.title("Malicious Alerts")
    st.write("This view contains only the flows marked as potentially malicious DDoS")
    # This is the view where we just display a preprocessed file
    # You can replace 'malicious_file.csv' with the path to your preprocessed malicious alerts file
    malicious_filepath = 'malicious_flows.csv'

    try:
        # Attempt to read the malicious alerts file as a pandas DataFrame
        df_malicious = pd.read_csv(malicious_filepath)
        st.write(f"Displaying preprocessed malicious alerts from {malicious_filepath}:")
        st.dataframe(df_malicious)

    except Exception as e:
        st.error(f"Could not read {malicious_filepath} as a DataFrame: {e}")

    else:
        st.warning("No files found in the directory.")
else:
    st.error("The specified directory does not exist.")
