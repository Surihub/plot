import streamlit as st

# Create a container
# Split the container into two columns
col1, col2 = st.columns(2)

# Use these columns to place your widgets
with col1:
    with st.container():

        st.write("This is the left column.")
        # Add more widgets here as needed

with col2:
    with st.container():

        st.write("This is the right column.")
        # Add more widgets here as needed