
# Imports
from PIL import Image

import streamlit as st

st.set_page_config(
    page_title="Get Around App",
    page_icon="🚗",
    layout="wide"
)
st.title("GA Analysis App")
### Displaying logo ###

image = Image.open('GetAround_logo.png')
st.image(image, use_column_width=True)
st.markdown("""
This app is a Streamlit dashboard that allows you to analyze the data from the [GA Data Analysis]
"""
)


### Sidebar ###
st.sidebar.title("GA Analysis App")