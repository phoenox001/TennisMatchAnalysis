import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Home", page_icon="🎾")


@st.cache_data
def load_data():
    data = []
    return data


st.title("Wimbledon 2025 Predictions")
st.divider()
st.header("Home")
