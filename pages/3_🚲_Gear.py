#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data, bike_data, get_elev_data_GOOGLE # Functions to retrive data using strava api and process for visualizations

# import ast
# import polyline

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import datetime as dt

# from PIL import Image
import base64

# import matplotlib.pyplot as plt
# # import seaborn as sns
# import calplot
# import plotly.express as px
# import plotly.graph_objects as go

# import folium
# from folium.features import CustomIcon
# from streamlit_folium import folium_static

# import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

# from PIL import Image

###############
# CREDENTIALS #
###############

token = MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

####################
# FOR IMAGES ICONS #
####################

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    body {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# Get data using strava api # For deployment

@st.cache_data(show_spinner=False, max_entries=5, ttl=43200)
def bikes():
    with st.spinner('Data Refreshing...'):
        bikes = bike_data()

        return bikes


bikes_df = bikes()


# # Get local data # For development

# bikes_df = pd.read_csv('./data/bike_data.csv')

############
# THE GEAR #
############

st.markdown('<h2 style="color:#45738F">The Gear</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

tcr_odometer = bikes_df[bikes_df.model_name == 'TCR']['converted_distance'].values[0]
# tcr_average_speed = bikes_df[bikes_df.model_name == 'TCR']['average_speed'].values[0]

storck_odometer = bikes_df[bikes_df.model_name == 'scenero G2']['converted_distance'].values[0]
# storck_average_speed = bikes_df[bikes_df.model_name == 'scenero G2']['average_speed'].values[0]

headlands_odometer = bikes_df[bikes_df.model_name == 'Headlands']['converted_distance'].values[0]
# headlands_average_speed = bikes_df[bikes_df.model_name == 'Headlands']['average_speed'].values[0]

slate_odometer = bikes_df[bikes_df.model_name == 'Slate']['converted_distance'].values[0]
# slate_average_speed = bikes_df[bikes_df.model_name == 'Slate']['average_speed'].values[0]

odometer_metric_color = '#DF553B'

with col1:
    st.markdown('<h4 style="text-align: center;">Giant TCR</h4>', unsafe_allow_html=True)
    st.image('./images/tcr.jpeg')
    st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(tcr_odometer)} miles</b></h5>', unsafe_allow_html=True)
with col2:
    st.markdown('<h4 style="text-align: center;">Storck Scenero</h4>', unsafe_allow_html=True)
    st.image('./images/scenero_2.jpeg')
    st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(storck_odometer)} miles</b></h5>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<h4 style="text-align: center;">Marin Headlands</h4>', unsafe_allow_html=True)
    st.image('./images/headlands.jpeg')
    st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(headlands_odometer)} miles</b></h5>', unsafe_allow_html=True)
with col2:
    st.markdown('<h4 style="text-align: center;">Cannondale Slate</h4>', unsafe_allow_html=True)
    st.image('./images/slate.jpeg')
    st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(slate_odometer)} miles</b></h5>', unsafe_allow_html=True)