##################
# Load libraries #
##################

# from get_strava_data import my_data, process_data, get_elevation

import requests

import pandas as pd
import numpy as np

from PIL import Image
import base64

import ast
import polyline

import matplotlib.pyplot as plt
# import seaborn as sns
import folium
from folium import IFrame
from folium.features import CustomIcon


import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator
raw_symbols = SymbolValidator().values
import streamlit as st

import streamlit.components.v1 as components
from streamlit_folium import folium_static

###############
# CREDENTIALS #
###############

token = MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']


#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# # Get data using strava api # For deployment

# @st.cache_data(show_spinner=False, max_entries=5, ttl=43200)
# def fetch_activities():
#     with st.spinner('Data Refreshing...'):

#         my_data_df = my_data()
#         processed_data = process_data(my_data_df)

#         return processed_data

# @st.cache_data(show_spinner=False, max_entries=5, ttl=43200)
# def bikes():
#     with st.spinner('Data Refreshing...'):
#         bikes = bike_data()

#         return bikes

# processed_data = fetch_activities()
# bikes_df = bikes()


# Get local data # For development
# processed_data = pd.read_csv('./data/processed_data.csv')
# bikes_df = pd.read_csv('./data/bike_data.csv')
# athlete_df = pd.read_csv('./data/athlete_data.csv')



# processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
# processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')

# Saving processed data to csv for future use
# processed_data.to_csv('./data/processed_data.csv', index=False)

polylines_df = pd.read_csv('./data/processed_data.csv', usecols=['name', 'distance', 'total_elevation_gain', 'average_speed', 'weighted_average_watts', 'suffer_score', 'year', 'month', 'day', 'type', 'map'])
polylines_df = polylines_df[polylines_df.type == 'Ride'] # 

polylines_df['decoded_polyline'] = polylines_df['map'].apply(ast.literal_eval)
polylines_df['decoded_polyline'] = pd.json_normalize(polylines_df['decoded_polyline'])['summary_polyline']
# Dropping row with decoded_polyline = None
polylines_df = polylines_df[polylines_df.decoded_polyline.notnull()]

polylines_df['decoded_polyline'] = polylines_df['decoded_polyline'].apply(polyline.decode)


# New column with latitudes
polylines_df['lat'] = polylines_df['decoded_polyline'].apply(lambda x: [coord[0] for coord in x])
# New column with longitudes
polylines_df['lon'] = polylines_df['decoded_polyline'].apply(lambda x: [coord[1] for coord in x])

home_coordinates = [37.664, 122.09292]



####################
# HEAT MAP SECTION #
####################

st.markdown('<h2 style="color:#45738F">Ride Heatmap</h2>', unsafe_allow_html=True)

@st.cache_resource(show_spinner=False, max_entries=5, ttl=86400)
def load_image():

    fig = go.Figure(go.Scattermapbox())

    # Adding a trace for each row in the dataframe
    for i in range(len(polylines_df)):
        fig.add_trace(go.Scattermapbox(mode='lines', lat=polylines_df.lat.values[i], lon=polylines_df.lon.values[i]))

    fig.update_layout(
        margin ={'l':0,'t':0,'b':0,'r':0})
    fig.update_layout(mapbox_accesstoken=MAPBOX_TOKEN)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0},
                    showlegend=False,
                    mapbox = {
                        'style': 'dark',
                        'center': {'lon': -home_coordinates[1], 'lat': home_coordinates[0]},
                        'zoom': 9
                    },
                    height=1000)
    fig.update_traces(line_color='#FC4C02', line_width=2)

    st.plotly_chart(fig, use_container_width=True)
load_image()


# # centroid = [np.mean([coord[0] for coord in all_decoded[0]]), np.mean([coord[1] for coord in all_decoded[0]])]

# # map = folium.Map(location=centroid, zoom_start=9, tiles='stamentoner', control_scale=True, prefer_canvas=True, width=1200)
# try:
#     for i in range(len(all_decoded)):
#         folium.PolyLine(all_decoded[i]).add_to(map)
# except:
#     pass

# # Creating  heatmap layer
# # lat_0 = [coord[0] for coord in all_decoded[0]]
# # lon_0 = [coord[1] for coord in all_decoded[0]]
# # heatmap = HeatMap(list(zip(lat_0, lon_0)), radius=15)

# # lat_1 = [coord[0] for coord in all_decoded[1]]
# # lon_1 = [coord[1] for coord in all_decoded[1]]
# # heatmap1 = HeatMap(list(zip(lat_1, lon_1)), radius=15)

# # lat_2 = [coord[0] for coord in all_decoded[2]]
# # lon_2 = [coord[1] for coord in all_decoded[2]]
# # heatmap2 = HeatMap(list(zip(lat_2, lon_2)), radius=15)

# # lat_3 = [coord[0] for coord in all_decoded[3]]
# # lon_3 = [coord[1] for coord in all_decoded[3]]
# # heatmap3 = HeatMap(list(zip(lat_3, lon_3)), radius=15)

# # for i in range(len(all_decoded)):
# #     lats = [coord[0] for coord in all_decoded[i]]
# #     lons = [coord[1] for coord in all_decoded[i]]

# # for lat, lon in zip(lats, lons):
# #     heatmap = HeatMap(list(zip([lat], [lon])), radius=15).add_to(map)

# # heatmap = HeatMap(list(zip([lat for lat in lats], [lon for lon in lons])), radius=15)
# # heatmap.add_to(map)

# # heatmap.add_to(map)
# # heatmap1.add_to(map)
# # heatmap2.add_to(map)
# # heatmap3.add_to(map)

# # Showing map
# folium_static(map)