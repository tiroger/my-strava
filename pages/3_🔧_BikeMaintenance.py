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

import os

import pyarrow.feather as feather

# from PIL import Image

###############
# CREDENTIALS #
###############

# MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
# GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']

CLIENT_ID = os.environ['CLIENT_ID']
CLIENT_SECRET = os.environ['CLIENT_SECRET']
REFRESH_TOKEN = os.environ['REFRESH_TOKEN']
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']

MAPBOX_TOKEN = os.environ['MAPBOX_TOKEN']

###########################
# Main Page Configuration #
###########################

st.set_page_config(page_title='My Strava', 
                    page_icon='./icons/cropped-rtc-favicon.png', 
                    layout="wide", 
                    initial_sidebar_state="auto")

####################
# FOR IMAGES ICONS #
####################

@st.cache_data()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# def set_png_as_page_bg(png_file):
#     bin_str = get_base64_of_bin_file(png_file)
#     page_bg_img = '''
#     <style>
#     body {
#     background-image: url("data:image/png;base64,%s");
#     background-size: cover;
#     }
#     </style>
#     ''' % bin_str
    
#     st.markdown(page_bg_img, unsafe_allow_html=True)
#     return

#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# Get data using strava api # For deployment

# @st.cache_data(show_spinner=False, max_entries=5, ttl=43200)
# def bikes():
#     with st.spinner('Data Refreshing...'):
#         bikes = bike_data()

#         return bikes


# bikes_df = bikes()


# # Get local data # For development

bikes_df = pd.read_csv('./data/bike_data.csv')
# bikes_df = feather.read_feather('./data/bike_data.feather')

bikes_df = bikes_df[(bikes_df.retired == False) & (bikes_df.name != 'Nike Zoom Fly 5')] # Remove retired bikes and running shoes
bikes_df = bikes_df.replace({'model_name': {'caad13': 'CAAD13'
                                            },
                            'brand_name': {'cannondale': 'Cannondale'
                                           }
                             })
# bikes_df = bikes_df.replace({'brand_name': {'cannondale': 'Cannondale'}})

#############
# THE BIKES #
#############

# serviceable_parts = ['Chain', 'Cassette', 'Crankset', 'Brake Pads', 'Tires', 'Bar Tape', 'Cables', 'Pedals', 'Saddle', 'Handlebar', 'Stem', 'Seatpost', 'Wheels', 'Bottom Bracket', 'Headset', 'Derailleur Hanger', 'Brake Cables', 'Brake Housing', 'Brake Pads', 'Brake Rotors', 'Front Derailleur', 'Rear Derailleur']
# services = ['Clean', 'Lube', 'Top-Up', 'Adjust', 'Replace', 'Upgrade', 'Install', 'Tune', 'True', 'Bleed', 'Wrap', 'Install', 'Remove', 'Repack', 'Rebuild']


parts_and_available_services_json = {
    'Chain': ['Top-Up', 'Lube', 'New', 'Hot Wax'],
    'Cassette': ['New'],
    'Crankset': ['New', 'Service'],
    'Brake Pads': ['New'],
    'Tires': ['New', 'Top-Up'],
    'Bar Tape': ['New'],
    'Cables': ['New'],
    'Pedals': ['New', 'Service'],
    'Saddle': ['New'],
    'Handlebar': ['New'],
    'Stem': ['New'],
    'Seatpost': ['New'],
    'Wheels': ['True', 'Service', 'Retape'],
    'Bottom Bracket': ['New', 'Service'],
    'Headset': ['New', 'Service'],
    'Derailleur Hanger': ['New'],
    'Brake Cables': ['New'],
    'Brake Housing': ['New'],
    'Brake Pads': ['New'],
    'Brake Rotors': ['New'],
    'Front Derailleur': ['New', 'Service'],
    'Rear Derailleur': ['New', 'Service'],
}

parts_keys = list(parts_and_available_services_json.keys()).sort()
services_values = list(parts_and_available_services_json.values()).sort()
parts_and_available_services_json = {k: parts_and_available_services_json[k] for k in sorted(parts_and_available_services_json)}


# Sidebar dropdown to select bike
with st.sidebar.markdown('<h4 style="color:#45738F">The Gear</h4>', unsafe_allow_html=True):
    selected_bike = st.selectbox('Select a Bike', bikes_df.name.unique())
with st.sidebar.markdown('<h4 style="color:#45738F">Service</h4>', unsafe_allow_html=True):
    selected_part = st.selectbox('Select a Part', list(parts_and_available_services_json.keys()))
with st.sidebar.markdown('<h4 style="color:#45738F">Service</h4>', unsafe_allow_html=True):
    if selected_part:
        selected_service = st.selectbox('Select a Service', parts_and_available_services_json[selected_part])

selected_bike_df = bikes_df[bikes_df['name'] == selected_bike]

# File where the DataFrame will be saved
maintenance_file = 'maintenance_log.csv'

if os.path.exists(maintenance_file) and os.stat(maintenance_file).st_size > 0:
    try:
        maintenance_df = pd.read_csv(maintenance_file)
        # Continue processing the DataFrame as required
    except pd.errors.EmptyDataError:
        # The file exists but is empty, handle the situation appropriately
        print("The file exists but is empty. Please check the file content.")
        maintenance_df = pd.DataFrame()  # Create an empty DataFrame as a fallback
    except Exception as e:
        # Handle other potential exceptions
        print(f"An error occurred: {e}")
        maintenance_df = pd.DataFrame()  # Create an empty DataFrame as a fallback
else:
    print("File does not exist or is empty.")
    maintenance_df = pd.DataFrame()  # Create an empty DataFrame as a fallback

# Function to load data from CSV if it exists and is not empty, otherwise return empty DataFrame
def load_maintenance_data(file_path):
    # Check if the file exists and has content that could be a valid header
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        with open(file_path, 'r') as file:
            header = file.readline()
            # Check if the header has at least one column name
            if header and len(header.split(',')) > 0:
                try:
                    # File exists, is not empty, and has a valid header, attempt to read it
                    return pd.read_csv(file_path)
                except pd.errors.EmptyDataError:
                    # The file exists, might not be completely empty but fails due to lack of data to form columns
                    print("The file exists but does not contain valid data to form columns.")
                    return pd.DataFrame(columns=['bike', 'part', 'service', 'date', 'mileage'])  # Adjust columns as necessary
            else:
                # File exists but the header is not valid
                print("The file exists but the header is invalid or empty.")
                return pd.DataFrame(columns=['bike', 'part', 'service', 'date', 'mileage'])  # Adjust columns as necessary
    else:
        # File does not exist or is empty
        print("File does not exist or is empty.")
        return pd.DataFrame()

maintenance_file = 'maintenance_log.csv'
maintenance_df = load_maintenance_data(maintenance_file)

# Initialize session state for maintenance list
if 'maintenance_list' not in st.session_state:
    try:
        # Try to load existing maintenance records from CSV
        maintenance_df = pd.read_csv(maintenance_file)
        st.session_state.maintenance_list = maintenance_df.to_dict('records')
    except FileNotFoundError:
        # If the file does not exist, start with an empty list
        st.session_state.maintenance_list = []

# Assuming bikes_df, selected_bike, selected_part, and selected_service are defined above this code
selected_bike_df = bikes_df[bikes_df['name'] == selected_bike]

with st.sidebar:
    log_service = st.button('Log Service')
    clear_last_entry = st.button('Clear Last Entry')

if log_service:
    logged_date = dt.datetime.now()
    logged_mileage = selected_bike_df['converted_distance'].values[0] if not selected_bike_df.empty else 0
    service_dict = {
        'bike': selected_bike,
        'part': selected_part,
        'service': selected_service,
        'date': logged_date.strftime('%Y-%m-%d'),
        'mileage': logged_mileage
    }

    # Append the service_dict to the session state maintenance list
    st.session_state.maintenance_list.append(service_dict)

    # Update the DataFrame with the new list and save to CSV
    maintenance_df = pd.DataFrame(st.session_state.maintenance_list)
    maintenance_df.to_csv(maintenance_file, index=False)

if clear_last_entry:
    # Remove the last entry from the session state if it exists
    if st.session_state.maintenance_list:
        st.session_state.maintenance_list.pop()
        # Update the DataFrame and save to CSV after removal
        maintenance_df = pd.DataFrame(st.session_state.maintenance_list)
        maintenance_df.to_csv(maintenance_file, index=False)

# Display the updated DataFrame in Streamlit, whether or not new data was added
# st.dataframe(maintenance_df)



# st.markdown('<h4 style="color:#45738F">About</h4>', unsafe_allow_html=True)
bike_info_col_1, bike_photo_col  = st.columns([7, 5])

bike_to_photo_dict = {
    'b8099416': './images/tcr.jpeg',
    'b4073790': './images/storck.jpeg',
    'b8615449': './images/headlands.jpeg',
    'b11212849': './images/caad13.jpeg',
    'b5245627': './images/slate.jpeg',
    'b4196400': './images/slate.jpeg',
    'b8029179': './images/marin.jpeg',
}

bike_parts_images = []

selected_bike_id = selected_bike_df.id.values[0]
selected_bike_nickname = selected_bike_df.nickname.values[0]
selected_bike_brand = selected_bike_df.brand_name.values[0]
selected_bike_model = selected_bike_df.model_name.values[0]
selected_bike_odometer = selected_bike_df.converted_distance.values[0]



try:
    maintenance_df = pd.read_csv(maintenance_file)
    # st.dataframe(maintenance_df)
    selected_bike_maintenance_df = maintenance_df[maintenance_df['bike'] == selected_bike]
    # st.dataframe(selected_bike_maintenance_df)
    
    chain_relube_schedule = 150
    chain_rewax_schedule = 1000
    chain_topup_schedule = 100
    tire_topup_schedule = 4
    tire_replace_schedule = 2000

    # Check if maintenance_df is not empty and variables are defined before performing operations
    if not selected_bike_maintenance_df.empty:
        if 'chain_relube_schedule' in locals():
            lube_chain_maintenance_df = selected_bike_maintenance_df[(selected_bike_maintenance_df['part'] == 'Chain') & (selected_bike_maintenance_df['service'] == 'Lube')]
            if not lube_chain_maintenance_df.empty:
                lube_chain_maintenance_df['NextLube'] = lube_chain_maintenance_df['mileage'] + chain_relube_schedule
                # st.dataframe(lube_chain_maintenance_df)

        if 'chain_rewax_schedule' in locals():
            hotwax_chain_maintenance_df = selected_bike_maintenance_df[(selected_bike_maintenance_df['part'] == 'Chain') & (selected_bike_maintenance_df['service'] == 'Hot Wax')]
            if not hotwax_chain_maintenance_df.empty:
                hotwax_chain_maintenance_df['NextHotWax'] = hotwax_chain_maintenance_df['mileage'] + chain_rewax_schedule
                # st.dataframe(hotwax_chain_maintenance_df)

        if 'chain_topup_schedule' in locals():
            waxtopup_chain_maintenance_df = selected_bike_maintenance_df[(selected_bike_maintenance_df['part'] == 'Chain') & (selected_bike_maintenance_df['service'] == 'Top-Up')]
            if not waxtopup_chain_maintenance_df.empty:
                waxtopup_chain_maintenance_df['NextWaxTopUp'] = waxtopup_chain_maintenance_df['mileage'] + chain_topup_schedule
                # st.dataframe(waxtopup_chain_maintenance_df)

        if 'tire_topup_schedule' in locals():
            tiretopup_maintenance_df = selected_bike_maintenance_df[(selected_bike_maintenance_df['part'] == 'Tires') & (selected_bike_maintenance_df['service'] == 'Top-Up')]
            if not tiretopup_maintenance_df.empty:
                tiretopup_maintenance_df['date'] = pd.to_datetime(tiretopup_maintenance_df['date'])
                tiretopup_maintenance_df['NextSealantTopUp'] = tiretopup_maintenance_df['date'] + pd.DateOffset(months=tire_topup_schedule)
                # st.dataframe(tiretopup_maintenance_df)
        if 'tire_replace_schedule' in locals():
            tire_replaced_maintenance_df = selected_bike_maintenance_df[(selected_bike_maintenance_df['part'] == 'Tires') & (selected_bike_maintenance_df['service'] == 'New')]
            if not tire_replaced_maintenance_df.empty:
                tire_replaced_maintenance_df['NextTireReplace'] = tire_replaced_maintenance_df['mileage'] + tire_replace_schedule
                # st.dataframe(tire_replaced_maintenance_df)
except:
    pass


with bike_info_col_1:
    st.markdown("""
                <style>
                .metric-block {
                    padding: 10px;
                    border-radius: 6px;
                    color: #808496;
                    margin: 1px 0px;
                    overflow: auto;
                }
                .metric-header {
                    font-size: 15px;
                    margin-bottom: 5px;
                }
                .metric-value {
                    font-size: 24px;
                    font-weight: bold;
                }
                .positive {
                    background-color: #4caf50;
                }
                .negative {
                    background-color: #f44336;
                }
                #block-text {
                    color: white;
                }
                #bold-text {
                    font-weight: bold;
                    color: #fc4c02;
                    }
                #strava-orange {
                    color: #fc4c02;
                }
                
                </style>
                """, unsafe_allow_html=True)
    

    col1, col2 = st.columns([3,2])
    with col1:
        html_template = f"""
        <div class="metric-block">
            <div class="metric-value">{selected_bike_brand} {selected_bike_model} | <span class="metric-value" id="strava-orange">{selected_bike_nickname}</span></div>
            <div class="metric-header"><b>Odometer</b></div>
            <div class="metric-value">{selected_bike_odometer:,}mi</div>
            <div class="odometer" id="odometer"></div>
            <div class="metric-header">---</div>
            <div class="metric-header"><b>Scheduled Maintenance</b></div>
            
        </div>
        <script>
            function updateOdometer(newVal) {{
                const digits = newVal.toString().padStart(5, '0').split('');
                digits.forEach((num, index) => {{
                    const digitElement = document.getElementById(`digit${{index + 1}}`);
                    const shift = -parseInt(num) * 40; // Make sure this matches the .num height
                    digitElement.style.transform = `translateY(${{shift}}px)`;
                }});
            }}

            // Ensure this is parsing correctly. No commas should be in the number for JavaScript.
            let mileage = parseInt('{selected_bike_odometer}'.replace(/,/g, ''));
            window.onload = () => {{
                updateOdometer(mileage);
            }};
        </script>
        """
        st.markdown(html_template, unsafe_allow_html=True)
        # with col2:
        #     st.markdown(html_template, unsafe_allow_html=True)

    # Writing out the maintenance schedule
    
    
    with col1:
        # Read daaframes and display next maintenance
        try:
            if 'NextLube' in lube_chain_maintenance_df.columns:
                next_lube = lube_chain_maintenance_df['NextLube'].values[0]
                st.markdown(f'<div class="metric-header">Next Chain Lube: <b id="bold-text">{next_lube:,} miles</b></div>', unsafe_allow_html=True)
            if 'NextHotWax' in hotwax_chain_maintenance_df.columns:
                next_hotwax = hotwax_chain_maintenance_df['NextHotWax'].values[0]
                st.markdown(f'<div class="metric-header">Next Chain Hot Wax: <b id="bold-text">{next_hotwax:,} miles</b></div>', unsafe_allow_html=True)
            if 'NextWaxTopUp' in waxtopup_chain_maintenance_df.columns:
                next_waxtopup = waxtopup_chain_maintenance_df['NextWaxTopUp'].values[0]
                st.markdown(f'<div class="metric-header">Next Chain Wax Top-Up: <b id="bold-text">{next_waxtopup:,} miles</b></div>', unsafe_allow_html=True)
            if 'NextSealantTopUp' in tiretopup_maintenance_df.columns:
                next_tiretopup = tiretopup_maintenance_df['NextSealantTopUp'].values[0]
                next_tiretopup_date = tiretopup_maintenance_df['NextSealantTopUp'].dt.strftime('%B %d, %Y').values[0]
                st.markdown(f'<div class="metric-header">Next Tire Sealant Top-Up: <b id="bold-text">{next_tiretopup_date}</b></div>', unsafe_allow_html=True)
            if 'NextTireReplace' in tire_replaced_maintenance_df.columns:
                next_tirereplace = tire_replaced_maintenance_df['NextTireReplace'].values[0]
                next_tirereplace_date = tire_replaced_maintenance_df['NextTireReplace'].values[0]
                st.markdown(f'<div class="metric-header">Next Tire Replacement: <b id="bold-text">{next_tirereplace_date} miles</b></div>', unsafe_allow_html=True)
            
        except:
            st.warning('No maintenance scheduled for this bike.')


with bike_photo_col:
    st.image(bike_to_photo_dict[selected_bike_id], use_column_width=True)


# Expander to display the maintenance log for the selected bike
st.markdown('<h4 style="color:#45738F">Service History</h4>', unsafe_allow_html=True)
try:
    with st.expander('Expand to View Maintenance Log'):
        st.dataframe(maintenance_df[maintenance_df['bike'] == selected_bike], width=800)
except:
    st.warning('No maintenance log available for this bike.')


















#     col1, col2 = st.columns([3,2])
#     with col1:
#         html_template = f"""
#         <div class="metric-block">
#             <div class="metric-value">{selected_bike_brand} {selected_bike_model} | <span class="metric-value" id="strava-orange">{selected_bike_nickname}</span></div>
#             <div class="metric-header"><b>Odometer</b></div>
#             <div class="metric-value">{selected_bike_odometer:,}mi</div>
#             <div class="odometer" id="odometer">
#             </div>
#             <!-- New section for 6x3 columns -->
#             <div class="metrics-grid">
#                 <div class="row">
#                     <div class="column"><div class="metric-cell">Metric 1</div></div>
#                     <div class="column"><div class="metric-cell">Metric 2</div></div>
#                     <div class="column"><div class="metric-cell">Metric 3</div></div>
#                 </div>
#                 <div class="row">
#                     <div class="column"><div class="metric-cell">Metric 4</div></div>
#                     <div class="column"><div class="metric-cell">Metric 5</div></div>
#                     <div class="column"><div class="metric-cell">Metric 6</div></div>
#                 </div>
#                 <div class="row">
#                     <div class="column"><div class="metric-cell">Metric 7</div></div>
#                     <div class="column"><div class="metric-cell">Metric 8</div></div>
#                     <div class="column"><div class="metric-cell">Metric 9</div></div>
#                 </div>
#             </div>
#         </div>
#         <script>
#             function updateOdometer(newVal) {{
#                 const digits = newVal.toString().padStart(5, '0').split('');
#                 digits.forEach((num, index) => {{
#                     const digitElement = document.getElementById(`digit${{index + 1}}`);
#                     const shift = -parseInt(num) * 40; // Make sure this matches the .num height
#                     digitElement.style.transform = `translateY(${{shift}}px)`;
#                 }});
#             }}

#             // Ensure this is parsing correctly. No commas should be in the number for JavaScript.
#             let mileage = parseInt('{selected_bike_odometer}'.replace(/,/g, ''));
#             window.onload = () => {{
#                 updateOdometer(mileage);
#             }};
#         </script>
#         """
#         st.markdown(html_template, unsafe_allow_html=True)
#         # with col2:
#         #     st.markdown(html_template, unsafe_allow_html=True)

# with bike_photo_col:
#     st.image(bike_to_photo_dict[selected_bike_id], use_column_width=True)
        











# col1, col2 = st.columns(2)

# tcr_odometer = bikes_df[bikes_df.model_name == 'TCR']['converted_distance'].values[0]
# # tcr_average_speed = bikes_df[bikes_df.model_name == 'TCR']['average_speed'].values[0]

# storck_odometer = bikes_df[bikes_df.model_name == 'scenero G2']['converted_distance'].values[0]
# # storck_average_speed = bikes_df[bikes_df.model_name == 'scenero G2']['average_speed'].values[0]

# headlands_odometer = bikes_df[bikes_df.model_name == 'Headlands']['converted_distance'].values[0]
# # headlands_average_speed = bikes_df[bikes_df.model_name == 'Headlands']['average_speed'].values[0]

# slate_odometer = bikes_df[bikes_df.model_name == 'Slate']['converted_distance'].values[0]
# # slate_average_speed = bikes_df[bikes_df.model_name == 'Slate']['average_speed'].values[0]

# odometer_metric_color = '#DF553B'

# with col1:
#     st.markdown('<h4 style="text-align: center;">Giant TCR</h4>', unsafe_allow_html=True)
#     st.image('./images/tcr.jpeg')
#     st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(tcr_odometer)} miles</b></h5>', unsafe_allow_html=True)
# with col2:
#     st.markdown('<h4 style="text-align: center;">Storck Scenero</h4>', unsafe_allow_html=True)
#     st.image('./images/scenero_2.jpeg')
#     st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(storck_odometer)} miles</b></h5>', unsafe_allow_html=True)

# col1, col2 = st.columns(2)

# with col1:
#     st.markdown('<h4 style="text-align: center;">Marin Headlands</h4>', unsafe_allow_html=True)
#     st.image('./images/headlands.jpeg')
#     st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(headlands_odometer)} miles</b></h5>', unsafe_allow_html=True)
# with col2:
#     st.markdown('<h4 style="text-align: center;">Cannondale Slate</h4>', unsafe_allow_html=True)
#     st.image('./images/slate.jpeg')
#     st.markdown(f'<h5 style="color:lightgrey">Odometer: <b style="color:{odometer_metric_color}">{"{:,}".format(slate_odometer)} miles</b></h5>', unsafe_allow_html=True)


