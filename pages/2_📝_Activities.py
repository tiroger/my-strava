#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data, bike_data, get_elev_data_GOOGLE, fetch_activity_streams # Functions to retrive data using strava api and process for visualizations

import ast
import polyline

import pandas as pd
from pandas.api.types import CategoricalDtype
import numpy as np
import datetime as dt

# from PIL import Image
import base64

# import matplotlib.pyplot as plt
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# import folium
# from folium.features import CustomIcon
# from streamlit_folium import folium_static

import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode

import os

import math
import pickle
import pyarrow.feather as feather

#################################
# FUNCTIONS FOR FITNESS METRICS #
#################################

def calculate_activity_metrics(df):
    """
    Calculate cycling metrics for each activity in the DataFrame.
    
    Parameters:
    - df: DataFrame with cycling activity data including 'watts' and 'heartrate'.
    
    Returns:
    - Dictionary with calculated metrics for the activity.
    """
    # Assuming the DataFrame is for a single activity
    # Calculate Average Power
    avg_power = df['watts'].mean()
    
    # Calculate Normalized Power (NP)
    # 1. Calculate the 30-second rolling average of power, raised to the fourth power.
    rolling_avg_power = df['watts'].rolling(window=30, min_periods=1).mean().pow(4)
    # 2. Calculate the average of these values.
    avg_rolling_power = rolling_avg_power.mean()
    # 3. Take the fourth root of this average.
    np = avg_rolling_power ** (1/4)
    
    # Calculate Max and Average Heart Rate
    max_hr = df['heartrate'].max()
    avg_hr = df['heartrate'].mean()
    
    # Calculate total duration in seconds
    total_seconds = df['time'].iloc[-1] - df['time'].iloc[0]
    
    # Calculate other metrics based on NP, avg_power, max_hr, avg_hr, and total_seconds
    metrics = calculate_cycling_metrics(ftp, avg_power, max_hr, avg_hr, np, total_seconds)  # Replace 300 with actual FTP
    
    return metrics


def calculate_cycling_metrics(ftp, avg_power, max_hr, avg_hr, np, total_seconds):
    resting_hr = 49
    if_intensity = np / ftp
    tss = (total_seconds * np * if_intensity) / (ftp * 3600) * 100
    variability_index = np / avg_power if avg_power > 0 else 0
    power_hr_ratio = avg_power / avg_hr if avg_hr > 0 else 0
    hrrc = max_hr - resting_hr
    duration_in_hours = total_seconds / 3600  # Convert seconds to hours
    trimp = duration_in_hours * avg_hr * 0.64 * math.exp(1.92 * (avg_hr / 100))
    # trimp = total_seconds * avg_hr * 0.64 * math.exp(1.92 * avg_hr)
    
    return {
        "Intensity (%)": if_intensity * 100,
        "TSS": tss,
        "Variability Index": variability_index,
        "Power/HR": power_hr_ratio,
        "HRRc": hrrc,
        "TRIMP": f'{trimp:,.0f}',
        "Normalized Power (w)": f'{np:,.0f}',
        "Average Power (w)": f'{avg_power:,.0f}',
        "Max HR": f'{max_hr:,.0f}',
        "Avg HR": f'{avg_hr:,.0f}',
    }
    
def calculate_zone_times(df, ftp):
    # Define the power zones based on FTP
    zones = {
        'Zone 1': (0, 0.55 * ftp),
        'Zone 2': (0.56 * ftp, 0.75 * ftp),
        'Zone 3': (0.76 * ftp, 0.9 * ftp),
        'Zone 4': (0.91 * ftp, 1.05 * ftp),
        'Zone 5': (1.06 * ftp, 1.2 * ftp),
        'Zone 6': (1.21 * ftp, 1.5 * ftp),
        'Zone 7': (1.51 * ftp, float('inf')),
    }
    
    # Initialize a dictionary to hold the total time in each zone
    zone_times = {zone: 0 for zone in zones}
    
    # Iterate over each row in the DataFrame to calculate zone times
    for _, row in df.iterrows():
        watt = row['watts']
        for zone, (lower_bound, upper_bound) in zones.items():
            if lower_bound <= watt <= upper_bound:
                zone_times[zone] += 1
                break
    
    # Convert times to percentages
    total_time = sum(zone_times.values())
    zone_percentages = {zone: (time / total_time) * 100 for zone, time in zone_times.items()}
    time_spent_in_zones = {zone: (time / 3600) for zone, time in zone_times.items()}
    
    return zone_percentages, time_spent_in_zones



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

strava_color_palette = ['#FC4C02', '#45738F', '#DF553B', '#3A18B0', '#FFAA06', '#26A39E', '#951B05', '#D22B0C', '#F5674E'] # [strava orange, strava blue, warm orange, advance indigo, intermediate gold, beginner teal, hard red, medium sienna, easy peach]

# Strava widgets
latest_activities = 'https://www.strava.com/athletes/644338/latest-rides/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'
my_week = 'https://www.strava.com/athletes/644338/activity-summary/53c89f1acdf2bf69a8bc937e3793e9bfb56d64be'

#################
# DATE ELEMENTS #
#################

today = dt.datetime.today()
months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
this_month = dt.datetime.today().month
this_year = dt.datetime.today().year


#################
# PERSONAL DATA #
#################

with open('ftp.pkl', 'rb') as f:
    ftp = pickle.load(f)

# ftp = 192
weight_lbs = 164
weight_kg = weight_lbs * 0.453592

bikes_dict = {'Tie Fighter': 'Storck Scenero', 'Caadie': 'Cannondale CAAD10', 'Dirty McDirtBag': 'Marin Headlands', 'Hillius Maximus': 'Giant TCR', 'Hurt Enforcer': 'Cannondale Slate'}

#################################
# ACQUIRING AND PROCESSING DATA #
#################################

# Get local data # For development
processed_data = pd.read_csv('./data/processed_data.csv')
# processed_data = feather.read_feather('./data/processed_data.feather')
# bikes_df = pd.read_csv('./data/bike_data.csv')
athlete_df = pd.read_csv('./data/athlete_data.csv')

processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')


# st.markdown('<h2 style="color:#45738F">Ride Maps</h2>', unsafe_allow_html=True)

use_cols = ['id', 'start_date_local', 'name', 'distance', 'average_speed', 'total_elevation_gain', 'weighted_average_watts', 'average_heartrate', 'suffer_score', 'year', 'month', 'day', 'type', 'map']
polylines_df = pd.read_csv('./data/processed_data.csv')
polylines_df.start_date_local = pd.DatetimeIndex(polylines_df.start_date_local)
polylines_df.start_date_local = polylines_df.start_date_local.dt.strftime('%m-%d-%Y')
# polylines_df = polylines_df[polylines_df.type == 'Ride'] # We'll only use rides which have a map

try:
    polylines_df['map'] = polylines_df['map'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
except Exception as e:
    st.write('Error in processing polyline data:', e)

####################
# Activities Table #
####################

st.markdown('<h4 style="color:#45738F">Activity Details</h4>', unsafe_allow_html=True)

with st.sidebar:
    # Toggle to select runs, rides or virtual rides
    activity_types = ['Ride', 'Run', 'VirtualRide']
    selected_types = []

    st.markdown("#### Filter by sport")
    for activity in activity_types:
        selected = st.checkbox(activity, True, key=activity)
        selected_types.append(selected)

    # Filter the data based on the selected types
    processed_data = processed_data[processed_data['type'].isin([activity for activity, selected in zip(activity_types, selected_types) if selected])]

# Processing data for table
streamlit_df = processed_data[['id', 'start_date_local', 'name', 'type', 'moving_time', 'distance', 'total_elevation_gain', 'elev_gain_per_mile', 'average_speed', 'average_cadence', 'average_watts', 'average_heartrate', 'suffer_score']]
streamlit_df['start_date_local'] = pd.to_datetime(streamlit_df['start_date_local'])
streamlit_df['start_date_local'] = streamlit_df['start_date_local'].dt.strftime('%Y-%m-%d')
streamlit_df.rename(columns={'start_date_local': 'Date','name': 'Name', 'type': 'Type', 'moving_time': 'Moving Time (h)', 'distance': 'Distance (mi)', 'total_elevation_gain': 'Elevation Gain (ft)', 'elev_gain_per_mile': 'Elevation Gain/mile (ft)', 'average_speed': 'Avg Speed (mph)', 'average_cadence': 'Avg Cadence (rpm)', 'average_watts': 'Avg Power (Watts)', 'average_heartrate': 'Avg Heartrate', 'suffer_score': 'Suffer Score'}, inplace=True)
#streamlit_df.set_index('Date', inplace=True)
# streamlit_df = streamlit_df[streamlit_df['Type'].isin([activity_type])]

# # Sorting table
# streamlit_df.sort_values(by=sort_preference, ascending=False, inplace=True)

gb = GridOptionsBuilder.from_dataframe(streamlit_df)
# gb.configure_pagination(enabled=True) #Add pagination
gb.configure_side_bar() #Add a sidebar
gb.configure_selection('single', use_checkbox=True) # Add selection
gridOptions = gb.build()

grid_response = AgGrid(
streamlit_df,
gridOptions=gridOptions,
data_return_mode='AS_INPUT', 
update_mode='MODEL_CHANGED', 
fit_columns_on_grid_load=True,
enable_enterprise_modules=False,
height=350, 
width='100%',
reload_data=False,
theme='alpine',
# Fit columns on load
)

data = grid_response['data']
selected = grid_response['selected_rows']

if selected:
    selection_df = pd.DataFrame(selected) # player selection from above

    selected_activity_name = selection_df['Name'].values[0]
    selected_activity_id = selection_df['id'].values[0]
    st.markdown(f'<h4 style="color:#45738F">Map</h4>', unsafe_allow_html=True)
    # st.write(f'Additional Details for: {selected_activity_name}')
    
    activity_stream_df = fetch_activity_streams(selected_activity_id)
    activity_stream_df['Id'] = selected_activity_id
    # st.dataframe(activity_stream_df)
    
    # Save the activity stream data to a csv file
    activity_stream_df.to_csv('./data/activity_stream.csv', index=False)
    metrics = calculate_activity_metrics(activity_stream_df)
    
    zone_percentages, time_spent_in_zones = calculate_zone_times(activity_stream_df, ftp)
    time_spent_in_zones_values = list(time_spent_in_zones.values())
    time_spent_in_zones_values = [round(x, 2) for x in time_spent_in_zones_values]
    
    # Filter the polylines dataframe based on the selected activity name
    try:
        selected_activity_name_df = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]
        # st.dataframe(selected_activity_name_df)
        polyline_str = selected_activity_name_df['map'].values[0].get('summary_polyline')
        if polyline_str:
            decoded = polyline.decode(polyline_str)
            # st.dataframe(decoded)
        else:
            decoded = None
            st.write('No polyline data available for this activity')    
    except Exception as e:
        st.write('Error decoding polyline:', e)


# Assuming you have a valid 'decoded' list of latitude-longitude pairs
    if decoded:
        # Adding elevation data from Google Elevation API
        @st.cache_data  # Use the correct decorator
        def elev_profile_chart(decoded_polyline):
            with st.spinner('Calculating elevation profile from Google Elevation. Hang tight...'):
                # Replace 'get_elev_data_GOOGLE' with your actual function to fetch elevation data
                elevation_profile_feet = [get_elev_data_GOOGLE(coord[0], coord[1]) for coord in decoded_polyline]
                return elevation_profile_feet

        # Call the function with the 'decoded' polyline
        elevation_profile_feet = elev_profile_chart(decoded)
        
        # Ensure all plotting libraries and data are correctly initialized
        try:
            centroid = [np.mean([coord[0] for coord in decoded]), np.mean([coord[1] for coord in decoded])]
            lat = [coord[0] for coord in decoded] 
            lon = [coord[1] for coord in decoded]

            # Scattermapbox Plot
            fig = go.Figure(go.Scattermapbox(
                mode="lines",
                lon=lon,
                lat=lat,
                marker=dict(size=2, color="red"),
                line=dict(color="midnightblue", width=2),
            ))
            fig.update_layout(
                mapbox=dict(
                    accesstoken=MAPBOX_TOKEN,
                    style="outdoors",
                    zoom=11,
                    center=dict(lon=centroid[1], lat=centroid[0])
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False
            )

            # # Gather additional details for annotations
            activity_name = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]['name'].values[0]
            activity_distance = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]['distance'].values[0]
            activity_elev_gain = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]['total_elevation_gain'].values[0]
            activity_avg_speed = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]['average_speed'].values[0]
            activity_avg_power = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]['weighted_average_watts'].values[0]
            activity_suffer = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]['suffer_score'].values[0]
            
            # Elevation profile Plot
            fig_elev = px.line(
                x=range(len(elevation_profile_feet)),
                y=pd.Series(elevation_profile_feet).rolling(5).mean(),
                labels={'x': '', 'y': 'Elevation (ft)'}
            ).update_traces(line_color='#FF4500')
            
            fig_elev.update_layout(
                    xaxis=dict(
                        showline=True,
                        showgrid=False,
                        showticklabels=False,
                        linecolor='rgb(204, 204, 204)',
                        linewidth=1,
                        ticks='',
                        tickfont=dict(
                            family='Arial',
                            size=12,
                            color='rgb(82, 82, 82)',
                        ),
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showline=False,
                        gridcolor = 'rgb(235, 236, 240)',
                        showticklabels=True,
                        title='Elevation (ft)',
                        autorange=False,
                        range=[0, 3000]
                    ),
                    autosize=True,
                    hovermode="x unified",
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis_title='',
                    margin=dict(l=0, r=0, t=0, b=0),
                )
            
            fig_elev.add_annotation(
                text=f"<b>RIDE STATS</b> ----------------------",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.99,
                font=dict(size=16, family='Arial'),
            )
            
            font_dict = dict(size=14, 
                            #  color='black', 
                            #  family='Arial'
                             )
            
            fig_elev.add_annotation(
                text=f"<b>Name</b>: {activity_name}",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.95,
                font=font_dict
            )
            
            fig_elev.add_annotation(
                text=f"<b>Distance</b>: {activity_distance} miles",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.91,
                font=font_dict
            )
            
            fig_elev.add_annotation(
                text=f"<b>Elevation Gain</b>: {activity_elev_gain:,.0f} feet",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.87,
                font=font_dict
            )
            
            fig_elev.add_annotation(
                text=f"<b>Avg Speed</b>: {activity_avg_speed} mph",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.83,
                font=font_dict
            )
            
            # fig_elev.add_annotation(
            #     text=f"<b>Avg Weighted Power</b>: {activity_avg_power:,.0f}w",
            #     align='left',
            #     showarrow=False,
            #     xref='paper',
            #     yref='paper',
            #     x=0.05,
            #     y=0.79,
            #     font=font_dict
            # )
            
            fig_elev.add_annotation(
                text=f"<b>Suffer Score</b>: {activity_suffer.astype(int)}",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.79,
                font=font_dict
            )
            
            fig_elev.add_annotation(
                text="----------------------------------------------",
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=0.05,
                y=0.75,
                font=font_dict
            )
            
            # Use Streamlit columns for side-by-side layout
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
            with col2:
                st.plotly_chart(fig_elev, use_container_width=True, config=dict(displayModeBar=False))
        except Exception as e:
            st.write('Error plotting the map or elevation profile:', e)

    
    side_bar = st.sidebar

    # Peak Power numbers in sidebar 
    seconds_in_60min = 60 * 60
    seconds_in_30min = 30 * 60
    seconds_in_20min = 20 * 60
    seconds_in_10min = 10 * 60
    seconds_in_5min = 5 * 60
    seconds_in_1min = 1 * 60
    seconds_in_halfmin = 30
    seconds_in_quartermin = 15
    seconds_in_twentiethmin = 5

    peak_times_list = [seconds_in_60min, seconds_in_30min, seconds_in_20min, seconds_in_10min, seconds_in_5min, seconds_in_1min, seconds_in_halfmin, seconds_in_quartermin, seconds_in_twentiethmin]

    # def find_peak_metric(df, metric, time):
    #     return round(df[metric].rolling(time).mean().max(), 0)
    
    
    def calculate_best_rolling_average_for_window(df, window_seconds, time_column='time', power_column='watts'):
        """
        Calculates the best rolling average for a specified time window based on the total duration of the dataset.

        Args:
        df (DataFrame): The dataframe containing the activity data.
        window_seconds (int): The time window for which to calculate the rolling average, in seconds.
        time_column (str): The name of the column containing time data.
        power_column (str): The name of the column containing power data.

        Returns:
        float: The best average power for the specified time window.
        """
        # Sort and clean data
        df = df.sort_values(time_column)
        df[power_column].fillna(0, inplace=True)  # Replace NA with zeros

        # Calculate total duration in seconds
        total_duration_seconds = df[time_column].max() - df[time_column].min()

        # Ensure the specified window does not exceed the total duration
        if window_seconds > total_duration_seconds:
            print(f"Specified window of {window_seconds} seconds exceeds total duration of {total_duration_seconds} seconds.")
            print("Using total duration as the window.")
            window_seconds = total_duration_seconds

        # Calculate the rolling average for the specified time window
        df[f'rolling_avg_power_{window_seconds}s'] = df[power_column].rolling(window=window_seconds, min_periods=int(window_seconds/2)).mean()

        # Find and return the maximum average power for this window
        max_power = df[f'rolling_avg_power_{window_seconds}s'].max()
        return max_power
    
    peak_60min = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_60min, time_column='time', power_column='watts')
    peak_30min = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_30min, time_column='time', power_column='watts')
    peak_20min = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_20min, time_column='time', power_column='watts')
    peak_10min = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_10min, time_column='time', power_column='watts')
    peak_5min = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_5min, time_column='time', power_column='watts')
    peak_1min = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_1min, time_column='time', power_column='watts')
    peak_halfmin = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_halfmin, time_column='time', power_column='watts')
    peak_quartermin = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_quartermin, time_column='time', power_column='watts')
    peak_twentiethmin = calculate_best_rolling_average_for_window(activity_stream_df, seconds_in_twentiethmin, time_column='time', power_column='watts')
    
    total_move_time = activity_stream_df['time'].max()
    # Convert seconds to human-readable time
    def convert_seconds_to_hms(seconds):
        """
        Converts seconds to human-readable time (hours, minutes, seconds).

        Args:
        seconds (int): The number of seconds to convert.

        Returns:
        str: The human-readable time string.
        """
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02.0f}:{minutes:02.0f}:{seconds:02.0f}"
    
    total_move_time_human_readable = convert_seconds_to_hms(total_move_time)
    # st.write(f'Total Move Time: {total_move_time/60}')
    
    

    # peak_60min = find_peak_metric(activity_stream_df, 'watts', seconds_in_60min)
    # peak_30min = find_peak_metric(activity_stream_df, 'watts', seconds_in_30min)
    # peak_20min = find_peak_metric(activity_stream_df, 'watts', seconds_in_20min)
    # peak_10min = find_peak_metric(activity_stream_df, 'watts', seconds_in_10min)
    # peak_5min = find_peak_metric(activity_stream_df, 'watts', seconds_in_5min)
    # peak_1min = find_peak_metric(activity_stream_df, 'watts', seconds_in_1min)
    # peak_halfmin = find_peak_metric(activity_stream_df, 'watts', seconds_in_halfmin)
    # peak_quartermin = find_peak_metric(activity_stream_df, 'watts', seconds_in_quartermin)
    # peak_twentiethmin = find_peak_metric(activity_stream_df, 'watts', seconds_in_twentiethmin)

    peak_powers = {
        'Peak ⚡': ['60 mins', '30 mins', '20 mins', '10 mins', '5 mins', '1 min', '30 secs', '15 secs', '5 secs'],
        'Power (Watts)': [peak_60min, peak_30min, peak_20min, peak_10min, peak_5min, peak_1min, peak_halfmin, peak_quartermin, peak_twentiethmin],
        'Duration': [60, 30, 20, 10, 5, 1, 0.5, 0.25, 0.2]
    }
    # Convert dictionary to DataFrame
    peak_power_df = pd.DataFrame(peak_powers).set_index('Peak ⚡').fillna('N/A')
    
    # Drop all rows where the ride duration is less than the peak power duration
    peak_power_df = peak_power_df[peak_power_df['Duration'] < total_move_time/60]
    # Drop duration column
    peak_power_df = peak_power_df.drop(columns='Duration')
    
    #  =invert the dataframe
    peak_power_df = peak_power_df.iloc[::-1] 
    
    # Drop columns as soon as they begin the be duplicated
    # peak_power_df = peak_power_df.drop_duplicates()


    # Append 'w' to the watts column if the value is not 'N/A'
    peak_power_df['Power (Watts)'] = peak_power_df['Power (Watts)'].apply(lambda x: f'{x:,.0f}w' if x != 'N/A' else x)
    # peak_power_df_formatted = peak_power_df.style.format({'Power (Watts)': '{:.0f}w'})

    with side_bar:
        st.markdown('<h2 style="color:#45738F">Activity Peak Power</h2>', unsafe_allow_html=True)
        
        # Styling the sidebar with CSS
        st.markdown("""
            <style>
            .sidebar .sidebar-content {
                background-image: linear-gradient(#45738F, #45738F);
                color: white;
            }
            </style>
            """, unsafe_allow_html=True)
        
        st.dataframe(peak_power_df, use_container_width=True)
    
    # st.write('Activity Metrics:', metrics)
    # st.dataframe(activity_stream_df)
    # st.dataframe(polylines_df)
    
    ###############
    # POWER ZOMES #
    ###############
    st.markdown('<h4 style="color:#45738F">Power Output</h4>', unsafe_allow_html=True)
    
    fig = go.Figure()

    activity_stream_df_3s_smoothed = activity_stream_df.rolling(window=3).mean()
    # Add a line chart for power
    fig.add_trace(go.Scatter(x=activity_stream_df['time'], 
                            y=activity_stream_df_3s_smoothed['watts'], 
                            mode='lines', 
                            name='Power (w)', 
                            line=dict(color='#00426A',
                                        width=2.5)))

    fig.add_trace(go.Scatter(x=activity_stream_df['time'], 
                            y=activity_stream_df_3s_smoothed['heartrate'], 
                            mode='lines', 
                            name='Heart Rate (bpm)', 
                            line=dict(color='purple',
                                    width=2.5)))
    fig.add_trace(go.Scatter(x=activity_stream_df['time'], 
                            y=activity_stream_df_3s_smoothed['cadence'], 
                            mode='lines', 
                            name='Cadence (rpm)', 
                            line=dict(color='orange',
                                    width=2.5)))
    
    # Veritcal box for Zone 5 and 6 --if power is greater than 300
    

    customdata = np.stack((activity_stream_df_3s_smoothed['watts'], activity_stream_df_3s_smoothed['heartrate'], activity_stream_df_3s_smoothed['cadence']), axis=-1)
    # Custom x-axis 15 min increments
    x_axis_labels = [f"{h}h {m}m" if h else f"{m}m" for i in range(0, int(activity_stream_df['time'].max()), 900) for h, remainder in [divmod(i, 3600)] for m, s in [divmod(remainder, 60)]]
    fig.update_xaxes(tickvals=list(range(0, int(activity_stream_df['time'].max()), 900)), ticktext=x_axis_labels)
    # Update the layout
    fig.update_layout(
        # title='Power, Normalized Power, and Heart Rate',
        xaxis_title='Time (s)',
        yaxis_title='Value',
        showlegend=True,
        # Transparent background
        plot_bgcolor='rgba(0,0,0,0)',
        # Transparent gridlines
        xaxis=dict(showgrid=False, gridwidth=1, gridcolor='lightgrey'),
        yaxis=dict(showgrid=False, gridwidth=1, gridcolor='lightgrey'),
        margin=dict(l=50, r=0, t=30, b=50),
        height=250,
        hovermode='x unified',
        legend=dict(
            orientation="v",
            yanchor="bottom",
            y=0.7,
            xanchor="right",
            x=1
        ),
    )
    
    fig.update_traces(
    hoverinfo='x+y',
    line=dict(width=2.0),
    # Hover template with customdata and formatting --for each trace
    hovertemplate='%{fullData.name}: <b>%{y:.0f}</b><extra></extra>',
    hoverlabel=dict(font=dict(size=12))
    )
    
    st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))

    # transparency
    fig.update_traces(marker=dict(size=4),
                    selector=dict(mode='markers'))
    
    gauge_col, fitness_metrics_col = st.columns([1, 1])
    
    with gauge_col:
        st.markdown(f'<h4 style="color:#45738F">Power Zones<p><i>Based on FTP of <b id="bold-text">{ftp:,.0f}w</b></i></p>', unsafe_allow_html=True)
        # st.write('Power Zones:', zone_percentages)
        # st.dataframe(zone
        highest_zone = max(zone_percentages, key=zone_percentages.get)
        
        fig = go.Figure()

        num_columns = 2
        num_rows = 3

        for i in range(6):
            row = i // num_columns
            col = i % num_columns

            zone_name = f'Zone {i+1}'
            title_text = zone_name if zone_name != highest_zone else f"<b>{zone_name}</b>"
            title_color = "black" if zone_name != highest_zone else "red"

            # Decrease horizontal spacing by reducing the gap between gauges
            x_padding = 0.0  # Decrease for less horizontal space
            x_start = col / num_columns + x_padding
            x_end = (col + 1) / num_columns - x_padding

            # Increase vertical spacing by increasing the gap between gauges
            y_padding = 0.07  # Increase for more vertical space
            y_start = (num_rows - 1 - row) / num_rows + y_padding  # Adjust for Plotly's y-axis direction
            y_end = (num_rows - row) / num_rows - y_padding
            time_in_zone = time_spent_in_zones_values[i]
            def convert_fraction_to_time(fraction):
                hours = int(fraction)
                minutes = int((fraction - hours) * 60)
                return f"{hours}h {minutes}m"
            
            time_in_zone = convert_fraction_to_time(time_in_zone)
             
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=zone_percentages[zone_name],
                domain={'x': [x_start, x_end], 'y': [y_start, y_end]},
                title={'text': f"<br><br><br><br>{title_text} ({time_in_zone})", 'font': {'color': title_color, 'size': 16}},
                number={'suffix': "%", 'valueformat': '.0f'},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#00426A"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 100], 'color': 'lightgray'}
                    ],
                    # 'threshold': {
                    #     'line': {'color': "red", 'width': 4},
                    #     'thickness': 0.75,
                    #     'value': 100
                    # }
                }
                
                
            ))

        fig.update_layout(height=430, width=600,
                          margin=dict(l=0, r=0, t=10, b=0),)

        st.plotly_chart(fig, use_container_width=True, config=dict(displayModeBar=False))
    
    with fitness_metrics_col:
        # Custom CSS for styling the metrics, especially for highlighting TSS
        st.markdown("""
        <style>
        .metric-block {
            padding: 10px;
            border-radius: 6px;
            color: #808496;
            margin: 5px;
            overflow: auto;
            font-size: 16px;
        }
        .metric-value {
            font-weight: bold;
            font-size: 20px;
        }
        .orange {
            color: #fc4c02;
        }
        
        </style>
        """, unsafe_allow_html=True)

        # Create a 5x2 grid layout
        st.markdown(f'<h4 style="color:#45738F">Fitness Metrics</h4>', unsafe_allow_html=True)
        # Create 2 columns
        cols = st.columns(2)

        # Function to display metric in the specified column
        def display_metric(col, name, value, is_orange=False):
            color_class = "orange" if is_orange else ""
            col.markdown(f"""
            <div class="metric-block">
                <div>{name}</div>
                <div class="metric-value {color_class}">{value}</div>
            </div>
            """, unsafe_allow_html=True)

        # Assuming 'metrics' is your dictionary containing all metrics
        # Format all metrics to 1 decimal place
        metrics = {k: f"{v:.1f}" if isinstance(v, float) else v for k, v in metrics.items()}
        metric_items = list(metrics.items())

        # Check if we have at least ten metrics
        min_metrics_needed = 10
        if len(metric_items) < min_metrics_needed:
            # If there are fewer metrics, pad the list
            metric_items += [(f"Placeholder {i}", "N/A") for i in range(min_metrics_needed - len(metric_items))]

        # Loop through the metrics dictionary and display each metric in the appropriate column
        # Display metrics in two columns, filling five rows
        for i in range(5):
            # Display metric in the first column
            display_metric(cols[0], metric_items[i][0], metric_items[i][1], is_orange=(metric_items[i][0] == "TSS"))
            # Display metric in the second column
            # Check if the second column's metric exists (for cases with fewer than 10 metrics)
            if i + 5 < len(metric_items):
                display_metric(cols[1], metric_items[i + 5][0], metric_items[i + 5][1], is_orange=(metric_items[i + 5][0] == "TSS"))

    st.markdown("""
            - <b>TSS (Training Stress Score)</b>: Measures the total workload of a training session considering duration and intensity.
            - <b>IF (Intensity Factor)</b>: The ratio of your NP (Normalized Power) for an activity to your FTP (Functional Threshold Power).
            - <b>NP (Normalized Power)</b>: An estimate of the power you could have maintained for the same physiological "cost" if your power had been perfectly constant.
            - <b>Variability Index</b>: The ratio of NP to Average Power.
            - <b>Power/HR</b>: The ratio of average power to average heart rate.
            - <b>TRIMP (Training Impulse)</b>: A measure of exercise volume that also considers exercise intensity.
            - <b>HRRc (Heart Rate Reserve used in Calculation)</b>: The difference between maximum heart rate and resting heart rate.
            """, unsafe_allow_html=True)
    
else:
    st.info('Select an activity to view additional details')
    # try:
    #     idx = polylines_df[polylines_df['name'] == selection_df['Name'].values[0]]  # Using index[0] to get first item of index array
    #     polyline_str = polylines_df.at[idx, 'map'].get('summary_polyline')
    #     # Decoding polylines
    #     if polyline_str:
    #         decoded = polyline.decode(polyline_str)
    #     else:
    #         decoded = None
    #         st.write('No polyline data available for this activity')
    # except Exception as e:
    #     st.write('Error decoding polyline:', e)
    #     decoded = None
    
    

# headerColor = '#45738F'
# rowEvenColor = 'lightcyan'
# rowOddColor = 'white'

# # Plotly table
# fig = go.Figure(data=[go.Table(
#     columnorder = [1,2,3,4,5,6,7,8,9,10,11,12],
#     columnwidth = [25,50,18,20,20,23,25,20,24,20,25,17],
#     header=dict(values=list(streamlit_df.columns),
#                 line_color='darkslategray',
#                 fill_color=headerColor,
#     font=dict(color='white', size=13)),
#     cells=dict(values=[streamlit_df['Date'], streamlit_df['Name'], streamlit_df['Type'], streamlit_df['Moving Time (h)'], streamlit_df['Distance (mi)'], streamlit_df['Elevation Gain (ft)'], streamlit_df['Elevation Gain/mile (ft)'], streamlit_df['Avg Speed (mph)'], streamlit_df['Avg Cadence (rpm)'], streamlit_df['Avg Power (Watts)'], streamlit_df['Avg Heartrate'], streamlit_df['Suffer Score']],
#                fill_color = [[rowOddColor,rowEvenColor]*len(streamlit_df.index),], font=dict(color='black', size=12), height=50))
# ])
# fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# # st.dataframe(streamlit_df)
# st.plotly_chart(fig, use_container_width = True, config=dict(displayModeBar = False))



# #################
#  # MAP OF RIDES #
#  ################

# st.markdown('<h2 style="color:#45738F">Ride Maps</h2>', unsafe_allow_html=True)

# polylines_df = pd.read_csv('./data/processed_data.csv', usecols=['start_date_local', 'name', 'distance', 'average_speed', 'total_elevation_gain', 'weighted_average_watts', 'average_heartrate', 'suffer_score', 'year', 'month', 'day', 'type', 'map'])
# polylines_df.start_date_local = pd.DatetimeIndex(polylines_df.start_date_local)
# polylines_df.start_date_local = polylines_df.start_date_local.dt.strftime('%m-%d-%Y')
# polylines_df = polylines_df[polylines_df.type == 'Ride'] # We'll only use rides which have a map


# try:
#     polylines_df['map'] = polylines_df['map'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
# except Exception as e:
#     st.write('Error in processing polyline data:', e)

# option = st.selectbox(
#      'Select a ride for more details',
#      polylines_df['name'].unique()  # This should ensure dropdown is populated correctly
# )

# # Finding dataframe index based on ride name

# if option:
#     try:
#         idx = polylines_df[polylines_df['name'] == option].index[0]  # Using index[0] to get first item of index array
#         polyline_str = polylines_df.at[idx, 'map'].get('summary_polyline')
#         # Decoding polylines
#         if polyline_str:
#             decoded = polyline.decode(polyline_str)
#         else:
#             decoded = None
#             st.write('No polyline data available for this activity')
#     except Exception as e:
#         st.write('Error decoding polyline:', e)
#         decoded = None

#     # Decoding polylines
#     # try:
#     #     # Extract 'summary_polyline' and decode it
#     #     decoded = polylines_df.at[idx, 'map']['summary_polyline']
#     #     decoded = polyline.decode(decoded)
#     # except KeyError:
#     #     st.write('Geocoordinates are unavailable for this activity')
#     #     decoded = None

#     if decoded:
#         # Adding elevation data from Google Elevation API
#         @st.cache_data
#         def elev_profile_chart():
#             with st.spinner('Calculating elevation profile from Google Elevation. Hang tight...'):
#                 elevation_profile_feet = [get_elev_data_GOOGLE(coord[0], coord[1]) for coord in decoded]
#                 return elevation_profile_feet

#         elevation_profile_feet = elev_profile_chart()

#     ########################
#     # Plotly scattermapbox #
#     ########################

#     try:
#         centroid = [
#             np.mean([coord[0] for coord in decoded]), 
#             np.mean([coord[1] for coord in decoded])
#         ]

#         lat = [coord[0] for coord in decoded] 
#         lon = [coord[1] for coord in decoded]

#         fig = go.Figure(go.Scattermapbox(
#             mode = "lines",
#             lon = lon, lat = lat,
#             marker = dict(size = 2, color = "red"),
#             line = dict(color = "midnightblue", width = 2),
#             # text = '‣',
#             textfont=dict(color='#E58606'),
#             textposition = 'bottom center',))
#         fig.update_traces(hovertext='', selector=dict(type='scattermapbox'))
#         fig.update_layout(
#             mapbox = {
#                 'accesstoken': MAPBOX_TOKEN,
#                 'style': "outdoors", 'zoom': 11,
#                 'center': {'lon': centroid[1], 'lat': centroid[0]}
#             },
#             margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
#             showlegend = False)

#         name = polylines_df[polylines_df.index == idx]['name'].values[0]
#         distance = polylines_df[polylines_df.index == idx]['distance'].values[0]
#         elev_gain = polylines_df[polylines_df.index == idx]['total_elevation_gain'].values[0]
#         avg_speed = polylines_df[polylines_df.index == idx]['average_speed'].values[0]
#         avg_power = polylines_df[polylines_df.index == idx]['weighted_average_watts'].values[0] 
#         suffer = polylines_df[polylines_df.index == idx]['suffer_score'].values[0]

#         fig_elev = px.line(elevation_profile_feet, x=range(len(elevation_profile_feet)), y=pd.Series(elevation_profile_feet).rolling(5).mean())
#         fig_elev.update_layout(
#                 xaxis=dict(
#                     showline=True,
#                     showgrid=False,
#                     showticklabels=False,
#                     linecolor='rgb(204, 204, 204)',
#                     linewidth=1,
#                     ticks='',
#                     tickfont=dict(
#                         family='Arial',
#                         size=12,
#                         color='rgb(82, 82, 82)',
#                     ),
#                 ),
#                 yaxis=dict(
#                     showgrid=False,
#                     zeroline=False,
#                     showline=False,
#                     gridcolor = 'rgb(235, 236, 240)',
#                     showticklabels=True,
#                     title='Elevation (ft)',
#                     autorange=False,
#                     range=[0, 3000]
#                 ),
#                 autosize=True,
#                 hovermode="x unified",
#                 showlegend=False,
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 xaxis_title='',
#                 margin=dict(l=0, r=0, t=0, b=0),
#             )

#         fig_elev.add_annotation(text=f"<b>RIDE STATS</b>--------------------", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.99,
#         )
#         fig_elev.add_annotation(text=f"<b>Name</b>: {name}", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.95,
#         )                    
#         fig_elev.add_annotation(text=f"<b>Distance</b>: {distance} miles", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.91,
#         )
#         fig_elev.add_annotation(text=f"<b>Elevation Gain</b>: {elev_gain} feet", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.87,
#         )
#         fig_elev.add_annotation(text=f"<b>Average Speed</b>: {avg_speed} mph", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.83,
#         )     
#         fig_elev.add_annotation(text=f"<b>Weighted Power</b>: {avg_power} Watts", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.79,
#         )
#         fig_elev.add_annotation(text=f"<b>Suffer Score</b>: {suffer.astype(int)}", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.75,
#         )  
#         fig_elev.add_annotation(text="----------------------------------", 
#                             align='left',
#                             showarrow=False,
#                             xref='paper',
#                             yref='paper',
#                             x=0.05,
#                             y=0.71,
#         )

#         col1, col2 = st.columns(2)
#         with col1:
#             st.plotly_chart(fig, use_container_width = True, config=dict(displayModeBar = False))
#         with col2:

#             st.plotly_chart(fig_elev, use_container_width = True, config=dict(displayModeBar = False))

#     except:
#         st.write('Maps not available for this activity')

####################################
# ACTIVITY ADVANCE FITNESS METRICS #
####################################




# def calculate_activity_metrics(df):
#     """
#     Calculate cycling metrics for each activity in the DataFrame.
    
#     Parameters:
#     - df: DataFrame with cycling activity data including 'watts' and 'heartrate'.
    
#     Returns:
#     - Dictionary with calculated metrics for the activity.
#     """
#     # Assuming the DataFrame is for a single activity
#     # Calculate Average Power
#     avg_power = df['watts'].mean()
    
#     # Calculate Normalized Power (NP)
#     # 1. Calculate the 30-second rolling average of power, raised to the fourth power.
#     rolling_avg_power = df['watts'].rolling(window=30, min_periods=1).mean().pow(4)
#     # 2. Calculate the average of these values.
#     avg_rolling_power = rolling_avg_power.mean()
#     # 3. Take the fourth root of this average.
#     np = avg_rolling_power ** (1/4)
    
#     # Calculate Max and Average Heart Rate
#     max_hr = df['heartrate'].max()
#     avg_hr = df['heartrate'].mean()
    
#     # Calculate total duration in seconds
#     total_seconds = df['time'].iloc[-1] - df['time'].iloc[0]
    
#     # Calculate other metrics based on NP, avg_power, max_hr, avg_hr, and total_seconds
#     metrics = calculate_cycling_metrics(ftp, avg_power, max_hr, avg_hr, np, total_seconds)  # Replace 300 with actual FTP
    
#     return metrics

# def calculate_cycling_metrics(ftp, avg_power, max_hr, avg_hr, np, total_seconds):
#     if_intensity = np / ftp
#     tss = (total_seconds * np * if_intensity) / (ftp * 3600) * 100
#     variability_index = np / avg_power if avg_power > 0 else 0
#     power_hr_ratio = avg_power / avg_hr if avg_hr > 0 else 0
#     hrrc = max_hr - avg_hr
#     trimp = total_seconds * avg_hr * 0.64 * math.exp(1.92 * avg_hr)
    
#     return {
#         "Intensity (%)": if_intensity * 100,
#         "TSS": tss,
#         "Variability Index": variability_index,
#         "Power/HR": power_hr_ratio,
#         "HRRc": hrrc,
#         "TRIMP": trimp,
#         "Normalized Power (w)": np,
#         "Average Power (w)": avg_power,
#         "Max HR": max_hr,
#         "Avg HR": avg_hr
#     }

# Example usage
# Assuming 'df' is your DataFrame for a single activity
# df = pd.DataFrame(...)  # Your DataFrame with activity data

# Calculate metrics for the activity
# activity_stream_df = pd.read_csv('./data/activity_stream.csv')
# metrics = calculate_activity_metrics(activity_stream_df)
# st.dataframe(polylines_df)

