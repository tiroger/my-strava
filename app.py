#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data, bike_data, get_elev_data_GOOGLE # Functions to retrive data using strava api and process for visualizations

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

import folium
from folium.features import CustomIcon
from streamlit_folium import folium_static

import matplotlib.pyplot as plt

import streamlit as st
import streamlit.components.v1 as components

###############
# CREDENTIALS #
###############

token = MAPBOX_TOKEN = st.secrets['MAPBOX_TOKEN']
GOOGLE_API_KEY = st.secrets['GOOGLE_API_KEY']


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

ftp = 192
weight_lbs = 164
weight_kg = weight_lbs * 0.453592

bikes_dict = {'Tie Fighter': 'Storck Scenero', 'Caadie': 'Cannondale CAAD10', 'Dirty McDirtBag': 'Marin Headlands', 'Hillius Maximus': 'Giant TCR', 'Hurt Enforcer': 'Cannondale Slate'}



#################################
# ACQUIRING AND PROCESSING DATA #
#################################


##### Use one of two options below #####

# Get data using strava api # For deployment

@st.cache(show_spinner=False, max_entries=5, ttl=86400, allow_output_mutation=True)
def fetch_activities():
    with st.spinner('Data Refreshing... May take up to 5 minutes.'):

        my_data_df = my_data()
        processed_data = process_data(my_data_df)

        return processed_data

@st.cache(show_spinner=False, max_entries=5, ttl=86400, allow_output_mutation=True)
def bikes():
    with st.spinner('Data Refreshing... May take up to 5 minutes.'):
        bikes = bike_data()

        return bikes

processed_data = fetch_activities()
bikes_df = bikes()


# Get local data # For development
# processed_data = pd.read_csv('./data/processed_data.csv')
# bikes_df = pd.read_csv('./data/bike_data.csv')
# athlete_df = pd.read_csv('./data/athlete_data.csv')



processed_data['start_date_local'] = pd.to_datetime(processed_data['start_date_local'])
processed_data['start_date_local'] = processed_data['start_date_local'].dt.strftime('%m-%d-%Y')

# Saving processed data to csv for future use
processed_data.to_csv('./data/processed_data.csv', index=False)



# Data for dahsboard
start_date = processed_data.year.min()
burger_calories = 354
total_activities = processed_data.id.count()
num_rides = (processed_data.type == 'Ride').sum()
num_runs = (processed_data.type == 'Workout').sum()
distance_traveled = processed_data.distance.sum().astype(int)
feet_climbed = processed_data.total_elevation_gain.sum().astype(int)
total_kudos = processed_data.kudos_count.sum()
earth_circumference = 24901 # earth circumference in miles 
perc_around_the_earth = (distance_traveled / earth_circumference)
total_time = processed_data.moving_time.sum()


##################
# SIDE BAR START #
##################

# with st.sidebar: # Option to show strava widgets
#     st.header('Overview')
    # components.iframe(my_week, height=170)
    # components.iframe(latest_activities, height=500)

############
# Overview #
############

with st.sidebar:
    # st.image('./icons/tri.jpeg')
    st.markdown('<h1 style="color:#FC4C02">Overview</h1>', unsafe_allow_html=True)
    st.subheader(f'Member since {start_date}')
    st.image('./data/profile.png', width=300, output_format='PNG')


    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/dumbbell.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Activities", value=total_activities)

    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/stopwatch.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Moving Time (hours)", value=f'{round(total_time,1)}')

    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/road.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Distance (miles)", value=f'{"{:,}".format(distance_traveled)}')

    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/mountain.png', width=100, output_format='PNG')
    with col2:
        st.metric(label="Elevation Gain (ft)", value=f'{"{:,}".format(feet_climbed)}')
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('./icons/like.png', width=80, output_format='PNG')
    with col2:
        st.metric(label="Kudos", value="{:,}".format(total_kudos))

    ########################
    # Activities Pie chart #
    ########################

    grouped_by_type = processed_data.groupby('type').agg({'type': 'count'}).rename(columns={'type': 'total'}).sort_values('total', ascending=False).reset_index()
    grouped_by_type.loc[grouped_by_type.total < 20, 'type'] = 'Other'
    pie_df = grouped_by_type.groupby('type').agg({'total': 'sum'}).rename(columns={'total': 'total'}).reset_index()

    activities = pie_df.type
    breakdown_by_type = pie_df.total

    fig = go.Figure(data=[go.Pie(labels=activities, values=breakdown_by_type, hole=.7)])
    fig.update_traces(textposition='outside', textinfo='label+value')
    fig.update_layout(showlegend=False, uniformtext_minsize=16, uniformtext_mode='hide', hovermode=False, paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), annotations=[dict(text='Activities', x=0.5, y=0.5, font_size=20, showarrow=False)])

    st.plotly_chart(fig, use_container_width=True, config= dict(
            displayModeBar = False))

################
# SIDE BAR END #
################

#######################################################################
#######################################################################

###################
# MAIN PAGE START #
###################

st.markdown('<h1 style="color:#FC4C02">MY STRAVA JOURNEY</h1>', unsafe_allow_html=True)


####################
# Activities Table #
####################

st.markdown('<h2 style="color:#45738F">Activities</h2>', unsafe_allow_html=True)

# Filter by activity type
activity_type = st.selectbox('Filter by sport', ['Ride', 'Workout', 'WeightTraining', 'Walk', 'Hike', 'Yoga',
       'VirtualRide', 'Elliptical', 'Run', 'Swim', 'AlpineSki']) # Select from dropdown

sort_preference = st.radio('Sort by', ('Date', 'Distance (mi)', 'Elevation Gain (ft)', 'Elevation Gain/mile (ft)', 'Avg Speed (mph)', 'Avg Power (Watts)', 'Avg Heartrate', 'Suffer Score'))

# Processing data for table
streamlit_df = processed_data[['start_date_local', 'name', 'type', 'moving_time', 'distance', 'total_elevation_gain', 'elev_gain_per_mile', 'average_speed', 'average_cadence', 'average_watts', 'average_heartrate', 'suffer_score']]
streamlit_df['start_date_local'] = pd.to_datetime(streamlit_df['start_date_local'])
streamlit_df['start_date_local'] = streamlit_df['start_date_local'].dt.strftime('%Y-%m-%d')
streamlit_df.rename(columns={'start_date_local': 'Date','name': 'Name', 'type': 'Type', 'moving_time': 'Moving Time (h)', 'distance': 'Distance (mi)', 'total_elevation_gain': 'Elevation Gain (ft)', 'elev_gain_per_mile': 'Elevation Gain/mile (ft)', 'average_speed': 'Avg Speed (mph)', 'average_cadence': 'Avg Cadence (rpm)', 'average_watts': 'Avg Power (Watts)', 'average_heartrate': 'Avg Heartrate', 'suffer_score': 'Suffer Score'}, inplace=True)
#streamlit_df.set_index('Date', inplace=True)
streamlit_df = streamlit_df[streamlit_df['Type'].isin([activity_type])]

# Sorting table
streamlit_df.sort_values(by=sort_preference, ascending=False, inplace=True)

headerColor = '#45738F'
rowEvenColor = 'lightcyan'
rowOddColor = 'white'

# Plotly table
fig = go.Figure(data=[go.Table(
    columnorder = [1,2,3,4,5,6,7,8,9,10,11,12],
    columnwidth = [25,50,18,20,20,23,25,20,24,20,25,17],
    header=dict(values=list(streamlit_df.columns),
                line_color='darkslategray',
                fill_color=headerColor,
    font=dict(color='white', size=13)),
    cells=dict(values=[streamlit_df['Date'], streamlit_df['Name'], streamlit_df['Type'], streamlit_df['Moving Time (h)'], streamlit_df['Distance (mi)'], streamlit_df['Elevation Gain (ft)'], streamlit_df['Elevation Gain/mile (ft)'], streamlit_df['Avg Speed (mph)'], streamlit_df['Avg Cadence (rpm)'], streamlit_df['Avg Power (Watts)'], streamlit_df['Avg Heartrate'], streamlit_df['Suffer Score']],
               fill_color = [[rowOddColor,rowEvenColor]*len(streamlit_df.index),], font=dict(color='black', size=12), height=50))
])
fig.update_layout(margin=dict(l=0, r=0, t=0, b=0))

# st.dataframe(streamlit_df)
st.plotly_chart(fig, use_container_width = True, config=dict(displayModeBar = False))



#################
 # MAP OF RIDES #
 ################

st.markdown('<h2 style="color:#45738F">Ride Maps</h2>', unsafe_allow_html=True)

polylines_df = pd.read_csv('./data/processed_data.csv', usecols=['start_date_local', 'name', 'distance', 'average_speed', 'total_elevation_gain', 'weighted_average_watts', 'average_heartrate', 'suffer_score', 'year', 'month', 'day', 'type', 'map'])
polylines_df.start_date_local = pd.DatetimeIndex(polylines_df.start_date_local)
polylines_df.start_date_local = polylines_df.start_date_local.dt.strftime('%m-%d-%Y')
polylines_df = polylines_df[polylines_df.type == 'Ride'] # We'll only use rides which have a map


option = st.selectbox(
     'Select a ride for more details',
     (polylines_df.name)
)

# Finding dataframe index based on ride name
idx = polylines_df[polylines_df.name == option].index.values[0]

# Decoding polylines
decoded = pd.json_normalize(polylines_df[polylines_df.index == idx]['map'].apply(ast.literal_eval))['summary_polyline'].apply(polyline.decode).values[0]


# Adding elevation data from Google Elevation API
@st.cache(persist=True, suppress_st_warning=True)

def elev_profile_chart():
    with st.spinner('Calculating elevation profile from Google Elevation. Hang tight...'):
        elevation_profile_feet = [get_elev_data_GOOGLE(coord[0], coord[1]) for coord in decoded]
        # elevation_profile_feet = [elevation_profile[i] * 3.28084 for i in range(len(elevation_profile))] # Converting elevation to feet

        return elevation_profile_feet

elevation_profile_feet = elev_profile_chart()

# Plotting elevation data
# fig, ax = plt.subplots(figsize=(10, 4))
# ax = pd.Series(elevation_profile_feet).rolling(3).mean().plot(
#     ax=ax, 
#     color='steelblue', 
#     legend=False
# )
# ax.set_ylabel('Elevation (ft)')
# ax.axes.xaxis.set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# # Saving plot
# plt.savefig('./data/elevation_profile.png', dpi=300)

# Mapping route with folium



# centroid = [
#     np.mean([coord[0] for coord in decoded]), 
#     np.mean([coord[1] for coord in decoded])
# ]
# my_map = folium.Map(location=centroid, zoom_start=12, tiles='OpenStreetMap')
# folium.PolyLine(decoded).add_to(my_map)

# icon = './icons/pin.png' # icon for ride start location
# icon_image = Image.open(icon)
        
# icon = CustomIcon(
# np.array(icon_image),
# icon_size=(50, 50),
# popup_anchor=(0, -30),
# )

# # popup image
# image_file = './data/elevation_profile.png'
# encoded = base64.b64encode(open(image_file, 'rb').read()).decode('UTF-8')

# resolution, width, height = 50, 5, 6.5

# # read png file
# # elevation_profile = base64.b64encode(open(image_file, 'rb').read()).decode()


# # popup text
# html = """
# <h3 style="font-family:arial">{}</h3>
#     <p style="font-family:arial">
#         <code>
#         Date : {} <br>
#         </code>
#     </p>
#     <p style="font-family:arial"> 
#         <code>
#             Distance&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp: {} miles <br>
#             Elevation Gain&nbsp;&nbsp;&nbsp;&nbsp;&nbsp: {} feet <br>
#             Average Speed&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp: {} mph<br>
#             Average Watts&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp: {} Watts <br>
#             Average HR&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp: {} <br>
#             Suffer Score&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp: {} <br>
#         </code>
#     </p>
# <img src="data:image/png;base64,{}">
# """.format(
#     polylines_df[polylines_df.index == idx]['name'].values[0], 
#     polylines_df[polylines_df.index == idx]['start_date_local'].values[0],
#     polylines_df[polylines_df.index == idx]['distance'].values[0], 
#     polylines_df[polylines_df.index == idx]['total_elevation_gain'].values[0], 
#     polylines_df[polylines_df.index == idx]['average_speed'].values[0], 
#     polylines_df[polylines_df.index == idx]['weighted_average_watts'].values[0],  
#     polylines_df[polylines_df.index == idx]['average_heartrate'].values[0],
#     polylines_df[polylines_df.index == idx]['suffer_score'].values[0], 
#     encoded
# )

# iframe = folium.IFrame(html, width=(width*resolution)+20, height=(height*resolution))
# popup = folium.Popup(iframe, max_width=2650)

# marker = folium.Marker(location=decoded[0],
#                        popup=popup, 
#                        icon=icon).add_to(my_map)

# folium_static(my_map, width=1040)

########################
# Plotly scattermapbox #
########################

centroid = [
    np.mean([coord[0] for coord in decoded]), 
    np.mean([coord[1] for coord in decoded])
]

lat = [coord[0] for coord in decoded] 
lon = [coord[1] for coord in decoded]




fig = go.Figure(go.Scattermapbox(
    mode = "lines",
    lon = lon, lat = lat,
    marker = dict(size = 2, color = "red"),
    line = dict(color = "midnightblue", width = 2),
    # text = '‣',
    textfont=dict(color='#E58606'),
    textposition = 'bottom center',))
fig.update_traces(hovertext='', selector=dict(type='scattermapbox'))
fig.update_layout(
    mapbox = {
        'accesstoken': token,
        'style': "outdoors", 'zoom': 11,
        'center': {'lon': centroid[1], 'lat': centroid[0]}
    },
    margin = {'l': 0, 'r': 0, 't': 0, 'b': 0},
    showlegend = False)

name = polylines_df[polylines_df.index == idx]['name'].values[0]
distance = polylines_df[polylines_df.index == idx]['distance'].values[0]
elev_gain = polylines_df[polylines_df.index == idx]['total_elevation_gain'].values[0]
avg_speed = polylines_df[polylines_df.index == idx]['average_speed'].values[0]
avg_power = polylines_df[polylines_df.index == idx]['weighted_average_watts'].values[0] 
suffer = polylines_df[polylines_df.index == idx]['suffer_score'].values[0]

fig_elev = px.line(elevation_profile_feet, x=range(len(elevation_profile_feet)), y=pd.Series(elevation_profile_feet).rolling(5).mean())
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

fig_elev.add_annotation(text=f"<b>RIDE STATS</b>--------------------", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.99,
)
fig_elev.add_annotation(text=f"<b>Name</b>: {name}", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.95,
)                    
fig_elev.add_annotation(text=f"<b>Distance</b>: {distance} miles", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.91,
)
fig_elev.add_annotation(text=f"<b>Elevation Gain</b>: {elev_gain} feet", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.87,
)
fig_elev.add_annotation(text=f"<b>Average Speed</b>: {avg_speed} mph", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.83,
)     
fig_elev.add_annotation(text=f"<b>Weighted Power</b>: {avg_power} Watts", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.79,
)
fig_elev.add_annotation(text=f"<b>Suffer Score</b>: {suffer.astype(int)}", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.75,
)  
fig_elev.add_annotation(text="----------------------------------", 
                    align='left',
                    showarrow=False,
                    xref='paper',
                    yref='paper',
                    x=0.05,
                    y=0.71,
)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig, use_container_width = True, config=dict(displayModeBar = False))
with col2:

    st.plotly_chart(fig_elev, use_container_width = True, config=dict(displayModeBar = False))

##########################
# HEAT MAP OF ACTIVITIES #
##########################

st.markdown('<h2 style="color:#45738F">Activity Frequency</h2>', unsafe_allow_html=True)

processed_data = processed_data[processed_data.type == activity_type]
def default_fig():
    
    grouped_by_day_hour = processed_data.groupby(['weekday', 'hour']).agg({'id': 'count'}).reset_index()

    # Custom sorting weekdays starting on monday
    day_order = CategoricalDtype(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], ordered=True)
    grouped_by_day_hour['weekday'] = grouped_by_day_hour['weekday'].astype(day_order)
    grouped_by_day_hour.sort_values('weekday', inplace=True)

    # Creating new entries for hours with no data
    mux = pd.MultiIndex.from_product([grouped_by_day_hour.weekday.unique(), range(0,24)], names=['weekday', 'hour'])
    grouped_by_day_hour = grouped_by_day_hour.set_index(['weekday', 'hour']).reindex(mux, fill_value=0).reset_index()

    # Pivoting table to get the number of activities per hour
    pivot_data = grouped_by_day_hour.pivot_table(index='weekday', columns='hour', values='id', aggfunc='sum')
    pivot_data.fillna(0, inplace=True)

    # Creating heatmap of workouts by weekday and hour
    hours_of_day = ['12am', '1am', '2am', '3am', '4am', '5am', '6am', '7am', '8am', '9am', '10am', '11am', '12pm', '1pm', '2pm', '3pm', '4pm', '5pm', '6pm', '7pm', '8pm', '9pm', '10pm', '11pm']

    fig = px.imshow(pivot_data,
    labels=dict(x="", y="", color="Activities"),
    color_continuous_scale='oranges',
    x=hours_of_day,
    aspect="auto"
    )
    fig.update_xaxes(side="top")
    fig.update_layout(
        xaxis_nticks=8,
        showlegend = False,
        # autocolorscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=50, b=10))
    fig.update_coloraxes(showscale=True)

    return fig

processed_data = processed_data[processed_data.type == activity_type]
def switch_graph():
    
    grouped_by_month_day = processed_data.groupby(['month', 'day']).agg({'id': 'count'}).reset_index()
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
    # Custom sorting weekdays starting on monday
    month_order = CategoricalDtype(['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'], ordered=True)
    # Converting month number to month name
    grouped_by_month_day['month'] = grouped_by_month_day['month'].apply(lambda x: months[x-1])
    grouped_by_month_day['month'] = grouped_by_month_day['month'].astype(month_order)
    grouped_by_month_day.sort_values('month', inplace=True)

    # Pivoting table to get the number of activities per hour
    pivot_data_month_day = grouped_by_month_day.pivot_table(index='month', columns='day', values='id', aggfunc='sum')
    pivot_data_month_day.fillna(0, inplace=True)

    fig_1 = px.imshow(pivot_data_month_day,
    labels=dict(x="", y="", color="Activities"),
    color_continuous_scale='oranges',
    # x=hours_of_day,
    aspect="auto"
    )
    fig_1.update_xaxes(side="top")
    fig_1.update_layout(
        title='',
        xaxis_nticks=8,
        showlegend = False,
        plot_bgcolor='rgba(0,0,0,0)')

    return fig_1

view = st.radio(
     "Select view", ('Weekly', 'Monthly'), index=0)

if view == 'Weekly':
    st.plotly_chart(default_fig(), use_container_width = True, config=dict(displayModeBar = False))
else:
    st.plotly_chart(switch_graph(), use_container_width = True, config=dict(displayModeBar = False))

#################################
# Yearly Progressions line chart #
#################################

st.markdown('<h2 style="color:#45738F">Year Progressions and Goals</h2>', unsafe_allow_html=True)

grouped_by_year_and_month = processed_data.groupby(['year', 'month', 'type']).agg({'distance': 'sum', 'total_elevation_gain': 'sum'}).reset_index() # Group by year and month

# Since not all months have data, we're creating entries for missing months and setting the distance and elevation gain to 0
mux = pd.MultiIndex.from_product([grouped_by_year_and_month.year.unique(), grouped_by_year_and_month.type.unique(), range(1,13)], names=['year', 'type' ,'month'])
grouped_by_year_and_month = grouped_by_year_and_month.set_index(['year', 'type', 'month']).reindex(mux, fill_value=0).reset_index()
grouped_by_year_and_month['Cumulative Distance'] = grouped_by_year_and_month.groupby(['year', 'type'])['distance'].cumsum()
grouped_by_year_and_month['Cumulative Elevation'] = grouped_by_year_and_month.groupby(['year', 'type'])['total_elevation_gain'].cumsum()

grouped_by_year_and_month['month'] = grouped_by_year_and_month['month'].apply(lambda x: months[x -1])

# Limiting data to current month
months_left = months[this_month:]

# Filtering out months beyond current one
no_data_yet = grouped_by_year_and_month[grouped_by_year_and_month.year == this_year]
no_data_yet = no_data_yet[no_data_yet.month.isin(months_left)]

# Removing upcoming months with no data from dataframe
grouped_by_year_and_month = grouped_by_year_and_month[~grouped_by_year_and_month.isin(no_data_yet)]
# Dropping na years
grouped_by_year_and_month = grouped_by_year_and_month.dropna(subset=['year'])

grouped_by_year_and_month['year'] = grouped_by_year_and_month['year'].astype(int)



# Plotly charts

selected_year = st.multiselect('Filter by Year', grouped_by_year_and_month.year.unique(), default=[2018, 2019, 2020, 2021, 2022]) # Filter for year
selected_metric = st.selectbox('Metric', ['Cumulative Distance', 'Cumulative Elevation']) # Filter for desired metric

best_distance = grouped_by_year_and_month['Cumulative Distance'].max()
best_distance_year = grouped_by_year_and_month[grouped_by_year_and_month['Cumulative Distance'] == best_distance]['year'].unique()[0]

best_elevation = grouped_by_year_and_month['Cumulative Elevation'].max()
best_elevation_year = grouped_by_year_and_month[grouped_by_year_and_month['Cumulative Elevation'] == best_elevation]['year'].unique()[0]

# Filtering year and activity type
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['type'].isin([activity_type])]
grouped_by_year_and_month = grouped_by_year_and_month[grouped_by_year_and_month['year'].isin(selected_year)]

# Plotly charts
fig = px.line(grouped_by_year_and_month, x='month', y=selected_metric, color='year')
fig.update_layout(
        xaxis=dict(
            showline=True,
            showgrid=True,
            showticklabels=True,
            linecolor='rgb(204, 204, 204)',
            linewidth=1,
            ticks='outside',
            tickfont=dict(
                family='Arial',
                size=12,
                color='rgb(82, 82, 82)',
            ),
        ),
        yaxis=dict(
            showgrid=True,
            zeroline=False,
            showline=False,
            gridcolor = 'rgb(235, 236, 240)',
            showticklabels=True,
            title='',
            autorange=True
        ),
        autosize=True,
        hovermode="x unified",
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title='',
        yaxis_title='Distamce (miles)' if selected_metric == 'Cumulative Distance' else 'Elevation (feet)',
        margin=dict(l=0, r=0, t=0, b=0)
    )
fig.for_each_trace(lambda trace: fig.add_annotation(
    x=trace.x[-1], y=trace.y[-1], text='  '+trace.name, 
    font_color=trace.line.color,
    ax=10, ay=10, xanchor="left", showarrow=False))
fig.update_traces(mode="markers+lines", hovertemplate=None)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.plotly_chart(fig, use_container_width=True, config= dict(
            displayModeBar = False))


##################
# Bests and Goals #
##################

col1, col2 = st.columns(2)

with col1:
    st.metric(f'Most Miles in a Year achieved in {best_distance_year}', "{:,}".format(best_distance) + ' miles')
with col2:
    st.metric(f'Most Elevation Gain in a Year achieved in {best_elevation_year}', "{:,}".format(best_elevation) + ' feet')

previous_year = this_year - 1
# Getting previous year's best distance
previous_best_distance = grouped_by_year_and_month[grouped_by_year_and_month['year'] == previous_year]['Cumulative Distance'].max()
# Getting previous year's best elevation
previous_best_elevation = grouped_by_year_and_month[grouped_by_year_and_month['year'] == previous_year]['Cumulative Elevation'].max()



# Limiting the data to today's date
d0 = dt.datetime(2022, 1, 1)
d1 = dt.datetime.today()
delta = d1 - d0

days_gone_by = delta.days # number of days since the beginning of the year

distance_goal = st.number_input("Choose a distance goal for the year (default is previous year's distance)", value=previous_best_distance) # distance goal for 2022
# st.write('The current distance goal is ', distance_goal)

monthly_goal = distance_goal/12 # monthly distance to reach 2500 miles
daily_goals = distance_goal/365 # daily distance to reach 2500 miles

# Cumulative distance per day
grouped_by_day = processed_data.groupby(['year', 'month', 'day']).agg({'distance': 'sum'}).reset_index()
# Daily cumulative distance
grouped_by_day['Cummulative Distance'] = grouped_by_day.groupby(['year'])['distance'].cumsum()

should_be_reached = daily_goals*days_gone_by # distance that should have been reached by now to be on pace for the desired goal for the year

today_year = dt.datetime.today().year
# print(f"Today's month is the {this_month}th month and year is {today_year}")


where_i_am = grouped_by_day[(grouped_by_day.year == today_year) & (grouped_by_day.month == this_month)]['Cummulative Distance'].max()
# print(f"I should have reached {should_be_reached} miles. I've done {where_i_am} miles")

pace = round(where_i_am - should_be_reached, 1)

col1, col2, = st.columns(2)

with col1:
    st.metric(f'{today_year} Distance Goal', "{:,}".format(distance_goal) + ' miles')
with col2:
    st.metric(f'Distance through {today.strftime("%m/%d/%Y")}', "{:,}".format(round(where_i_am, 1)) + ' miles', f'{pace} ' + 'miles behind' if pace <0 else f'{pace} ' + 'miles ahead')


elevation_goal = st.number_input("Choose an elevation goal for the year (default is previous year's elevation)", value=previous_best_elevation) # distance goal for 2022
# st.write('The current distance goal is ', distance_goal)

monthly_goal_elev = elevation_goal/12 # monthly distance to reach 2500 miles
daily_goals_elev = elevation_goal/365 # daily distance to reach 2500 miles

# Cumulative distance per day
grouped_by_day = processed_data.groupby(['year', 'month', 'day']).agg({'total_elevation_gain': 'sum'}).reset_index()
# Daily cumulative distance
grouped_by_day['Cummulative Elevation'] = grouped_by_day.groupby(['year'])['total_elevation_gain'].cumsum()

should_be_reached_elev = daily_goals_elev*days_gone_by # elevation that should have been reached by now to be on pace to reach the set goal for the year

today_year = dt.datetime.today().year
# print(f"Today's month is the {this_month}th month and year is {today_year}")


where_i_am_elev = grouped_by_day[(grouped_by_day.year == today_year) & (grouped_by_day.month == this_month)]['Cummulative Elevation'].max()
# print(f"I should have reached {should_be_reached} miles. I've done {where_i_am} miles")

pace_elev = round(where_i_am_elev - should_be_reached_elev, 1)

col1, col2, = st.columns(2)

with col1:
    st.metric(f'{today_year} Elevation Goal', "{:,}".format(elevation_goal) + ' feet')
with col2:
    st.metric(f'Elevation Gain through {today.strftime("%m/%d/%Y")}', "{:,}".format(round(where_i_am_elev, 1)) + ' feet', f'{"{:,}".format(pace_elev)} ' + 'feet behind' if pace_elev <0 else f'{"{:,}".format(pace_elev)} ' + 'feet ahead')


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



######################
# PERFORMACE METRICS #
######################

st.markdown('<h2 style="color:#45738F">Performance Insights</h2>', unsafe_allow_html=True)


metric_1 = ['distance', 'moving_time',
       'total_elevation_gain',
       'achievement_count', 'kudos_count', 'comment_count', 'photo_count',
       'average_speed',
       'average_heartrate', 'pr_count', 'total_photo_count',
       'suffer_score', 'average_cadence', 'average_temp', 'average_watts',
       'max_watts', 'weighted_average_watts', 'kilojoules']
metric_2 = ['distance', 'moving_time',
       'total_elevation_gain',
       'achievement_count', 'kudos_count', 'comment_count', 'photo_count',
       'average_speed',
       'average_heartrate', 'pr_count', 'total_photo_count',
       'suffer_score', 'average_cadence', 'average_temp', 'average_watts',
       'max_watts', 'weighted_average_watts', 'kilojoules']

col1, col2 = st.columns(2)

with col1:
    metric_X = st.selectbox('Which metrics would you like to plot?', metric_1, key='metric_X', index=12)
with col2:
    metric_Y = st.selectbox('', metric_2, key='metric_Y', index=14)

processed_data['year'] = processed_data['year'].astype(str)
# Scatterplots
fig = px.scatter(processed_data, x=metric_X, y=metric_Y, size='distance', color='year')

fig.update_layout(
    # title=f'{metric_X} vs {metric_Y}',
    xaxis_title=f'{metric_X}',
    yaxis_title=f'{metric_Y}',
    plot_bgcolor='rgba(0,0,0,0)',
    autosize=True,
    hovermode="x unified",
    showlegend=True,
    margin=dict(l=0, r=0, t=30, b=0),
    yaxis=dict(
    showticklabels=True,
    )
)
st.plotly_chart(fig, use_container_width=True, config= dict(
            displayModeBar = False))
st.caption('Point sizes are relative to ride distance')

