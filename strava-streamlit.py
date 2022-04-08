#############
# LIBRARIES #
#############

from get_strava_data import my_data, process_data # Function to retrive data using strava api

import pandas as pd
import numpy as np
import datetime as dt


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px







#################################
# ACQUIRING AND PROCESSING DATA #
#################################

all_activities = my_data()
processed_data = process_data(all_activities)

####################################
# BASIC ANALYSIS AND VISUALIZATION #
####################################





