# VISUALIZING STRAVA DATA WITH STREAMLIT AND PLOTLY

### Highlights
- Lifetime stats
- Summary table of all activities
- Ride maps + stats and elevation profile-Google Elevation API 
- Goals and progress charts
- Bike-specific metrics
- Performance insights
- Fitness, Fatigue, and Form (TSB) chart


#### Fitness (CTL), Fatigue (ATL), and Form (TSB):

**1. Training Stress Score (TSS)**

Before diving into Fitness, Fatigue, and Form, let's define TSS, as it's a critical component for these calculations. TSS measures the workload of a workout, considering its duration and intensity, relative to your personal threshold. For cycling, it's commonly calculated using power data:
$$
\text{TSS} = \left( \frac{\text{Activity Duration in seconds} \times \text{Normalized Power (NP)} \times \text{Intensity Factor (IF)}}{\text{Functional Threshold Power (FTP)} \times 3600} \right) \times 100
$$

Normalized Power (NP): An adjusted average power for the ride that accounts for higher intensity efforts being more taxing than steady efforts.
Intensity Factor (IF): The ratio of the normalized power to your Functional Threshold Power (FTP). FTP is the highest power that you can maintain through an hour's effort without fatiguing.
If you're using heart rate data instead of power, the calculation for TSS changes and becomes more subjective, based on heart rate zones and perceived exertion, as there isn't a direct, universally accepted formula like there is with power-based TSS.

**2. Fitness (Chronic Training Load, CTL)**

CTL is calculated as an exponentially weighted moving average (EWMA) of your daily TSS values:
$$
\text{CTL}_{\text{today}} = \text{CTL}_{\text{yesterday}} + \left( \frac{\text{TSS}_{\text{today}} - \text{CTL}_{\text{yesterday}}}{\text{Time Constant}} \right)
$$

Time Constant: Typically 42 days for CTL, representing the long-term, rolling average.

**3. Fatigue (Acute Training Load, ATL)**

ATL is calculated similarly to CTL but over a shorter timeframe:
$$
\text{ATL}_{\text{today}} = \text{ATL}_{\text{yesterday}} + \left( \frac{\text{TSS}_{\text{today}} - \text{ATL}_{\text{yesterday}}}{\text{Time Constant}} \right)
$$

Time Constant: Typically 7 days for ATL, representing the short-term, rolling average.

**4. Form (Training Stress Balance, TSB)**

TSB is calculated by subtracting your ATL from your CTL:
$$
\text{TSB}_{\text{today}} = \text{CTL}_{\text{today}} - \text{ATL}_{\text{today}}
$$

A higher TSB suggests you are well-rested (but possibly detraining if too high for too long), while a lower or negative TSB suggests fatigue.


Calculating Training Stress Score (TSS) depends on whether you are using power or heart rate data for your activities. Here are the general approaches for both:

**1. TSS Calculation Using Power Data**

If you have power data, TSS can be calculated using the following formula:

$$
\text{TSS} = \left( \frac{\text{Workout Duration} \times \text{Normalized Power (NP)} \times \text{Intensity Factor (IF)}}{\text{Functional Threshold Power (FTP)} \times 3600} \right) \times 100
$$

Workout Duration: The total time of the workout in seconds.

Normalized Power (NP): A weighted average of power outputs over the ride, giving more importance to higher power outputs. This requires calculating the fourth power of your power values over time, averaging them, and then taking the fourth root.
Intensity Factor (IF): The ratio of the normalized power (NP) to your Functional Threshold Power (FTP). 

$$
\text{IF} = \frac{\text{NP}}{\text{FTP}}
$$
​
Functional Threshold Power (FTP): The highest power that you can maintain in a quasi-steady state without fatiguing for approximately one hour.

**2. TSS Calculation Using Heart Rate Data**

If you're using heart rate data instead of power, the calculation is more complex and less precise because it's hard to directly correlate heart rate to power output. However, a simplified version can be based on heart rate zones and their relation to perceived exertion levels. There isn't a universally accepted standard formula like there is for power-based TSS, but a general approach is to compare your heart rate during the workout to your Lactate Threshold Heart Rate (LTHR) or a similar benchmark:

$$
\text{TSS} = \left( \frac{\text{Workout Duration} \times \text{Average HR Intensity}}{\text{LTHR}} \right) \times 100
$$

HR Intensity: Can be estimated as the ratio of your average heart rate during the workout to your Lactate Threshold Heart Rate (LTHR).
LTHR: Your average heart rate at the lactate threshold, similar in concept to FTP but for heart rate.


