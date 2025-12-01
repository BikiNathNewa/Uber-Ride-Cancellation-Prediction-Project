import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import os

# --- 1. LOAD THE TRAINED MODEL PIPELINE ---
MODEL_FILE = 'cancellation_model_pipeline.joblib'

if os.path.exists(MODEL_FILE):
    pipeline = joblib.load(MODEL_FILE)
    print(f"Model loaded from {MODEL_FILE}")
else:
    print(f"ERROR: {MODEL_FILE} not found. Please ensure the model file is in the same directory.")
    pipeline = None

# --- 2. DEFINE INPUT OPTIONS ---
vehicle_types = ['Bike', 'Go Mini', 'eBike', 'Auto', 'Premier Sedan', 'Uber XL', 'Go Sedan']
zones = ['Gurgaon', 'East Delhi', 'South Delhi', 'Central Delhi', 'West Delhi', 'North Delhi', 'Outer NCR', 'Noida', 
         'Other (Delhi)', 'Ghaziabad', 'Airport Area']
patience_levels = ['new_customer', 'no_cancel_history', 'extremely patient','very patient', 'very_impatient', 'patient']
weather_conditions = ['Clear', 'Mainly Clear/Cloudy', 'Drizzle', 'Rain']
rating_history = ['Historically Good', 'Historically Poor/Avg', 'No Prior Rating']

# --- 3. INITIALIZE DASH APP ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# 'server' is needed for WSGI deployments (like Gunicorn on AWS)
server = app.server 

# --- 4. DEFINE LAYOUT ---
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Uber Ride Cancellation Predictor App", className="text-center mt-4 mb-2"),
            html.H5("By Team - When September Ends", className="text-center text-muted mb-5")
        ], width=12)
    ]),

    dbc.Row([
        # --- LEFT COLUMN: INPUTS ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Ride Details"),
                dbc.CardBody([
                    
                    # Row 1: Vehicle & Weather
                    dbc.Row([
                        dbc.Col([
                            html.Label("Vehicle Type"),
                            dcc.Dropdown(id='vehicle_type', options=vehicle_types, value='Auto')
                        ], width=6),
                        dbc.Col([
                            html.Label("Weather Condition"),
                            dcc.Dropdown(id='weather_condition', options=weather_conditions, value='Clear')
                        ], width=6),
                    ], className="mb-3"),

                    # Row 2: Locations
                    dbc.Row([
                        dbc.Col([
                            html.Label("Pickup Zone"),
                            dcc.Dropdown(id='pickup_zone', options=zones, value='Gurgaon')
                        ], width=6),
                        dbc.Col([
                            html.Label("Drop Zone"),
                            dcc.Dropdown(id='drop_zone', options=zones, value='South Delhi')
                        ], width=6),
                    ], className="mb-3"),

                    # Row 3: Customer Info
                    dbc.Row([
                        dbc.Col([
                            html.Label("Customer Patience"),
                            dcc.Dropdown(id='customer_patience', options=patience_levels, value='patient')
                        ], width=6),
                        dbc.Col([
                            html.Label("History Rating"),
                            dcc.Dropdown(id='historical_rating', options=rating_history, value='No Prior Rating')
                        ], width=6),
                    ], className="mb-3"),

                    # Row 4: Numerical Sliders
                    html.Label("Vehicle Arrival Time (min)"),
                    dcc.Slider(id='arrival_time', min=0, max=60, step=1, value=5, 
                               marks={0:'0', 15:'15', 30:'30', 45:'45', 60:'60'}, className="mb-3"),

                    html.Label("Distance (km)"),
                    dcc.Slider(id='distance', min=0, max=50, step=1, value=10, 
                               marks={0:'0', 10:'10', 25:'25', 50:'50'}, className="mb-3"),
                    
                    html.Label("Ride Cost (₹)"),
                    dcc.Slider(id='cost', min=0, max=2000, step=50, value=300, 
                               marks={0:'0', 500:'500', 1000:'1k', 2000:'2k'}, className="mb-3"),

                    html.Label("Temperature (°C)"),
                    dcc.Slider(id='temp', min=0, max=50, step=1, value=30, 
                               marks={0:'0', 25:'25', 40:'40', 50:'50'}, className="mb-3"),
                    
                    # Hidden defaults
                    html.Div([
                        html.Label("Hour of Day (0-23)"),
                        dcc.Input(id='hour', type='number', value=18, min=0, max=23),
                        html.Label("Humidity (%)"),
                        dcc.Input(id='humidity', type='number', value=50),
                        html.Label("Precipitation (mm)"),
                        dcc.Input(id='precip', type='number', value=0),
                        html.Label("Day of Week"),
                        dcc.Dropdown(id='day', options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], value='Monday'),
                        html.Label("Month"),
                        dcc.Input(id='month', type='number', value=6),
                        html.Label("Day of Month"),
                        dcc.Input(id='day_of_month', type='number', value=15),
                    ], style={'display': 'none'})

                ])
            ], className="shadow-sm")
        ], width=12, lg=6),

        # --- RIGHT COLUMN: PREDICTION ---
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Prediction Result", className="bg-primary text-white"),
                dbc.CardBody([
                    html.H2(id='prediction-text', className="text-center display-4 my-4"),
                    html.Hr(),
                    html.H5("Cancellation Probability:", className="text-center text-muted"),
                    dbc.Progress(id='prob-bar', value=0, striped=True, animated=True, className="mb-3", style={"height": "30px"}),
                    html.P(id='prob-text', className="text-center lead"),
                    
                    dbc.Button("Predict Now", id='btn-predict', color="success", size="lg", className="w-100 mt-3")
                ])
            ], className="shadow-sm h-100")
        ], width=12, lg=6)
    ])

], fluid=True, className="p-5")

# --- 5. CALLBACK FUNCTION ---
@app.callback(
    [Output('prediction-text', 'children'),
     Output('prediction-text', 'className'),
     Output('prob-bar', 'value'),
     Output('prob-bar', 'color'),
     Output('prob-text', 'children')],
    [Input('btn-predict', 'n_clicks')],
    [State('vehicle_type', 'value'),
     State('weather_condition', 'value'),
     State('pickup_zone', 'value'),
     State('drop_zone', 'value'),
     State('customer_patience', 'value'),
     State('historical_rating', 'value'),
     State('arrival_time', 'value'),
     State('distance', 'value'),
     State('cost', 'value'),
     State('temp', 'value'),
     State('hour', 'value'),
     State('humidity', 'value'),
     State('precip', 'value'),
     State('day', 'value'),
     State('month', 'value'),
     State('day_of_month', 'value')]
)
def predict_cancellation(n_clicks, v_type, weather, pickup, drop, patience, rating, 
                         arrival, dist, cost, temp, hour, hum, prec, day, month, dom):
    
    if not n_clicks:
        return "Ready", "text-center text-muted", 0, "info", "Click Predict to start"

    if pipeline is None:
        return "Error", "text-danger", 0, "danger", "Model file not found."

    input_data = pd.DataFrame([{
        'vehicle_arrival_time': arrival,
        'distance': dist,
        'ride_cost': cost,
        'temperature': temp,
        'humidity': hum,
        'precipitation_mm': prec,
        'vehicle_type': v_type,
        'customer_patience': patience,
        'pickup_zone': pickup,
        'drop_zone': drop,
        'historical_customer_rating_binned': rating,
        'hour': hour,
        'day': day,
        'month': month,
        'day_of_month': dom,
        'weather_condition': weather
    }])

    try:
        prob = pipeline.predict_proba(input_data)[0][1]
        prob_pct = round(prob * 100, 2)
        threshold = 0.5
        
        if prob >= threshold:
            pred_text = "Cancelled"
            text_class = "text-center text-danger font-weight-bold"
            bar_color = "danger"
        else:
            pred_text = "Not Cancelled"
            text_class = "text-center text-success font-weight-bold"
            bar_color = "success"

        return pred_text, text_class, prob_pct, bar_color, f"{prob_pct}% Risk"

    except Exception as e:
        return "Error", "text-danger", 0, "danger", f"Prediction failed: {str(e)}"

# --- 6. RUN APP ---
if __name__ == '__main__':
    # host='0.0.0.0' allows connections from outside the server (Internet)
    # port=8050 is the default Dash port
    app.run(debug=False, host='0.0.0.0', port=8050)