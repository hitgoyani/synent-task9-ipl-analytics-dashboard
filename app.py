from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR
MODEL_DIR = PROJECT_ROOT / 'model'
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_PATH = MODEL_DIR / 'model.pkl'
DATASET_PATH = DATA_DIR / 'ipl_live_win_predictor_dataset.csv'


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(
            f"**Model not found:** `{MODEL_PATH}`\n\n"
            "Run the notebook (Phases 1–12) to train and save the model first, then restart the app."
        )
        st.stop()
    model = joblib.load(MODEL_PATH)
    repair_loaded_model(model)
    return model


def repair_loaded_model(estimator):
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps:
            repair_loaded_model(step)
        return
    if isinstance(estimator, ColumnTransformer):
        for _, transformer, _ in estimator.transformers:
            if transformer not in {'drop', 'passthrough'}:
                repair_loaded_model(transformer)
        return
    if isinstance(estimator, SimpleImputer) and not hasattr(estimator, '_fill_dtype'):
        estimator._fill_dtype = getattr(estimator, '_fit_dtype', object)


@st.cache_data
def load_dataset():
    if not DATASET_PATH.exists():
        st.error(
            f"**Dataset not found:** `{DATASET_PATH}`\n\n"
            "Run the notebook (Phases 1–5) to generate the dataset first, then restart the app."
        )
        st.stop()
    return pd.read_csv(DATASET_PATH, dtype={'season': str})


@st.cache_data
def build_season_maps(dataset):
    season_team_map = {}
    season_venue_map = {}
    
    for season in dataset['season'].astype(str).unique():
        sdf = dataset[dataset['season'].astype(str) == season]
        teams = sorted(pd.unique(
            pd.concat([sdf['batting_team'], sdf['bowling_team']])
            .dropna().astype(str)
        ))
        venues = sorted(sdf['venue'].dropna().astype(str).unique().tolist())
        season_team_map[season] = teams
        season_venue_map[season] = venues
    
    return season_team_map, season_venue_map


def season_sort_key(value):
    text = str(value)
    for token in text.split('/'):
        if token.isdigit() and len(token) == 4:
            return int(token)
    digits = ''.join(ch for ch in text if ch.isdigit())
    return int(digits[:4]) if digits else 9999


def normalize_text(value):
    if value is None:
        return ''
    return ' '.join(str(value).strip().split())


def validate_input(target, current_runs, current_wickets, balls_completed, balls_remaining, runs_last_30, wickets_last_30):
    errors = []
    if current_runs >= target:
        errors.append('Current Runs must be less than Target because the chase is already complete.')
    if current_wickets < 0 or current_wickets > 9:
        errors.append('Current Wickets must be between 0 and 9.')
    if balls_completed == 0:
        if runs_last_30 != 0:
            errors.append('Runs Last 30 Balls must be 0 if 0 balls are completed.')
        if wickets_last_30 != 0:
            errors.append('Wickets Last 30 Balls must be 0 if 0 balls are completed.')
    else:
        if runs_last_30 > current_runs:
            errors.append('Runs Last 30 Balls cannot exceed Current Runs.')
        if wickets_last_30 > current_wickets:
            errors.append('Wickets Last 30 Balls cannot exceed Current Wickets.')
    if balls_completed < 0 or balls_completed > 120:
        errors.append('Balls Completed must be between 0 and 120.')
    if balls_remaining < 0 or balls_remaining > 120:
        errors.append('Balls Remaining must be between 0 and 120.')
    if balls_remaining == 0 and current_runs < target:
        errors.append('All 120 balls have been bowled but the target has not been reached — this is a completed (lost) match, not a live chase.')
    if balls_completed > 0 and current_runs > target:
        errors.append('Current Runs exceed Target — the chase is already complete. This is not a live in-progress state.')
    return errors


TEAM_COLORS = {
    'Mumbai Indians':               {'primary': '#005DA0', 'secondary': '#D4A843', 'text': '#FFFFFF'},
    'Chennai Super Kings':          {'primary': '#F5C518', 'secondary': '#0081E9', 'text': '#000000'},
    'Royal Challengers Bangalore':  {'primary': '#C8102E', 'secondary': '#000000', 'text': '#FFFFFF'},
    'Royal Challengers Bengaluru':  {'primary': '#C8102E', 'secondary': '#000000', 'text': '#FFFFFF'},
    'Kolkata Knight Riders':        {'primary': '#3A225D', 'secondary': '#F5A800', 'text': '#FFFFFF'},
    'Delhi Daredevils':             {'primary': '#0078BC', 'secondary': '#EF1C25', 'text': '#FFFFFF'},
    'Delhi Capitals':               {'primary': '#0078BC', 'secondary': '#EF1C25', 'text': '#FFFFFF'},
    'Sunrisers Hyderabad':          {'primary': '#F7A721', 'secondary': '#E8192C', 'text': '#000000'},
    'Rajasthan Royals':             {'primary': '#254AA5', 'secondary': '#FF69B4', 'text': '#FFFFFF'},
    'Kings XI Punjab':              {'primary': '#ED1B24', 'secondary': '#A7A9AC', 'text': '#FFFFFF'},
    'Punjab Kings':                 {'primary': '#ED1B24', 'secondary': '#A7A9AC', 'text': '#FFFFFF'},
    'Gujarat Lions':                {'primary': '#00ACC1', 'secondary': '#F5A623', 'text': '#FFFFFF'},
    'Gujarat Titans':               {'primary': '#1B254B', 'secondary': '#D4A843', 'text': '#FFFFFF'},
    'Lucknow Super Giants':         {'primary': '#0057B8', 'secondary': '#FF6A00', 'text': '#FFFFFF'},
    'Rising Pune Supergiant':       {'primary': '#7B3F9E', 'secondary': '#1EACD5', 'text': '#FFFFFF'},
    'Rising Pune Supergiants':      {'primary': '#7B3F9E', 'secondary': '#1EACD5', 'text': '#FFFFFF'},
    'Deccan Chargers':              {'primary': '#E97320', 'secondary': '#002147', 'text': '#FFFFFF'},
    'Pune Warriors':                {'primary': '#1C4B9C', 'secondary': '#59A608', 'text': '#FFFFFF'},
    'Kochi Tuskers Kerala':         {'primary': '#EE1B26', 'secondary': '#F7A721', 'text': '#FFFFFF'},
}
FALLBACK_COLOR = {'primary': '#f59e0b', 'secondary': '#3b82f6', 'text': '#000000'}


def get_team_color(team_name):
    return TEAM_COLORS.get(team_name, FALLBACK_COLOR)


CSS_VARS = """
:root {
    --bg-base:       #070d1a;
    --bg-surface:    #0d1626;
    --bg-elevated:   #111f38;
    --bg-hover:      #162540;
    --border:        #1a2f4e;
    --border-focus:  #2a4a7f;
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #475569;
    --accent:        #f59e0b;
    --accent-dim:    #f59e0b22;
    --green:         #22c55e;
    --red:           #ef4444;
    --font-display:  'Outfit', sans-serif;
    --font-body:     'Inter', sans-serif;
    --radius-sm:     6px;
    --radius-md:     10px;
    --radius-lg:     14px;
}
"""


def section_header(label):
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:24px 0 16px 0;">
      <div style="color:var(--text-muted);font-size:11px;text-transform:uppercase;
           letter-spacing:0.1em;font-family:var(--font-body);font-weight:500;">{label}</div>
      <div style="flex:1;height:1px;background:var(--border);"></div>
    </div>
    """, unsafe_allow_html=True)


# Initialize Streamlit Page
st.set_page_config(
    page_title='IPL Live Win Probability Predictor',
    page_icon='🏏',
    layout='wide'
)

# Apply CSS Design System Tokens
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Inter:wght@400;500;600&display=swap');

{CSS_VARS}

* {{ font-family: var(--font-body); }}
h1, h2, h3, h4 {{ font-family: var(--font-display) !important; }}

/* Main background */
.stApp {{ background-color: var(--bg-base); }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, var(--bg-surface) 0%, var(--bg-elevated) 100%) !important;
    border-right: 1px solid var(--border) !important;
}}
section[data-testid="stSidebar"] * {{ color: var(--text-primary) !important; }}
section[data-testid="stSidebar"] caption,
section[data-testid="stSidebar"] p {{ color: var(--text-secondary) !important; }}

/* Hide default header */
header[data-testid="stHeader"] {{ background: transparent; }}

/* Metric cards styling */
div[data-testid="stMetric"] {{
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    padding: 16px 20px;
    border-top: 1px solid var(--border) !important;
}}
div[data-testid="stMetric"] label {{ color: var(--text-secondary) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.05em; font-family: var(--font-body) !important; }}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: var(--accent) !important; font-family: var(--font-display) !important; font-size: 28px !important; font-weight: 700 !important; }}

/* Selectbox, number input, slider labels */
div[data-testid="stWidgetLabel"] p {{ color: var(--text-secondary) !important; font-size: 12px !important; text-transform: uppercase; letter-spacing: 0.04em; }}

/* Input fields */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}}

/* Slider */
div[data-testid="stSlider"] div[role="slider"] {{ background: var(--accent) !important; }}

/* Primary button */
div[data-testid="stButton"] > button {{
    background: linear-gradient(135deg, var(--accent), #d97706) !important;
    color: #000 !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 14px !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s ease !important;
}}
div[data-testid="stButton"] > button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 20px rgba(245,158,11,0.4) !important;
}}

/* Expander */
div[data-testid="stExpander"] {{
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}}

/* Divider */
hr {{ border-color: var(--border) !important; }}

/* Info/error boxes */
div[data-testid="stAlert"] {{ border-radius: var(--radius-md) !important; }}

/* Tab styling */
button[data-baseweb="tab"] {{ color: var(--text-secondary) !important; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: var(--accent) !important; border-bottom-color: var(--accent) !important; }}

/* Sidebar Radio Group styling to look like links */
section[data-testid="stSidebar"] div[role="radiogroup"] {{
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 0;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label {{
    padding: 8px 16px !important;
    background-color: transparent !important;
    border-radius: var(--radius-sm) !important;
    border-left: 4px solid transparent !important;
    margin: 0 !important;
    transition: all 0.2s ease;
    cursor: pointer;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
    background-color: var(--bg-hover) !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label[data-checked="true"] {{
    border-left: 4px solid var(--accent) !important;
    background-color: var(--bg-elevated) !important;
}}
section[data-testid="stSidebar"] div[role="radiogroup"] label [data-baseweb="radio"] div:first-child {{
    display: none !important;
}}
</style>
""", unsafe_allow_html=True)

# Load model and dataset
model = load_model()
dataset = load_dataset()

# Setup maps and stats
season_team_map, season_venue_map = build_season_maps(dataset)
available_seasons = sorted(dataset['season'].dropna().astype(str).unique(), key=season_sort_key)
match_level_df = dataset.sort_values(['date', 'match_id']).drop_duplicates('match_id', keep='first').copy()
all_teams_list = sorted(pd.unique(pd.concat([dataset['batting_team'], dataset['bowling_team']]).dropna().astype(str)))

total_matches = len(match_level_df)
total_teams = len(all_teams_list)
min_season = min(available_seasons, key=season_sort_key)
max_season = max(available_seasons, key=season_sort_key)

# Session state initialization
if 'page' not in st.session_state:
    st.session_state.page = 'Live Predictor'
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Sidebar header with single allowed brand icon styled small
st.sidebar.markdown("""
<div style="padding:8px 0 16px 0;">
  <div style="font-family:var(--font-display); font-size:20px; font-weight:800;
       color:var(--accent);">🏏 IPL Predictor</div>
  <div style="color:var(--text-muted); font-size:11px; margin-top:2px;">
    Official Cricsheet Data
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation radio
st.sidebar.radio('PAGES', ['Live Predictor', 'Team & Venue Analytics'], key='page')

# Sidebar metadata
st.sidebar.markdown(f"""
<div style="margin: 20px 0; border-top: 1px solid var(--border); padding-top: 16px;">
  <div style="color:var(--text-muted); font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; font-family:var(--font-body);">DATA</div>
  <div style="font-size:12px; color:var(--text-secondary); line-height:1.8; margin-top:8px;">
    Seasons: {min_season}–{max_season}<br>
    {total_matches} Matches<br>
    {total_teams} Teams
  </div>
</div>
<div style="margin: 20px 0; border-top: 1px solid var(--border); padding-top: 16px;">
  <div style="color:var(--text-muted); font-size:11px; font-weight:500; text-transform:uppercase; letter-spacing:0.1em; font-family:var(--font-body);">MODEL</div>
  <div style="font-size:12px; color:var(--text-secondary); line-height:1.8; margin-top:8px;">
    Type: Classification<br>
    Output: Win Probability
  </div>
</div>
<div style="margin: 20px 0; border-top: 1px solid var(--border); padding-top: 16px;">
  <div style="font-size:11px; color:var(--text-muted); line-height:1.6;">
    <span style="color:var(--accent); font-weight:600;">Hit Goyani</span><br>
    Synent Technologies<br>
    SYN/M2/IP1050
  </div>
</div>
""", unsafe_allow_html=True)

# Page routing
if st.session_state.page == 'Live Predictor':
    # Default season selection
    season = st.selectbox('Season', available_seasons, index=len(available_seasons) - 1)
    
    current_teams = season_team_map.get(str(season), [])
    current_venues = season_venue_map.get(str(season), [])
    
    # Render layout columns
    col_context, col_situation = st.columns(2)
    
    with col_context:
        section_header("Match Context")
        venue = st.selectbox('Venue', current_venues)
        batting_team = st.selectbox('Batting Team', current_teams)
        
        # Exclude batting team from bowling options
        bowling_options = [t for t in current_teams if t != batting_team]
        if not bowling_options:
            bowling_options = current_teams
        bowling_team = st.selectbox('Bowling Team', bowling_options)
        target = st.number_input('Target', min_value=1, value=150, step=1)
        
    with col_situation:
        section_header("Chase Situation")
        overs_completed = st.slider('Overs Completed', min_value=0, max_value=19, value=12, step=1)
        balls_in_current_over = st.slider('Balls in Current Over', min_value=0, max_value=5, value=0, step=1)
        current_runs = st.number_input('Current Runs', min_value=0, value=80, step=1)
        current_wickets = st.slider('Current Wickets', min_value=0, max_value=9, value=2, step=1)
        
        # Runs and Wickets last 30 legal deliveries
        runs_last_30 = st.number_input('Runs Last 30 Balls', min_value=0, value=28, step=1)
        wickets_last_30 = st.slider('Wickets Last 30 Balls', min_value=0, max_value=9, value=1, step=1)

    # Compute values
    balls_completed = overs_completed * 6 + balls_in_current_over
    balls_remaining = max(0, 120 - balls_completed)
    runs_required = max(0, target - current_runs)
    current_run_rate = (current_runs / (balls_completed / 6)) if balls_completed > 0 else 0.0
    required_run_rate = (runs_required / (balls_remaining / 6)) if balls_remaining > 0 else 0.0

    # Team colors & dynamic header banner
    bat_color = get_team_color(batting_team)['primary']
    bowl_color = get_team_color(bowling_team)['primary']
    
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#0a1628,#0d1f3c,#0a1e40);
         border-left:4px solid {bat_color}; border-radius:var(--radius-lg);
         padding:20px 28px; margin-bottom:20px; margin-top:20px; border-top: 1px solid var(--border); border-right: 1px solid var(--border); border-bottom: 1px solid var(--border);">
      <div style="font-size:11px;color:var(--text-muted);text-transform:uppercase;
           letter-spacing:0.1em;margin-bottom:8px;font-family:var(--font-body);font-weight:500;">Live Win Probability Predictor</div>
      <div style="font-family:var(--font-display);font-size:26px;font-weight:800;
           color:var(--text-primary);">
        <span style="color:{bat_color};">{batting_team}</span>
        <span style="color:var(--text-muted);font-weight:300;margin:0 12px;">vs</span>
        <span style="color:{bowl_color};">{bowling_team}</span>
      </div>
      <div style="color:var(--text-muted);font-size:13px;margin-top:6px;font-family:var(--font-body);">
        {venue} &nbsp;·&nbsp; Season {season}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Live Metrics Bar
    section_header("Live Metrics")
    
    # RRR Color coding
    if required_run_rate <= current_run_rate:
        rrr_color = "#22c55e"
    elif required_run_rate <= current_run_rate + 2:
        rrr_color = "#f59e0b"
    else:
        rrr_color = "#ef4444"
        
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin: 16px 0;">
      <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px; text-align: center;">
        <div style="color: var(--text-secondary); font-size: 11px; text-transform: uppercase; font-family: var(--font-body); font-weight: 500;">Current Run Rate</div>
        <div style="color: var(--accent); font-family: var(--font-display); font-size: 28px; font-weight: 700; margin-top: 4px;">{current_run_rate:.2f}</div>
      </div>
      <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px; text-align: center;">
        <div style="color: var(--text-secondary); font-size: 11px; text-transform: uppercase; font-family: var(--font-body); font-weight: 500;">Required Run Rate</div>
        <div style="color: {rrr_color}; font-family: var(--font-display); font-size: 28px; font-weight: 700; margin-top: 4px;">{required_run_rate:.2f}</div>
      </div>
      <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px; text-align: center;">
        <div style="color: var(--text-secondary); font-size: 11px; text-transform: uppercase; font-family: var(--font-body); font-weight: 500;">Balls Left</div>
        <div style="color: var(--accent); font-family: var(--font-display); font-size: 28px; font-weight: 700; margin-top: 4px;">{balls_remaining}</div>
      </div>
      <div style="background: var(--bg-surface); border: 1px solid var(--border); border-radius: var(--radius-md); padding: 16px; text-align: center;">
        <div style="color: var(--text-secondary); font-size: 11px; text-transform: uppercase; font-family: var(--font-body); font-weight: 500;">Runs Needed</div>
        <div style="color: var(--accent); font-family: var(--font-display); font-size: 28px; font-weight: 700; margin-top: 4px;">{runs_required}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Dynamic CSS injection for predict button color based on batting team
    st.markdown(f"""
    <style>
    div[data-testid="stButton"] > button {{
        background: linear-gradient(135deg, {bat_color}cc, {bat_color}) !important;
        color: {'#000' if get_team_color(batting_team)['text'] == '#000000' else '#fff'} !important;
        width: 100% !important;
        border: none !important;
    }}
    div[data-testid="stButton"] > button:hover {{
        background: linear-gradient(135deg, {bat_color}, {bat_color}cc) !important;
        box-shadow: 0 4px 20px {bat_color}44 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

    current_inputs = {
        'season': season,
        'venue': venue,
        'batting_team': batting_team,
        'bowling_team': bowling_team,
        'target': target,
        'current_runs': current_runs,
        'current_wickets': current_wickets,
        'overs_completed': overs_completed,
        'balls_in_current_over': balls_in_current_over,
        'runs_last_30': runs_last_30,
        'wickets_last_30': wickets_last_30
    }

    # If the inputs have changed since the last prediction, clear the last prediction
    if 'last_prediction' in st.session_state:
        if st.session_state.last_prediction['inputs'] != current_inputs:
            del st.session_state.last_prediction

    validation_errors = validate_input(
        target,
        current_runs,
        current_wickets,
        balls_completed,
        balls_remaining,
        runs_last_30,
        wickets_last_30
    )

    if validation_errors:
        for error in validation_errors:
            st.error(error)
    else:
        if st.button('Predict Win Probability'):
            feature_frame = pd.DataFrame([
                {
                    'season': normalize_text(season),
                    'venue': normalize_text(venue),
                    'batting_team': normalize_text(batting_team),
                    'bowling_team': normalize_text(bowling_team),
                    'target': target,
                    'current_runs': current_runs,
                    'current_wickets': current_wickets,
                    'balls_completed': balls_completed,
                    'balls_remaining': balls_remaining,
                    'runs_required': runs_required,
                    'current_run_rate': current_run_rate,
                    'required_run_rate': required_run_rate,
                    'runs_last_30_legal_deliveries': runs_last_30,
                    'wickets_last_30_legal_deliveries': wickets_last_30,
                }
            ])
            probabilities = model.predict_proba(feature_frame)[0]
            batting_probability = float(probabilities[1])
            bowling_probability = float(probabilities[0])

            # Prepend to session history
            st.session_state.prediction_history.insert(0, {
                'teams': f"{batting_team} vs {bowling_team}",
                'situation': f"{overs_completed}.{balls_in_current_over} ov | {current_runs}/{current_wickets}",
                'bat_prob': batting_probability,
                'bowl_prob': bowling_probability,
                'season': season
            })
            st.session_state.prediction_history = st.session_state.prediction_history[:5]

            # Save prediction to session state
            st.session_state.last_prediction = {
                'inputs': current_inputs,
                'bat_prob': batting_probability,
                'bowl_prob': bowling_probability
            }

    # Render prediction outputs if a prediction has been made and is not stale
    if 'last_prediction' in st.session_state:
        batting_probability = st.session_state.last_prediction['bat_prob']
        bowling_probability = st.session_state.last_prediction['bowl_prob']

        # Gauges
        def make_gauge(title, probability, color):
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=round(probability * 100, 1),
                number={'suffix': '%', 'font': {'size': 48, 'color': color, 'family': 'Outfit'}},
                title={'text': title, 'font': {'size': 16, 'color': '#94a3b8', 'family': 'Inter'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#1a2f4e', 'tickwidth': 1},
                    'bar': {'color': color, 'thickness': 0.25},
                    'bgcolor': 'var(--bg-surface)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 100], 'color': 'var(--bg-surface)'}
                    ],
                    'threshold': {
                        'line': {'color': color, 'width': 3},
                        'thickness': 0.8,
                        'value': round(probability * 100, 1)
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='#070d1a',
                plot_bgcolor='#070d1a',
                font_color='#f1f5f9',
                height=280,
                margin=dict(t=40, b=10, l=20, r=20)
            )
            return fig

        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(make_gauge(batting_team, batting_probability, bat_color), use_container_width=True)
        with g2:
            st.plotly_chart(make_gauge(bowling_team, bowling_probability, bowl_color), use_container_width=True)

        # Winner Banner styled based on winning team primary color
        winner = batting_team if batting_probability > bowling_probability else bowling_team
        winner_color = get_team_color(winner)['primary']
        win_pct = max(batting_probability, bowling_probability)
        
        st.markdown(f"""
        <div style="background:var(--bg-surface); border:1px solid {winner_color}; border-radius:var(--radius-md);
             padding:20px; text-align:center; margin-top:12px; margin-bottom: 20px;">
          <div style="font-family:var(--font-display); font-size:22px; font-weight:700; color:{winner_color};">
            {winner} is more likely to win
          </div>
          <div style="color:var(--text-secondary); font-size:14px; margin-top:4px;">
            Win probability: {win_pct:.1%}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature Importance
        try:
            classifier = model.named_steps['classifier']
            preprocessor = model.named_steps['preprocessor']
            importances = classifier.feature_importances_
            feature_names = preprocessor.get_feature_names_out()
            
            aggregated = {}
            for name, imp in zip(feature_names, importances):
                if name.startswith('categorical__'):
                    col_name = name[len('categorical__'):]
                    for cat in ['season', 'venue', 'batting_team', 'bowling_team']:
                        if col_name.startswith(cat + '_'):
                            aggregated[cat] = aggregated.get(cat, 0.0) + imp
                            break
                elif name.startswith('numeric__'):
                    col_name = name[len('numeric__'):]
                    aggregated[col_name] = aggregated.get(col_name, 0.0) + imp
            
            section_header("Prediction Factors")
            
            friendly_names = {
                'required_run_rate': 'Required Run Rate',
                'target': 'Target Score',
                'venue': 'Venue Matchup',
                'season': 'Season Context',
                'batting_team': 'Batting Team Power',
                'bowling_team': 'Bowling Team Power',
                'current_wickets': 'Wickets Fallen',
                'runs_required': 'Runs Required',
                'current_run_rate': 'Current Run Rate',
                'wickets_last_30_legal_deliveries': 'Wickets Last 30 Balls',
                'current_runs': 'Current Score Progress',
                'runs_last_30_legal_deliveries': 'Runs Last 30 Balls',
                'balls_remaining': 'Balls Remaining',
                'balls_completed': 'Balls Completed',
            }
            
            sorted_imps = sorted(aggregated.items(), key=lambda x: x[1], reverse=True)
            top_n = sorted_imps[:8]
            labels = [friendly_names.get(k, k) for k, _ in top_n][::-1]
            values = [float(v) * 100 for _, v in top_n][::-1]

            fig_imp = go.Figure()
            fig_imp.add_trace(go.Bar(
                y=labels,
                x=values,
                orientation='h',
                marker_color=bat_color,
                marker_line_width=0,
                hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
            ))
            fig_imp.update_layout(
                paper_bgcolor='#070d1a',
                plot_bgcolor='#0d1626',
                font_color='#f1f5f9',
                xaxis=dict(
                    title='Relative Influence %',
                    color='#94a3b8',
                    gridcolor='#1a2f4e',
                    showgrid=True,
                ),
                yaxis=dict(
                    color='#f1f5f9',
                    gridcolor='#1a2f4e',
                ),
                height=320,
                margin=dict(t=10, b=40, l=150, r=20)
            )
            st.plotly_chart(fig_imp, use_container_width=True)
        except AttributeError:
            pass

        # Head to Head history panel
        h2h_all = match_level_df[
            ((match_level_df['batting_team'] == batting_team) & (match_level_df['bowling_team'] == bowling_team)) |
            ((match_level_df['batting_team'] == bowling_team) & (match_level_df['bowling_team'] == batting_team))
        ].copy()
        
        if not h2h_all.empty:
            section_header("Head-to-Head History")
            h2h_scope = st.radio(
                'Head-to-Head Scope', 
                ['All Time', f'Season {season} only'], 
                horizontal=True
            )
            
            if h2h_scope == f'Season {season} only':
                h2h = h2h_all[h2h_all['season'].astype(str) == str(season)].copy()
            else:
                h2h = h2h_all.copy()

            total_h2h = len(h2h)
            if total_h2h > 0:
                bat_wins = int(
                    ((h2h['batting_team'] == batting_team) & (h2h['target_label'] == 1)).sum() +
                    ((h2h['bowling_team'] == batting_team) & (h2h['target_label'] == 0)).sum()
                )
                bowl_wins = int(
                    ((h2h['batting_team'] == bowling_team) & (h2h['target_label'] == 1)).sum() +
                    ((h2h['bowling_team'] == bowling_team) & (h2h['target_label'] == 0)).sum()
                )
                avg_target = float(h2h['target'].mean()) if total_h2h > 0 else 0.0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Matches", total_h2h)
                c2.metric(f"{batting_team} Wins", bat_wins)
                c3.metric(f"{bowling_team} Wins", bowl_wins)
                c4.metric("Avg Target Score", f"{avg_target:.0f}" if avg_target > 0 else "N/A")
            else:
                st.info(f"No matchups recorded between these teams for {h2h_scope.lower()}.")

    # Always render session history at bottom of page if predictions exist
    if st.session_state.prediction_history:
        section_header("Recent Predictions")
        history_df = pd.DataFrame(st.session_state.prediction_history)
        display_hist = history_df.rename(columns={
            'teams': 'Match',
            'situation': 'Situation',
            'bat_prob': 'Bat Win %',
            'bowl_prob': 'Bowl Win %'
        })[['Match', 'Situation', 'Bat Win %', 'Bowl Win %']]
        
        display_hist['Bat Win %'] = display_hist['Bat Win %'].apply(lambda x: f"{x*100:.1f}%")
        display_hist['Bowl Win %'] = display_hist['Bowl Win %'].apply(lambda x: f"{x*100:.1f}%")

        def style_history(row):
            bat_p = history_df.loc[row.name, 'bat_prob']
            bowl_p = history_df.loc[row.name, 'bowl_prob']
            bat_style = 'color: var(--green); font-weight: bold;' if bat_p > bowl_p else 'color: var(--red);'
            bowl_style = 'color: var(--green); font-weight: bold;' if bowl_p > bat_p else 'color: var(--red);'
            return ['', '', bat_style, bowl_style]

        with st.expander("Show Recent Predictions", expanded=True):
            st.dataframe(
                display_hist.style.apply(style_history, axis=1),
                use_container_width=True,
                hide_index=True
            )

elif st.session_state.page == 'Team & Venue Analytics':
    # Define tabs
    tab_team, tab_venue = st.tabs(["Team Analytics", "Venue Analytics"])
    
    with tab_team:
        team = st.selectbox('Select Team', all_teams_list)
        team_color = get_team_color(team)['primary']
        
        # Swatch dot and title in a row
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 24px; margin-top: 10px;">
          <div style="width: 14px; height: 14px; border-radius: 50%; background-color: {team_color}; border: 1px solid var(--border);"></div>
          <h2 style="margin: 0; padding: 0; font-family: var(--font-display); color: var(--text-primary);">{team}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # CSS override for metric cards border-top and expander left border on Page 2
        st.markdown(f"""
        <style>
        div[data-testid="stMetric"] {{
            border-top: 4px solid {team_color} !important;
        }}
        div[data-testid="stExpander"] {{
            border-left: 4px solid {team_color} !important;
            border-top: 1px solid var(--border) !important;
            border-right: 1px solid var(--border) !important;
            border-bottom: 1px solid var(--border) !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Calculate career metrics
        team_matches = match_level_df[(match_level_df['batting_team'] == team) | (match_level_df['bowling_team'] == team)].copy()
        team_wins = int(((team_matches['batting_team'] == team) & (team_matches['target_label'] == 1)).sum() + ((team_matches['bowling_team'] == team) & (team_matches['target_label'] == 0)).sum())
        team_losses = int(len(team_matches) - team_wins)
        matches_played = int(len(team_matches))
        win_percentage = (team_wins / matches_played * 100) if matches_played else 0.0

        # Avg score when batting first
        first_innings_batting_first = team_matches[team_matches['bowling_team'] == team]['target'].sub(1)
        average_first_innings_score = float(first_innings_batting_first.mean()) if not first_innings_batting_first.empty else 0.0

        # Chase Stats
        chases_played = int((team_matches['batting_team'] == team).sum())
        chases_won = int(((team_matches['batting_team'] == team) & (team_matches['target_label'] == 1)).sum())
        chase_success_rate = (chases_won / chases_played * 100) if chases_played else 0.0

        # Defending Stats
        defends_played = int((team_matches['bowling_team'] == team).sum())
        defends_won = int(((team_matches['bowling_team'] == team) & (team_matches['target_label'] == 0)).sum())
        defend_success_rate = (defends_won / defends_played * 100) if defends_played else 0.0

        # Highest Successful Chase & Lowest Defended Target
        successful_chases = team_matches[(team_matches['batting_team'] == team) & (team_matches['target_label'] == 1)]
        highest_successful_chase = int(successful_chases['target'].max() - 1) if not successful_chases.empty else 0

        successful_defends = team_matches[(team_matches['bowling_team'] == team) & (team_matches['target_label'] == 0)]
        lowest_defended_score = int(successful_defends['target'].min() - 1) if not successful_defends.empty else 0

        # Recent Form
        last_5_matches = team_matches.sort_values(['date', 'match_id'], ascending=False).head(5)
        form_list = []
        for _, r in last_5_matches.iterrows():
            is_win = (r['batting_team'] == team and r['target_label'] == 1) or (r['bowling_team'] == team and r['target_label'] == 0)
            form_list.append('W' if is_win else 'L')
        form_str = ' '.join(form_list[::-1]) if form_list else 'N/A'

        # Render 12 metrics cards in a 3x4 grid
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        r1c1.metric("Matches Played", matches_played)
        r1c2.metric("Wins", team_wins)
        r1c3.metric("Losses", team_losses)
        r1c4.metric("Win Rate", f"{win_percentage:.1f}%")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        r2c1.metric("Chase Success Rate", f"{chase_success_rate:.1f}%")
        r2c2.metric("Chases (Won/Played)", f"{chases_won}/{chases_played}")
        r2c3.metric("Defending Success Rate", f"{defend_success_rate:.1f}%")
        r2c4.metric("Defends (Won/Played)", f"{defends_won}/{defends_played}")

        r3c1, r3c2, r3c3, r3c4 = st.columns(4)
        r3c1.metric("Avg 1st Innings Score", f"{average_first_innings_score:.0f}")
        r3c2.metric("Highest Chased Target", f"{highest_successful_chase}" if highest_successful_chase > 0 else "N/A")
        r3c3.metric("Lowest Defended Score", f"{lowest_defended_score}" if lowest_defended_score > 0 else "N/A")
        r3c4.metric("Recent Form (Last 5)", form_str)

        st.write("")

        # Form Guide (Last 10 matches pills)
        last_10_matches = team_matches.sort_values(['date', 'match_id'], ascending=False).head(10)
        form_pills = []
        for _, r in last_10_matches.iterrows():
            is_win = (r['batting_team'] == team and r['target_label'] == 1) or (r['bowling_team'] == team and r['target_label'] == 0)
            p_color = "#22c55e" if is_win else "#ef4444"
            p_text = "W" if is_win else "L"
            form_pills.append(
                f'<span style="display: inline-block; background-color: {p_color}22; color: {p_color}; '
                f'border: 1px solid {p_color}; border-radius: var(--radius-sm); '
                f'width: 28px; height: 28px; line-height: 26px; text-align: center; '
                f'font-weight: 700; font-family: var(--font-display); margin-right: 6px;">'
                f'{p_text}</span>'
            )
        form_html = "".join(form_pills[::-1]) if form_pills else "N/A"
        
        section_header("Form Guide (Last 10 Matches)")
        st.markdown(f'<div style="margin: 12px 0 24px 0;">{form_html}</div>', unsafe_allow_html=True)

        # Season-wise win rate chart
        season_stats = []
        for s, sdf in match_level_df.groupby('season'):
            tm = sdf[(sdf['batting_team'] == team) | (sdf['bowling_team'] == team)]
            played = len(tm)
            if played == 0:
                continue
            wins = int(
                ((tm['batting_team'] == team) & (tm['target_label'] == 1)).sum() +
                ((tm['bowling_team'] == team) & (tm['target_label'] == 0)).sum()
            )
            season_stats.append({'Season': str(s), 'Win Rate': round(wins/played*100, 1), 'Wins': wins, 'Played': played})

        if season_stats:
            sdf_plot = pd.DataFrame(season_stats)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=sdf_plot['Season'], y=sdf_plot['Win Rate'],
                marker_color=team_color, marker_line_width=0,
                hovertemplate='Season %{x}<br>Win Rate: %{y}%<br><extra></extra>'
            ))
            fig.update_layout(
                title=dict(text=f'{team} — Season-wise Win Rate', font=dict(color='#f1f5f9', size=16, family='var(--font-display)')),
                paper_bgcolor='#070d1a', plot_bgcolor='#0d1626',
                xaxis=dict(color='#94a3b8', gridcolor='#1a2f4e'),
                yaxis=dict(color='#94a3b8', gridcolor='#1a2f4e', title='Win Rate %', range=[0,100]),
                height=320, margin=dict(t=50, b=40, l=50, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        # Strongest & Hardest Opponents
        opponents_stats = []
        for opp in all_teams_list:
            if opp == team:
                continue
            m = team_matches[(team_matches['batting_team'] == opp) | (team_matches['bowling_team'] == opp)]
            played = len(m)
            if played == 0:
                continue
            wins = int(
                ((m['batting_team'] == team) & (m['target_label'] == 1)).sum() +
                ((m['bowling_team'] == team) & (m['target_label'] == 0)).sum()
            )
            opponents_stats.append({
                'Opponent': opp,
                'Win Rate': round(wins / played * 100, 1),
                'Played': played,
                'Wins': wins
            })
        
        if opponents_stats:
            opp_df = pd.DataFrame(opponents_stats).sort_values('Win Rate', ascending=True)
            col_easy, col_hard = st.columns(2)
            
            with col_easy:
                section_header("Easiest Opponents")
                easy_df = opp_df.sort_values('Win Rate', ascending=False).head(5)
                fig_easy = go.Figure()
                fig_easy.add_trace(go.Bar(
                    y=easy_df['Opponent'], x=easy_df['Win Rate'],
                    orientation='h',
                    marker_color='#22c55e', marker_line_width=0,
                    hovertemplate='%{y}<br>Win Rate: %{x}%<br><extra></extra>'
                ))
                fig_easy.update_layout(
                    paper_bgcolor='#070d1a', plot_bgcolor='#0d1626',
                    font_color='#f1f5f9',
                    xaxis=dict(color='#94a3b8', gridcolor='#1a2f4e', range=[0, 100]),
                    yaxis=dict(color='#f1f5f9', gridcolor='#1a2f4e', autorange="reversed"),
                    height=220, margin=dict(t=10, b=30, l=120, r=10)
                )
                st.plotly_chart(fig_easy, use_container_width=True)
                
            with col_hard:
                section_header("Hardest Opponents")
                hard_df = opp_df.sort_values('Win Rate', ascending=True).head(5)
                fig_hard = go.Figure()
                fig_hard.add_trace(go.Bar(
                    y=hard_df['Opponent'], x=hard_df['Win Rate'],
                    orientation='h',
                    marker_color='#ef4444', marker_line_width=0,
                    hovertemplate='%{y}<br>Win Rate: %{x}%<br><extra></extra>'
                ))
                fig_hard.update_layout(
                    paper_bgcolor='#070d1a', plot_bgcolor='#0d1626',
                    font_color='#f1f5f9',
                    xaxis=dict(color='#94a3b8', gridcolor='#1a2f4e', range=[0, 100]),
                    yaxis=dict(color='#f1f5f9', gridcolor='#1a2f4e', autorange="reversed"),
                    height=220, margin=dict(t=10, b=30, l=120, r=10)
                )
                st.plotly_chart(fig_hard, use_container_width=True)

        # Best and Worst Venues
        venue_stats = []
        for ven, vdf in team_matches.groupby('venue'):
            played = len(vdf)
            if played < 3:
                continue
            wins = int(
                ((vdf['batting_team'] == team) & (vdf['target_label'] == 1)).sum() +
                ((vdf['bowling_team'] == team) & (vdf['target_label'] == 0)).sum()
            )
            venue_stats.append({
                'Venue': ven,
                'Played': played,
                'Wins': wins,
                'Win Rate': round(wins / played * 100, 1)
            })
            
        if venue_stats:
            section_header("Venue Performance Summary")
            v_df = pd.DataFrame(venue_stats).sort_values('Win Rate', ascending=False).reset_index(drop=True)
            st.dataframe(
                v_df.style.apply(
                    lambda col: ['color: var(--green)' if v >= 60.0 else ('color: var(--red)' if v <= 40.0 else 'color: var(--text-secondary)') for v in col]
                    if col.name == 'Win Rate' else ['' for _ in col],
                    axis=0
                ),
                use_container_width=True,
                hide_index=True
            )

        # Match History by Season
        section_header("Match History by Season")
        for s in sorted(team_matches['season'].astype(str).unique(), key=season_sort_key, reverse=True):
            season_rows = team_matches[team_matches['season'].astype(str) == s].copy()
            season_rows['Opponent'] = season_rows.apply(
                lambda r: r['bowling_team'] if r['batting_team'] == team else r['batting_team'], axis=1
            )
            season_rows['Result'] = season_rows.apply(
                lambda r: 'Won (Chasing)' if (r['batting_team'] == team and r['target_label'] == 1) else
                          ('Won (Defending)' if (r['bowling_team'] == team and r['target_label'] == 0) else
                           ('Lost (Chasing)' if (r['batting_team'] == team and r['target_label'] == 0) else 'Lost (Defending)')), axis=1
            )
            display_df = season_rows[['date', 'Opponent', 'venue', 'target', 'Result']].rename(columns={
                'date': 'Date', 'venue': 'Venue', 'target': 'Target'
            }).reset_index(drop=True)

            wins_in_season = display_df['Result'].str.startswith('Won').sum()
            total_in_season = len(display_df)

            with st.expander(f"Season {s}  —  {wins_in_season}W / {total_in_season - wins_in_season}L  ({total_in_season} matches)"):
                st.dataframe(
                    display_df.style.apply(
                        lambda col: ['color: var(--green)' if 'Won' in str(v) else 'color: var(--red)' for v in col]
                        if col.name == 'Result' else ['' for _ in col],
                        axis=0
                    ),
                    use_container_width=True,
                    hide_index=True
                )

    with tab_venue:
        all_venues_list = sorted(dataset['venue'].dropna().astype(str).unique())
        venue = st.selectbox('Select Venue', all_venues_list)
        
        # Reset border-top for Venue Analytics Tab
        st.markdown(f"""
        <style>
        div[data-testid="stMetric"] {{
            border-top: 1px solid var(--border) !important;
        }}
        </style>
        """, unsafe_allow_html=True)

        venue_matches = match_level_df[match_level_df['venue'] == venue].copy()
        total_venue_matches = len(venue_matches)
        avg_target_venue = float(venue_matches['target'].mean()) if total_venue_matches > 0 else 0.0
        chasing_wins_venue = int((venue_matches['target_label'] == 1).sum())
        chasing_win_pct_venue = (chasing_wins_venue / total_venue_matches * 100) if total_venue_matches > 0 else 0.0
        active_seasons_venue = venue_matches['season'].nunique()
        chases_per_season_venue = (total_venue_matches / active_seasons_venue) if active_seasons_venue > 0 else 0.0

        # Venue metric cards
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Total Matches", total_venue_matches)
        vc2.metric("Avg Target Score", f"{avg_target_venue:.0f}" if avg_target_venue > 0 else "N/A")
        vc3.metric("Chasing Win Rate", f"{chasing_win_pct_venue:.1f}%")
        vc4.metric("Average Matches / Season", f"{chases_per_season_venue:.1f}")

        # Venue Season Chasing Win Rate Chart
        venue_season_stats = []
        for s, sdf in venue_matches.groupby('season'):
            played = len(sdf)
            if played == 0:
                continue
            wins = int((sdf['target_label'] == 1).sum())
            venue_season_stats.append({
                'Season': str(s),
                'Win Rate': round(wins / played * 100, 1),
                'Wins': wins,
                'Played': played
            })
        
        if venue_season_stats:
            vs_df = pd.DataFrame(venue_season_stats)
            fig_vs = go.Figure()
            fig_vs.add_trace(go.Bar(
                x=vs_df['Season'], y=vs_df['Win Rate'],
                marker_color='#f59e0b', marker_line_width=0,
                hovertemplate='Season %{x}<br>Chasing Win Rate: %{y}%<br><extra></extra>'
            ))
            fig_vs.update_layout(
                title=dict(text='Chasing Win Rate by Season', font=dict(color='#f1f5f9', size=16, family='var(--font-display)')),
                paper_bgcolor='#070d1a', plot_bgcolor='#0d1626',
                xaxis=dict(color='#94a3b8', gridcolor='#1a2f4e'),
                yaxis=dict(color='#94a3b8', gridcolor='#1a2f4e', title='Win Rate %', range=[0, 100]),
                height=320, margin=dict(t=50, b=40, l=50, r=20)
            )
            st.plotly_chart(fig_vs, use_container_width=True)

        # Venue match history
        section_header("Match History at Venue")
        display_venue_df = venue_matches.copy()
        display_venue_df['Matchup'] = display_venue_df.apply(
            lambda r: f"{r['batting_team']} vs {r['bowling_team']}", axis=1
        )
        display_venue_df['Result'] = display_venue_df.apply(
            lambda r: f"{r['batting_team']} won chasing" if r['target_label'] == 1 else f"{r['bowling_team']} won defending", axis=1
        )
        display_venue_table = display_venue_df[['date', 'Matchup', 'target', 'Result']].rename(columns={
            'date': 'Date', 'target': 'Target'
        }).sort_values('Date', ascending=False).reset_index(drop=True)
        
        st.dataframe(
            display_venue_table.style.apply(
                lambda col: ['color: var(--green)' if 'chasing' in str(v) else 'color: var(--accent)' for v in col]
                if col.name == 'Result' else ['' for _ in col],
                axis=0
            ),
            use_container_width=True,
            hide_index=True
        )

# Clean, no-emojis caption footer
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:var(--text-muted); font-size:11px; padding:8px 0; font-family:var(--font-body);">
  Built by Hit Goyani &nbsp;|&nbsp;
  Synent Technologies Data Science Internship &nbsp;|&nbsp; SYN/M2/IP1050
</div>
""", unsafe_allow_html=True)
