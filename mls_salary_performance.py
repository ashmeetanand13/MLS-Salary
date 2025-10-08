import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import unicodedata
from difflib import SequenceMatcher
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="MLS Salary vs Performance Analysis", 
    page_icon="⚽", 
    layout="wide"
)

# Title and description
st.title("⚽ MLS Salary vs Performance Analysis")
st.markdown("Analyze how MLS player performance in one season affects their salary in the next season")

# Initialize session state
if 'salary_df' not in st.session_state:
    st.session_state.salary_df = None
if 'performance_df' not in st.session_state:
    st.session_state.performance_df = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Function to load data from GitHub
@st.cache_data
def load_data_from_github():
    """Load CSV files directly from GitHub"""
    try:
        # GitHub raw URLs for the CSV files
        performance_url = "https://raw.githubusercontent.com/ashmeetanand13/MLS-Salary/main/mls_23_24.csv"
        salary_url = "https://raw.githubusercontent.com/ashmeetanand13/MLS-Salary/main/mlspa_salary.csv"
        
        # Load the data
        performance_df = pd.read_csv(performance_url)
        salary_df = pd.read_csv(salary_url)
        
        return salary_df, performance_df, True
    except Exception as e:
        st.error(f"Error loading data from GitHub: {str(e)}")
        return None, None, False

def normalize_name(name):
    """Normalize player names for matching"""
    if pd.isna(name) or name == '':
        return ''
    
    # Convert to lowercase and remove accents
    name = str(name).lower()
    name = unicodedata.normalize('NFD', name)
    name = re.sub(r'[\u0300-\u036f]', '', name)  # Remove diacritical marks
    
    # Remove special characters except spaces
    name = re.sub(r'[^a-z\s]', '', name)
    
    return name.strip()


@st.cache_data
def create_name_mapping(salary_df, performance_df, threshold=0.85):
    """Create a mapping between salary and performance names with team validation"""
    name_mapping = {}
    
    # Get unique player-team combinations
    salary_players = salary_df[['full_name', 'club_standard']].drop_duplicates()
    performance_players = performance_df[['Player', 'Squad']].drop_duplicates()
    
    for _, perf_row in performance_players.iterrows():
        perf_name = perf_row['Player']
        perf_team = perf_row['Squad']
        norm_perf = normalize_name(perf_name)
        
        best_match = None
        best_score = 0
        
        for _, sal_row in salary_players.iterrows():
            sal_name = sal_row['full_name']
            sal_team = sal_row['club_standard']
            norm_sal = normalize_name(sal_name)
            
            # Calculate name similarity
            name_score = SequenceMatcher(None, norm_perf, norm_sal).ratio()
            
            # Check if teams match (normalize team names too)
            norm_perf_team = normalize_name(perf_team)
            norm_sal_team = normalize_name(sal_team)
            team_match = norm_perf_team == norm_sal_team
            
            # Boost score if teams match, or require higher threshold if they don't
            if name_score >= threshold:
                if team_match:
                    # Same team - accept lower name threshold
                    if name_score >= threshold * 0.9:  # 10% more lenient
                        if name_score > best_score:
                            best_score = name_score
                            best_match = sal_name
                else:
                    # Different teams - require exact threshold
                    if name_score > best_score:
                        best_score = name_score
                        best_match = sal_name
        
        if best_match:
            name_mapping[perf_name] = best_match
    
    return name_mapping

def calculate_percentile_rank(df, column):
    """Calculate percentile rank for a column"""
    return df[column].rank(pct=True, method='average') * 100

def calculate_position_specific_score(row, all_players_df, use_percentiles=True):
    """
    Calculate comprehensive position-specific performance scores
    using multiple relevant metrics for each position
    """
    pos = row.get('general_position', 'UNKNOWN')
    
    # Get minutes played for weighting
    minutes = row.get('Playing Time Min', 0) or 0
    if minutes < 100:  # Minimal playing time
        return 0
    
    # Minutes weight (players with more minutes are more valuable)
    minutes_weight = min(minutes / 2000, 1.0)  # Max out at 2000 minutes
    
    # Initialize score
    score = 0
    
    try:
        if pos == 'FW':
            # FORWARDS - Focus on attacking output (10+ metrics)
            # Primary metrics (high weight)
            goals = row.get('Performance Gls', 0) or 0
            goals_90 = row.get('Per 90 Minutes Gls', 0) or 0
            assists = row.get('Performance Ast', 0) or 0
            assists_90 = row.get('Per 90 Minutes Ast', 0) or 0
            
            # Expected metrics
            xg = row.get('Expected xG', 0) or 0
            xag = row.get('Expected xAG', 0) or 0
            xg_90 = row.get('Per 90 Minutes xG', 0) or 0
            
            # Shooting efficiency
            shots = row.get('Standard Sh', 0) or 0
            shots_on_target = row.get('Standard SoT', 0) or 0
            shot_accuracy = (shots_on_target / shots * 100) if shots > 0 else 0
            goals_per_shot = (goals / shots * 100) if shots > 0 else 0
            
            # Creative and progression
            key_passes = row.get('KP', 0) or 0
            sca = row.get('SCA SCA', 0) or 0  # Shot creating actions
            gca = row.get('GCA GCA', 0) or 0  # Goal creating actions
            
            # Dribbling and carries
            successful_dribbles = row.get('Take-Ons Succ', 0) or 0
            progressive_carries = row.get('Carries PrgC', 0) or 0
            carries_into_box = row.get('Carries CPA', 0) or 0
            
            # Team contribution
            plus_minus = row.get('Team Success +/-', 0) or 0
            
            if use_percentiles and len(all_players_df) > 0:
                # Calculate percentiles for forwards only
                fw_df = all_players_df[all_players_df['general_position'] == 'FW']
                if len(fw_df) > 0:
                    # Create percentile scores
                    percentiles = {}
                    for col in ['Per 90 Minutes Gls', 'Per 90 Minutes Ast', 'Expected xG', 
                               'SCA SCA', 'GCA GCA', 'Take-Ons Succ', 'Carries PrgC']:
                        if col in fw_df.columns:
                            percentiles[col] = calculate_percentile_rank(fw_df, col)
                    
                    # Use percentile of current player
                    player_idx = fw_df.index[fw_df.index == row.name][0] if row.name in fw_df.index else None
                    if player_idx is not None:
                        score = (
                            percentiles.get('Per 90 Minutes Gls', pd.Series()).get(player_idx, 0) * 0.25 +
                            percentiles.get('Per 90 Minutes Ast', pd.Series()).get(player_idx, 0) * 0.15 +
                            percentiles.get('Expected xG', pd.Series()).get(player_idx, 0) * 0.15 +
                            percentiles.get('SCA SCA', pd.Series()).get(player_idx, 0) * 0.15 +
                            percentiles.get('GCA GCA', pd.Series()).get(player_idx, 0) * 0.10 +
                            percentiles.get('Take-Ons Succ', pd.Series()).get(player_idx, 0) * 0.10 +
                            percentiles.get('Carries PrgC', pd.Series()).get(player_idx, 0) * 0.10
                        )
            else:
                # Raw score calculation
                score = (
                    goals_90 * 100 +           # Goals are primary for FW
                    assists_90 * 60 +           # Assists important
                    xg_90 * 50 +               # Expected goals
                    shot_accuracy * 0.5 +       # Shooting efficiency
                    goals_per_shot * 2 +        # Conversion rate
                    (sca / 10) * 20 +          # Shot creation
                    (gca / 5) * 30 +           # Goal creation
                    successful_dribbles * 2 +   # Dribbling ability
                    progressive_carries * 1.5 + # Ball progression
                    carries_into_box * 3 +     # Penalty box entries
                    plus_minus * 0.5           # Team contribution
                )
        
        elif pos == 'MF':
            # MIDFIELDERS - Balance of creation, progression, and defense (10+ metrics)
            # Passing and creation
            assists = row.get('Performance Ast', 0) or 0
            assists_90 = row.get('Per 90 Minutes Ast', 0) or 0
            xag = row.get('Expected xAG', 0) or 0
            key_passes = row.get('KP', 0) or 0
            pass_completion = row.get('Total Cmp%', 0) or 0
            
            # Progression
            progressive_passes = row.get('PrgP', 0) or 0
            progressive_carries = row.get('Carries PrgC', 0) or 0
            progressive_receptions = row.get('Progression PrgR', 0) or 0
            
            # Defensive contribution
            tackles = row.get('Tackles Tkl', 0) or 0
            interceptions = row.get('Performance Int', 0) or 0
            recoveries = row.get('Performance Recov', 0) or 0
            
            # Ball carrying
            successful_dribbles = row.get('Take-Ons Succ', 0) or 0
            carries = row.get('Carries Carries', 0) or 0
            
            # Shooting (secondary for MF)
            goals = row.get('Performance Gls', 0) or 0
            goals_90 = row.get('Per 90 Minutes Gls', 0) or 0
            shots = row.get('Standard Sh', 0) or 0
            
            # Creative actions
            sca = row.get('SCA SCA', 0) or 0
            gca = row.get('GCA GCA', 0) or 0
            
            # Team contribution
            plus_minus = row.get('Team Success +/-', 0) or 0
            
            if use_percentiles and len(all_players_df) > 0:
                mf_df = all_players_df[all_players_df['general_position'] == 'MF']
                if len(mf_df) > 0:
                    score = calculate_mf_percentile_score(row, mf_df)
            else:
                # Raw score calculation
                score = (
                    assists_90 * 80 +              # Assists very important for MF
                    goals_90 * 60 +                # Goals still valuable
                    xag * 40 +                     # Expected assists
                    (key_passes / 5) * 30 +        # Chance creation
                    pass_completion * 0.8 +        # Passing accuracy
                    progressive_passes * 2 +        # Forward passing
                    progressive_carries * 2.5 +     # Ball carrying forward
                    progressive_receptions * 1.5 + # Getting into good positions
                    (tackles + interceptions) * 2 + # Defensive work
                    recoveries * 1 +               # Ball recovery
                    successful_dribbles * 1.5 +    # Dribbling
                    (sca / 10) * 15 +             # Shot creation
                    (gca / 5) * 20 +              # Goal creation
                    plus_minus * 0.8              # Team contribution
                )
        
        elif pos == 'DF':
            # DEFENDERS - Focus on defensive metrics and build-up (10+ metrics)
            # Core defensive
            tackles = row.get('Tackles Tkl', 0) or 0
            tackles_won = row.get('Performance TklW', 0) or 0
            tackle_success = (tackles_won / tackles * 100) if tackles > 0 else 0
            interceptions = row.get('Performance Int', 0) or 0
            blocks = row.get('Blocks Blocks', 0) or 0
            clearances = row.get('Clr', 0) or 0
            recoveries = row.get('Performance Recov', 0) or 0
            
            # Aerial ability
            aerials_won = row.get('Aerial Duels Won', 0) or 0
            aerial_win_pct = row.get('Aerial Duels Won%', 0) or 0
            
            # Passing and progression
            pass_completion = row.get('Total Cmp%', 0) or 0
            progressive_passes = row.get('PrgP', 0) or 0
            long_pass_completion = row.get('Long Cmp%', 0) or 0
            
            # Carries and progression
            progressive_carries = row.get('Carries PrgC', 0) or 0
            carries = row.get('Carries Carries', 0) or 0
            
            # Discipline and errors
            yellows = row.get('Performance CrdY', 0) or 0
            reds = row.get('Performance CrdR', 0) or 0
            errors = row.get('Err', 0) or 0
            fouls = row.get('Performance Fls', 0) or 0
            
            # Attacking contribution (secondary)
            goals = row.get('Performance Gls', 0) or 0
            assists = row.get('Performance Ast', 0) or 0
            
            # Team metrics
            plus_minus = row.get('Team Success +/-', 0) or 0
            clean_sheets_proxy = row.get('Team Success onGA', 0) or 0  # Lower is better
            
            if use_percentiles and len(all_players_df) > 0:
                df_df = all_players_df[all_players_df['general_position'] == 'DF']
                if len(df_df) > 0:
                    score = calculate_df_percentile_score(row, df_df)
            else:
                # Raw score calculation
                score = (
                    tackles_won * 4 +               # Successful tackles
                    tackle_success * 0.5 +          # Tackle accuracy
                    interceptions * 4 +             # Reading the game
                    blocks * 3 +                    # Blocking shots/passes
                    clearances * 1.5 +              # Defensive clearances
                    recoveries * 1 +                # Ball recoveries
                    aerials_won * 2 +               # Aerial dominance
                    aerial_win_pct * 0.3 +          # Aerial success rate
                    pass_completion * 0.5 +         # Passing accuracy
                    progressive_passes * 1.5 +       # Ball progression
                    progressive_carries * 2 +        # Carrying forward
                    goals * 30 +                    # Rare but valuable goals
                    assists * 25 +                  # Rare but valuable assists
                    plus_minus * 1.5 -              # Team success
                    (yellows * 5) -                 # Avoid bookings
                    (reds * 50) -                   # Really avoid reds
                    (errors * 20) -                 # Minimize errors
                    (clean_sheets_proxy * 0.5)     # Clean sheet contribution
                )
        
        elif pos == 'GK':
            # GOALKEEPERS - Specialized metrics (10+ metrics)
            # We need to handle GK differently as many field metrics don't apply
            # Using team defensive metrics as proxy
            goals_against = row.get('Team Success onGA', 0) or 0
            plus_minus = row.get('Team Success +/-', 0) or 0
            xg_against = row.get('Team Success (xG) onxGA', 0) or 0
            
            # Games and minutes for consistency
            games = row.get('Playing Time MP', 0) or 0
            minutes_gk = row.get('Playing Time Min', 0) or 0
            
            # Estimate saves and clean sheets from team data
            # This is approximate since we don't have GK-specific stats
            goals_against_per90 = (goals_against / minutes_gk * 90) if minutes_gk > 0 else 99
            
            # Passing ability (modern GKs)
            pass_completion = row.get('Total Cmp%', 0) or 0
            long_passes = row.get('Long Att', 0) or 0
            
            if use_percentiles and len(all_players_df) > 0:
                gk_df = all_players_df[all_players_df['general_position'] == 'GK']
                if len(gk_df) > 0:
                    score = calculate_gk_percentile_score(row, gk_df)
            else:
                # GK score based on team defensive performance
                score = (
                    max(0, 100 - goals_against) * 2 +     # Fewer goals = better
                    plus_minus * 2 +                       # Team success
                    max(0, 100 - goals_against_per90 * 20) + # Goals against per 90
                    games * 2 +                            # Reliability/availability
                    (minutes_gk / 100) * 2 +              # Playing time
                    pass_completion * 0.3                 # Distribution ability
                )
        
        else:
            # Unknown position - use generic scoring
            score = (
                row.get('Per 90 Minutes Gls', 0) * 50 +
                row.get('Per 90 Minutes Ast', 0) * 50 +
                row.get('Team Success +/-', 0) * 1
            )
        
        # Apply minutes weight to give preference to regular starters
        score = score * minutes_weight
        
    except Exception as e:
        st.warning(f"Error calculating score for player: {e}")
        score = 0
    
    return max(0, score)

def calculate_mf_percentile_score(row, mf_df):
    """Calculate percentile-based score for midfielders"""
    # Implementation would be similar to forwards but with MF-specific metrics
    # This is a simplified version
    return 50  # Placeholder

def calculate_df_percentile_score(row, df_df):
    """Calculate percentile-based score for defenders"""
    # Implementation would be similar but with defensive metrics
    return 50  # Placeholder

def calculate_gk_percentile_score(row, gk_df):
    """Calculate percentile-based score for goalkeepers"""
    # Implementation for GK-specific percentiles
    return 50  # Placeholder

@st.cache_data(ttl=600)
def process_data(_salary_df, _performance_df, fuzzy_threshold=0.85, min_minutes_filter=450, use_percentiles=True):
    """Process and merge salary and performance data with year-to-year matching"""
    
    # Create copies to avoid modifying cached data
    salary_df = _salary_df.copy()
    performance_df = _performance_df.copy()
    
    # Position mapping
    position_mapping = {
        'CB': 'DF', 'LB': 'DF', 'RB': 'DF', 'LWB': 'DF', 'RWB': 'DF', 'WB': 'DF',
        'CDM': 'MF', 'CM': 'MF', 'CAM': 'MF', 'LM': 'MF', 'RM': 'MF', 'AM': 'MF', 'DM': 'MF',
        'ST': 'FW', 'LW': 'FW', 'RW': 'FW', 'CF': 'FW', 'W': 'FW',
        'GK': 'GK'
    }
    
    # Get unique names for mapping
    salary_names = salary_df['full_name'].dropna().unique().tolist()
    performance_names = performance_df['Player'].dropna().unique().tolist()
    
    # Create name mapping
    name_mapping = create_name_mapping(salary_df, performance_df, fuzzy_threshold)
    
    # Apply name mapping to performance data
    performance_df['matched_name'] = performance_df['Player'].map(name_mapping).fillna(performance_df['Player'])
    
    # Prepare salary data
    salary_df['normalized_name'] = salary_df['full_name'].apply(normalize_name)
    salary_df['general_position'] = salary_df['position_code'].map(position_mapping).fillna('UNKNOWN')
    
    # Prepare performance data
    performance_df['normalized_name'] = performance_df['matched_name'].apply(normalize_name)
    
    # Map positions from performance data
    pos_mapping_perf = {
        'DF': 'DF', 'MF': 'MF', 'FW': 'FW', 'GK': 'GK', 
        'DF,MF': 'DF', 'MF,FW': 'MF', 'FW,MF': 'FW', 
        'DF,GK': 'GK', 'MF,DF': 'MF'
    }
    performance_df['general_position'] = performance_df['Pos'].map(pos_mapping_perf).fillna('UNKNOWN')
    
    # Create two datasets: 2023 performance -> 2024 salary, 2024 performance -> 2025 salary
    merged_data = []
    
    # 1. Match 2023 performance with 2024 salary
    perf_2023 = performance_df[performance_df['Season'] == 2023].copy()
    salary_2024 = salary_df[salary_df['salary_year'] == 2024].copy()
    
    if len(perf_2023) > 0 and len(salary_2024) > 0:
        merged_2023_24 = perf_2023.merge(
            salary_2024[['normalized_name', 'guaranteed_compensation', 'base_salary', 
                         'club_standard', 'position_code', 'general_position']],
            on='normalized_name',
            how='inner',
            suffixes=('_perf', '_sal')
        )
        merged_2023_24['analysis_year'] = '2023 Performance → 2024 Salary'
        merged_data.append(merged_2023_24)
    
    # 2. Match 2024 performance with 2025 salary
    perf_2024 = performance_df[performance_df['Season'] == 2024].copy()
    salary_2025 = salary_df[salary_df['salary_year'] == 2025].copy()
    
    if len(perf_2024) > 0 and len(salary_2025) > 0:
        merged_2024_25 = perf_2024.merge(
            salary_2025[['normalized_name', 'guaranteed_compensation', 'base_salary', 
                         'club_standard', 'position_code', 'general_position']],
            on='normalized_name',
            how='inner',
            suffixes=('_perf', '_sal')
        )
        merged_2024_25['analysis_year'] = '2024 Performance → 2025 Salary'
        merged_data.append(merged_2024_25)
    
    # Combine both datasets
    if merged_data:
        merged = pd.concat(merged_data, ignore_index=True)
        
        # Use salary position if performance position is unknown
        merged['general_position'] = merged.apply(
            lambda x: x['general_position_sal'] if x['general_position_perf'] == 'UNKNOWN' 
            else x['general_position_perf'], axis=1
        )
        
        # Filter qualified players
        qualified = merged[
            (merged['Playing Time MP'] >= 5) & 
            (merged['Playing Time Min'] >= min_minutes_filter) & 
            (merged['guaranteed_compensation'] > 0)
        ].copy()
        
        # Calculate performance scores with position-specific metrics
        qualified['performance_score'] = qualified.apply(
            lambda row: calculate_position_specific_score(row, qualified, use_percentiles), 
            axis=1
        )
        
        # Calculate value ratio (performance per $100k)
        qualified['value_ratio'] = qualified['performance_score'] / (qualified['guaranteed_compensation'] / 100000)
        
        # Add display name
        qualified['display_name'] = qualified['matched_name']
        
        return qualified
    else:
        return pd.DataFrame()

# Display position-specific metrics explanation
def show_scoring_methodology():
    """Display the scoring methodology for each position"""
    with st.expander("📊 Performance Scoring Methodology"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ⚽ Forwards (FW)
            **Primary Metrics (High Weight):**
            - Goals & Goals per 90
            - Assists & Assists per 90
            - Expected Goals (xG)
            - Shot Creating Actions (SCA)
            - Goal Creating Actions (GCA)
            
            **Secondary Metrics:**
            - Shot accuracy & conversion rate
            - Progressive carries
            - Successful dribbles
            - Penalty area entries
            - Team +/- contribution
            
            ### 🎯 Midfielders (MF)
            **Creation & Progression:**
            - Assists & Expected Assists (xAG)
            - Key passes
            - Progressive passes & carries
            - Pass completion %
            
            **Defensive Contribution:**
            - Tackles & Interceptions
            - Ball recoveries
            - Defensive duels won
            """)
        
        with col2:
            st.markdown("""
            ### 🛡️ Defenders (DF)
            **Defensive Excellence:**
            - Tackles won & success rate
            - Interceptions
            - Blocks & Clearances
            - Aerial duels won & %
            - Ball recoveries
            
            **Build-up Play:**
            - Pass completion %
            - Progressive passes & carries
            - Long pass accuracy
            
            **Discipline:**
            - Cards (negative weight)
            - Errors (negative weight)
            - Clean sheet contribution
            
            ### 🧤 Goalkeepers (GK)
            **Team Defensive Metrics:**
            - Goals against
            - Goals against per 90
            - Team +/- rating
            - Games played (reliability)
            - Distribution accuracy
            """)

# Main app logic
if salary_df and performance_df:
    try:
        # Load data with caching
        with st.spinner("Loading data files..."):
            if st.session_state.salary_df is None or st.session_state.performance_df is None:
                salary_df, performance_df = load_data_from_github(salary_file, performance_file)
                st.session_state.salary_df = salary_df
                st.session_state.performance_df = performance_df
            else:
                salary_df = st.session_state.salary_df
                performance_df = st.session_state.performance_df
        
        # Display initial data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"📊 {len(salary_df)} salary records")
            st.caption(f"Years: {sorted(salary_df['salary_year'].unique())}")
        with col2:
            st.info(f"⚽ {len(performance_df)} performance records")
            st.caption(f"Seasons: {sorted(performance_df['Season'].unique())}")
        with col3:
            # Show scoring methodology
            show_scoring_methodology()
        
        # Process data with caching
        with st.spinner("Processing and matching player data..."):
            if st.session_state.processed_data is None:
                merged_df = process_data(
                    salary_df, 
                    performance_df, 
                    fuzzy_threshold, 
                    min_minutes,
                    use_percentiles
                )
                st.session_state.processed_data = merged_df
            else:
                merged_df = st.session_state.processed_data
        
        if len(merged_df) == 0:
            st.error("No matching players found between datasets. Please check your data files.")
        else:
            st.success(f"✅ Successfully matched {len(merged_df)} player-season combinations")
            
            # Show matching statistics
            with st.expander("🔗 Name Matching Statistics"):
                total_salary_players = salary_df['full_name'].nunique()
                total_perf_players = performance_df['Player'].nunique()
                matched_players = merged_df['display_name'].nunique()
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Salary Dataset", f"{total_salary_players} players")
                with col2:
                    st.metric("Performance Dataset", f"{total_perf_players} players")
                with col3:
                    st.metric("Matched Players", f"{matched_players} players")
                with col4:
                    match_rate = (matched_players / min(total_salary_players, total_perf_players)) * 100
                    st.metric("Match Rate", f"{match_rate:.1f}%")
            
            # Year selection
            st.header("📅 Select Analysis Period")
            year_options = merged_df['analysis_year'].unique()
            selected_years = st.multiselect(
                "Choose which year transitions to analyze:",
                options=year_options,
                default=year_options
            )
            
            # Filter by selected years
            analysis_df = merged_df[merged_df['analysis_year'].isin(selected_years)]
            
            # Main dashboard metrics
            st.header("📊 Overall Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                unique_players = analysis_df['display_name'].nunique()
                st.metric("Unique Players", unique_players)
            with col2:
                avg_salary = analysis_df['guaranteed_compensation'].mean()
                st.metric("Avg Salary", f"${avg_salary:,.0f}")
            with col3:
                avg_performance = analysis_df['performance_score'].mean()
                st.metric("Avg Performance Score", f"{avg_performance:.1f}")
            with col4:
                correlation = analysis_df['performance_score'].corr(analysis_df['guaranteed_compensation'])
                st.metric("Performance-Salary Correlation", f"{correlation:.3f}")
            
            # Position analysis with detailed metrics
            st.header("🏃 Position-Based Analysis")
            
            # Calculate position stats with more metrics
            position_stats = analysis_df.groupby('general_position').agg({
                'guaranteed_compensation': ['count', 'mean', 'median'],
                'performance_score': ['mean', 'std', 'max'],
                'value_ratio': ['mean', 'std'],
                'Playing Time Min': 'mean',
                'Per 90 Minutes Gls': 'mean',
                'Per 90 Minutes Ast': 'mean',
                'Per 90 Minutes xG': 'mean',
                'Tackles Tkl': 'mean',
                'Performance Int': 'mean',
                'PrgP': 'mean',
                'Team Success +/-': 'mean'
            }).round(2)
            
            # Flatten column names
            position_stats.columns = ['_'.join(col).strip() for col in position_stats.columns]
            position_stats.columns = [
                'Count', 'Avg Salary', 'Median Salary',
                'Avg Score', 'Score StdDev', 'Max Score',
                'Avg Value', 'Value StdDev', 'Avg Minutes',
                'Goals/90', 'Assists/90', 'xG/90',
                'Tackles', 'Interceptions', 'Prog Passes', 'Team +/-'
            ]
            
            # Format for display
            display_stats = position_stats.copy()
            for col in ['Avg Salary', 'Median Salary']:
                display_stats[col] = display_stats[col].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(
                display_stats.style.highlight_max(subset=['Avg Score', 'Avg Value'], color='lightgreen')
                                  .highlight_min(subset=['Avg Score', 'Avg Value'], color='lightcoral'),
                use_container_width=True
            )
            
            # Visualizations
            st.header("📈 Performance vs Salary Analysis")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Scatter Plot", 
                "Position Distribution", 
                "Top Players", 
                "Team Analysis", 
                "Performance Breakdown"
            ])
            
            with tab1:
    # Position Deep Dive
                st.subheader("📍 Position Deep Dive Analysis")
                
                # Position selector
                selected_position = st.selectbox(
                    "Select Position to Analyze:",
                    options=['FW', 'MF', 'DF', 'GK'],
                    format_func=lambda x: {'FW': 'Forwards', 'MF': 'Midfielders', 'DF': 'Defenders', 'GK': 'Goalkeepers'}[x]
                )
                
                # Filter data for selected position
                position_data = analysis_df[analysis_df['general_position'] == selected_position].copy()
                
                if len(position_data) > 0:
                    # Calculate medians for quadrant splits
                    median_salary = position_data['guaranteed_compensation'].median()
                    median_performance = position_data['performance_score'].median()
                    
                    # Assign quadrants
                    position_data['quadrant'] = position_data.apply(
                        lambda row: 'Stars' if row['guaranteed_compensation'] >= median_salary and row['performance_score'] >= median_performance
                        else 'Bargains' if row['guaranteed_compensation'] < median_salary and row['performance_score'] >= median_performance
                        else 'Overpaid' if row['guaranteed_compensation'] >= median_salary and row['performance_score'] < median_performance
                        else 'Developing',
                        axis=1
                    )
                    
                    # QUADRANT CHART
                    st.markdown("### Quadrant Analysis")
                    
                    # Show quadrant counts
                    col1, col2, col3, col4 = st.columns(4)
                    quadrant_counts = position_data['quadrant'].value_counts()
                    
                    with col1:
                        st.metric("⭐ Stars", quadrant_counts.get('Stars', 0))
                        st.caption("High salary, high performance")
                    with col2:
                        st.metric("💎 Bargains", quadrant_counts.get('Bargains', 0))
                        st.caption("Low salary, high performance")
                    with col3:
                        st.metric("💸 Overpaid", quadrant_counts.get('Overpaid', 0))
                        st.caption("High salary, low performance")
                    with col4:
                        st.metric("🌱 Developing", quadrant_counts.get('Developing', 0))
                        st.caption("Low salary, low performance")
                    
                    # Create quadrant scatter plot
                    fig_quadrant = px.scatter(
                        position_data,
                        x='guaranteed_compensation',
                        y='performance_score',
                        color='quadrant',
                        hover_data=['display_name', 'club_standard', 'Playing Time Min', 'value_ratio'],
                        title=f'{selected_position} Players: Salary vs Performance Quadrants',
                        labels={
                            'guaranteed_compensation': 'Guaranteed Compensation ($)',
                            'performance_score': 'Performance Score',
                            'quadrant': 'Category'
                        },
                        color_discrete_map={
                            'Stars': '#FFD700',
                            'Bargains': '#00FF00', 
                            'Overpaid': '#FF4444',
                            'Developing': '#888888'
                        }
                    )
                    
                    # Add median lines
                    fig_quadrant.add_hline(
                        y=median_performance, 
                        line_dash="dash", 
                        line_color="white",
                        annotation_text=f"Median Performance: {median_performance:.1f}",
                        annotation_position="right"
                    )
                    fig_quadrant.add_vline(
                        x=median_salary,
                        line_dash="dash",
                        line_color="white", 
                        annotation_text=f"Median Salary: ${median_salary:,.0f}",
                        annotation_position="top"
                    )
                    
                    fig_quadrant.update_layout(height=500)
                    st.plotly_chart(fig_quadrant, use_container_width=True)
                    
                    # SALARY BANDS TABLE
                    st.markdown("---")
                    st.markdown("### Salary Band Analysis")
                    
                    # Define salary bands
                    bands = [
                        (0, 200000, '<$200k'),
                        (200000, 400000, '$200k-$400k'),
                        (400000, 700000, '$400k-$700k'),
                        (700000, 1200000, '$700k-$1.2M'),
                        (1200000, 2000000, '$1.2M-$2M'),
                        (2000000, 3000000, '$2M-$3M'),
                        (3000000, float('inf'), '$3M+')
                    ]
                    
                    band_data = []
                    for min_sal, max_sal, label in bands:
                        band_players = position_data[
                            (position_data['guaranteed_compensation'] >= min_sal) & 
                            (position_data['guaranteed_compensation'] < max_sal)
                        ]
                        
                        if len(band_players) > 0:
                            avg_perf = band_players['performance_score'].mean()
                            top_3 = band_players.nlargest(3, 'performance_score')['display_name'].tolist()
                            top_3_str = ', '.join(top_3)
                            
                            band_data.append({
                                'Salary Range': label,
                                'Players': len(band_players),
                                'Avg Performance': f"{avg_perf:.1f}",
                                'Top 3 Players': top_3_str
                            })
                    
                    if band_data:
                        st.dataframe(pd.DataFrame(band_data), use_container_width=True, hide_index=True)
                    
                    # BENCHMARK TABLE
                    st.markdown("---")
                    st.markdown("### Performance Benchmarks")
                    
                    # Calculate percentiles
                    percentiles = [90, 75, 50, 25, 10]
                    benchmark_data = []
                    
                    for p in percentiles:
                        perf_threshold = position_data['performance_score'].quantile(p/100)
                        
                        # Find players at or above this performance level
                        qualified = position_data[position_data['performance_score'] >= perf_threshold]
                        
                        if len(qualified) > 0:
                            min_salary = qualified['guaranteed_compensation'].min()
                            avg_salary = qualified['guaranteed_compensation'].mean()
                            median_salary_q = qualified['guaranteed_compensation'].median()
                            
                            benchmark_data.append({
                                'Performance Level': f'Top {100-p}%',
                                'Min Performance Score': f"{perf_threshold:.1f}",
                                'Minimum Salary': f"${min_salary:,.0f}",
                                'Average Salary': f"${avg_salary:,.0f}",
                                'Median Salary': f"${median_salary_q:,.0f}"
                            })
                    
                    st.table(pd.DataFrame(benchmark_data))
                    
                    # Key insights
                    st.markdown("---")
                    st.markdown("### Key Insights")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Best value players
                        best_value = position_data.nlargest(5, 'value_ratio')[['display_name', 'guaranteed_compensation', 'performance_score', 'value_ratio']].copy()
                        best_value['guaranteed_compensation'] = best_value['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
                        best_value['performance_score'] = best_value['performance_score'].round(1)
                        best_value['value_ratio'] = best_value['value_ratio'].round(2)
                        
                        st.markdown("**Best Value Players:**")
                        st.dataframe(best_value, use_container_width=True, hide_index=True)
                    
                    with col2:
                        # Worst value players (overpaid)
                        worst_value = position_data[position_data['performance_score'] > 10].nsmallest(5, 'value_ratio')[['display_name', 'guaranteed_compensation', 'performance_score', 'value_ratio']].copy()
                        worst_value['guaranteed_compensation'] = worst_value['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
                        worst_value['performance_score'] = worst_value['performance_score'].round(1)
                        worst_value['value_ratio'] = worst_value['value_ratio'].round(2)
                        
                        st.markdown("**Lowest Value Players:**")
                        st.dataframe(worst_value, use_container_width=True, hide_index=True)
                
                else:
                    st.warning(f"No data available for {selected_position} position")

            with tab2:
    # Cross-Position Comparison
                st.subheader("⚖️ Cross-Position Salary Comparison")
                
                # Salary slider
                budget = st.slider(
                    "Budget Available:",
                    min_value=100000,
                    max_value=5000000,
                    value=500000,
                    step=50000,
                    format="$%d"
                )
                
                st.markdown(f"### What does ${budget:,} get you in each position?")
                
                # Cards for each position
                col1, col2, col3, col4 = st.columns(4)
                
                positions = ['FW', 'MF', 'DF', 'GK']
                position_names = {'FW': 'Forwards', 'MF': 'Midfielders', 'DF': 'Defenders', 'GK': 'Goalkeepers'}
                cols = [col1, col2, col3, col4]
                
                for pos, col in zip(positions, cols):
                    pos_data = analysis_df[analysis_df['general_position'] == pos]
                    
                    if len(pos_data) > 0:
                        # Find players near this salary (within 20%)
                        tolerance = budget * 0.2
                        similar_salary = pos_data[
                            (pos_data['guaranteed_compensation'] >= budget - tolerance) &
                            (pos_data['guaranteed_compensation'] <= budget + tolerance)
                        ]
                        
                        with col:
                            st.markdown(f"**{position_names[pos]}**")
                            
                            if len(similar_salary) > 0:
                                avg_performance = similar_salary['performance_score'].mean()
                                min_performance = similar_salary['performance_score'].min()
                                max_performance = similar_salary['performance_score'].max()
                                
                                st.metric("Avg Performance", f"{avg_performance:.1f}")
                                st.caption(f"Range: {min_performance:.1f} - {max_performance:.1f}")
                                st.caption(f"{len(similar_salary)} players")
                                
                                # Show top player at this price
                                top_player = similar_salary.nlargest(1, 'performance_score').iloc[0]
                                st.info(f"Best: {top_player['display_name']}\n({top_player['performance_score']:.1f})")
                            else:
                                st.caption("No players at this salary")
                
                # HEATMAP
                st.markdown("---")
                st.markdown("### Player Density Heatmap")
                st.caption("Shows where players cluster across salary and performance ranges")
                
                # Create heatmap data for each position
                heatmap_data_list = []
                
                for pos in positions:
                    pos_data = analysis_df[analysis_df['general_position'] == pos].copy()
                    
                    if len(pos_data) > 0:
                        # Bin salaries - 8 edges, 7 labels
                        pos_data['salary_bin'] = pd.cut(
                            pos_data['guaranteed_compensation'],
                            bins=[0, 200000, 400000, 700000, 1200000, 2000000, 3000000, 10000000],
                            labels=['<$200k', '$200k-$400k', '$400k-$700k', '$700k-$1.2M', '$1.2M-$2M', '$2M-$3M', '$3M+'],
                            include_lowest=True
                        )
                        
                        # Bin performance
                        try:
                            pos_data['perf_bin'] = pd.qcut(
                                pos_data['performance_score'],
                                q=5,
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                                duplicates='drop'
                            )
                        except ValueError:
                            # If qcut fails due to duplicate edges, use cut instead
                            perf_min = pos_data['performance_score'].min()
                            perf_max = pos_data['performance_score'].max()
                            perf_range = perf_max - perf_min
                            pos_data['perf_bin'] = pd.cut(
                                pos_data['performance_score'],
                                bins=5,
                                labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
                            )
                        
                        # Count players in each bin
                        grouped = pos_data.groupby(['salary_bin', 'perf_bin'], observed=True).size().reset_index(name='count')
                        grouped['position'] = position_names[pos]
                        
                        heatmap_data_list.append(grouped)
                
                if heatmap_data_list:
                    all_heatmap = pd.concat(heatmap_data_list, ignore_index=True)
                    
                    # Create subplot for each position
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=list(position_names.values()),
                        vertical_spacing=0.15,
                        horizontal_spacing=0.1
                    )
                    
                    positions_grid = [('FW', 1, 1), ('MF', 1, 2), ('DF', 2, 1), ('GK', 2, 2)]
                    
                    for pos, row, col in positions_grid:
                        pos_heatmap = all_heatmap[all_heatmap['position'] == position_names[pos]]
                        
                        if len(pos_heatmap) > 0:
                            # Pivot for heatmap
                            pivot = pos_heatmap.pivot(index='perf_bin', columns='salary_bin', values='count').fillna(0)
                            
                            # Reorder to have performance high to low
                            perf_order = ['Very High', 'High', 'Medium', 'Low', 'Very Low']
                            pivot = pivot.reindex([p for p in perf_order if p in pivot.index])
                            
                            heatmap = go.Heatmap(
                                z=pivot.values,
                                x=pivot.columns,
                                y=pivot.index,
                                colorscale='Viridis',
                                showscale=(row == 1 and col == 2)
                            )
                            
                            fig.add_trace(heatmap, row=row, col=col)
                    
                    fig.update_layout(height=700, title_text="Player Distribution: Salary vs Performance")
                    st.plotly_chart(fig, use_container_width=True)
                        
            with tab3:
    # Top players analysis
                        st.subheader("🎯 Agent Contract Negotiation Tool")
                        
                        # Split into two main sections
                        agent_tab1, agent_tab2 = st.tabs(["📊 Player Performance Analysis", "💼 Contract Justification"])
                        
                        with agent_tab1:
                            # Player Selection and Comparison
                            st.markdown("### Select Your Player")
                            
                            col1, col2 = st.columns([2, 1])
                            with col1:
                                selected_agent_player = st.selectbox(
                                    "Choose player to analyze:",
                                    options=sorted(analysis_df['display_name'].unique()),
                                    key="agent_player"
                                )
                            
                            with col2:
                                analysis_year_filter = st.selectbox(
                                    "Analysis Period:",
                                    options=analysis_df['analysis_year'].unique(),
                                    key="agent_year"
                                )
                            
                            if selected_agent_player:
                                # Get player data
                                player_current = analysis_df[
                                    (analysis_df['display_name'] == selected_agent_player) & 
                                    (analysis_df['analysis_year'] == analysis_year_filter)
                                ]
                                
                                if len(player_current) > 0:
                                    player_row = player_current.iloc[0]
                                    player_pos = player_row['general_position']
                                    player_score = player_row['performance_score']
                                    player_salary = player_row['guaranteed_compensation']
                                    player_minutes = player_row['Playing Time Min']
                                    
                                    # Get all players in same position for comparison
                                    position_peers = analysis_df[
                                        (analysis_df['general_position'] == player_pos) &
                                        (analysis_df['analysis_year'] == analysis_year_filter)
                                    ]
                                    
                                    # Calculate percentiles
                                    perf_percentile = (position_peers['performance_score'] < player_score).sum() / len(position_peers) * 100
                                    salary_percentile = (position_peers['guaranteed_compensation'] < player_salary).sum() / len(position_peers) * 100
                                    minutes_percentile = (position_peers['Playing Time Min'] < player_minutes).sum() / len(position_peers) * 100
                                    
                                    # KEY METRICS DISPLAY
                                    st.markdown(f"### {selected_agent_player} - {player_pos}")
                                    
                                    col1, col2, col3, col4, col5 = st.columns(5)
                                    
                                    with col1:
                                        st.metric("Performance Score", f"{player_score:.1f}")
                                        st.caption(f"{perf_percentile:.0f}th percentile")
                                    
                                    with col2:
                                        st.metric("Current Salary", f"${player_salary:,.0f}")
                                        st.caption(f"{salary_percentile:.0f}th percentile")
                                    
                                    with col3:
                                        st.metric("Value Ratio", f"{player_row['value_ratio']:.2f}")
                                        avg_value = position_peers['value_ratio'].mean()
                                        delta = player_row['value_ratio'] - avg_value
                                        st.caption(f"{delta:+.2f} vs avg")
                                    
                                    with col4:
                                        st.metric("Minutes Played", f"{player_minutes:.0f}")
                                        st.caption(f"{minutes_percentile:.0f}th percentile")
                                    
                                    with col5:
                                        # Calculate gap
                                        gap = perf_percentile - salary_percentile
                                        st.metric("Perf-Salary Gap", f"{gap:+.0f}%")
                                        if gap > 20:
                                            st.caption("🟢 UNDERPAID")
                                        elif gap < -20:
                                            st.caption("🔴 OVERPAID")
                                        else:
                                            st.caption("🟡 FAIR")
                                    
                                    # VISUAL COMPARISON
                                    st.markdown("---")
                                    st.markdown("### Position Comparison")
                                    
                                    # Create comparison scatter plot
                                    fig = px.scatter(
                                        position_peers,
                                        x='guaranteed_compensation',
                                        y='performance_score',
                                        size='Playing Time Min',
                                        hover_data=['display_name', 'club_standard', 'value_ratio'],
                                        title=f'{player_pos} Position: Performance vs Salary',
                                        labels={
                                            'guaranteed_compensation': 'Salary ($)',
                                            'performance_score': 'Performance Score'
                                        },
                                        opacity=0.6
                                    )
                                    
                                    # Highlight selected player
                                    player_point = position_peers[position_peers['display_name'] == selected_agent_player]
                                    fig.add_scatter(
                                        x=player_point['guaranteed_compensation'],
                                        y=player_point['performance_score'],
                                        mode='markers',
                                        marker=dict(size=20, color='red', line=dict(width=2, color='white')),
                                        name='Your Player',
                                        showlegend=True
                                    )
                                    
                                    # Add benchmark lines
                                    median_salary = position_peers['guaranteed_compensation'].median()
                                    median_perf = position_peers['performance_score'].median()
                                    
                                    fig.add_hline(y=median_perf, line_dash="dash", line_color="gray", 
                                                annotation_text="Median Performance")
                                    fig.add_vline(x=median_salary, line_dash="dash", line_color="gray",
                                                annotation_text="Median Salary")
                                    
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # BENCHMARK TABLE
                                    st.markdown("### Position Salary Benchmarks")
                                    
                                    # Calculate percentile salaries
                                    percentiles = [10, 25, 50, 75, 90]
                                    benchmark_data = []
                                    
                                    for p in percentiles:
                                        salary_at_p = position_peers['guaranteed_compensation'].quantile(p/100)
                                        perf_at_p = position_peers['performance_score'].quantile(p/100)
                                        benchmark_data.append({
                                            'Percentile': f'{p}th',
                                            'Salary': f'${salary_at_p:,.0f}',
                                            'Avg Performance': f'{perf_at_p:.1f}',
                                            'Your Player': '✓' if player_score >= perf_at_p else ''
                                        })
                                    
                                    st.table(pd.DataFrame(benchmark_data))
                                    
                                    # UNDERPERFORMERS vs YOUR PLAYER
                                    st.markdown("---")
                                    st.markdown("### Players Earning More, Performing Less")
                                    st.caption("Use this to show why your player deserves a raise")
                                    
                                    # Find players earning more but performing worse
                                    overpaid_comparison = position_peers[
                                        (position_peers['guaranteed_compensation'] > player_salary) &
                                        (position_peers['performance_score'] < player_score)
                                    ].sort_values('guaranteed_compensation', ascending=False)
                                    
                                    if len(overpaid_comparison) > 0:
                                        comparison_display = overpaid_comparison[[
                                            'display_name', 'club_standard', 'guaranteed_compensation', 
                                            'performance_score', 'Playing Time Min'
                                        ]].head(10).copy()
                                        
                                        comparison_display['Salary Diff'] = comparison_display['guaranteed_compensation'] - player_salary
                                        comparison_display['Perf Diff'] = player_score - comparison_display['performance_score']
                                        
                                        comparison_display['guaranteed_compensation'] = comparison_display['guaranteed_compensation'].apply(lambda x: f'${x:,.0f}')
                                        comparison_display['Salary Diff'] = comparison_display['Salary Diff'].apply(lambda x: f'+${x:,.0f}')
                                        comparison_display['performance_score'] = comparison_display['performance_score'].round(1)
                                        comparison_display['Perf Diff'] = comparison_display['Perf Diff'].apply(lambda x: f'+{x:.1f}')
                                        comparison_display['Playing Time Min'] = comparison_display['Playing Time Min'].round(0)
                                        
                                        st.dataframe(comparison_display, use_container_width=True, hide_index=True)
                                        
                                        st.success(f"✅ Found {len(overpaid_comparison)} players earning more but performing worse than {selected_agent_player}")
                                    else:
                                        st.info("Your player is already among the highest paid for their performance level")
                        
                        with agent_tab2:
                            # Contract Justification Report
                            st.markdown("### 💼 Contract Justification Report")
                            st.caption("Copy this data for contract negotiations")
                            
                            if selected_agent_player and len(player_current) > 0:
                                
                                # Calculate suggested salary based on performance percentile
                                target_salary_percentile = min(perf_percentile, 95)  # Cap at 95th
                                suggested_salary = position_peers['guaranteed_compensation'].quantile(target_salary_percentile/100)
                                salary_increase = suggested_salary - player_salary
                                increase_pct = (salary_increase / player_salary * 100) if player_salary > 0 else 0
                                
                                # Header metrics
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Current Salary", f"${player_salary:,.0f}")
                                with col2:
                                    st.metric("Justified Salary", f"${suggested_salary:,.0f}")
                                with col3:
                                    st.metric("Suggested Raise", f"${salary_increase:,.0f}", f"{increase_pct:.1f}%")
                                
                                st.markdown("---")
                                
                                # Generate negotiation report
                                negotiation_report = f"""
                    CONTRACT NEGOTIATION BRIEF
                    Player: {selected_agent_player}
                    Position: {player_pos}
                    Analysis Period: {analysis_year_filter}
                    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d')}

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    EXECUTIVE SUMMARY
                    {selected_agent_player} is currently UNDERPAID by ${salary_increase:,.0f} ({increase_pct:.1f}%)
                    Performance Rank: {perf_percentile:.0f}th percentile among {player_pos}s
                    Salary Rank: {salary_percentile:.0f}th percentile among {player_pos}s
                    Gap: {perf_percentile - salary_percentile:.0f} percentile points

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    PERFORMANCE METRICS
                    Current Performance Score: {player_score:.1f}
                    Position Average: {position_peers['performance_score'].mean():.1f}
                    Outperformance: +{player_score - position_peers['performance_score'].mean():.1f} points

                    Minutes Played: {player_minutes:.0f} (Top {100-minutes_percentile:.0f}% in position)
                    Value Ratio: {player_row['value_ratio']:.2f} (performance per $100k)

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    COMPARABLE PLAYER ANALYSIS
                    Players performing WORSE but earning MORE:

                    """
                                
                                if len(overpaid_comparison) > 0:
                                    for idx, comp in overpaid_comparison.head(5).iterrows():
                                        negotiation_report += f"""
                    - {comp['display_name']} ({comp['club_standard']})
                    Salary: ${comp['guaranteed_compensation']:,.0f} (+${comp['guaranteed_compensation'] - player_salary:,.0f})
                    Performance: {comp['performance_score']:.1f} (-{player_score - comp['performance_score']:.1f} vs {selected_agent_player})
                    """
                                
                                negotiation_report += f"""

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

                    SALARY RECOMMENDATION
                    Based on {perf_percentile:.0f}th percentile performance among {player_pos} position:

                    Current Salary:    ${player_salary:,.0f}
                    Market Value:      ${suggested_salary:,.0f}
                    Recommended Raise: ${salary_increase:,.0f} ({increase_pct:.1f}%)

                    JUSTIFICATION:
                    - Performing at {perf_percentile:.0f}th percentile but paid at {salary_percentile:.0f}th percentile
                    - {len(overpaid_comparison)} comparable players earn more despite lower performance
                    - Consistent playing time ({player_minutes:.0f} minutes)
                    - Strong value delivery ({player_row['value_ratio']:.2f} performance per $100k)

                    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    """
                                
                                # Display and download
                                st.text_area("Negotiation Brief (Copy/Paste)", negotiation_report, height=400)
                                
                                st.download_button(
                                    label="📥 Download Negotiation Brief",
                                    data=negotiation_report,
                                    file_name=f'{selected_agent_player.replace(" ", "_")}_contract_justification.txt',
                                    mime='text/plain',
                                )
                                
                            else:
                                st.info("Select a player in the Performance Analysis tab to generate contract justification")
                                
            with tab4:
                # Team analysis with new features
                st.subheader("🏟️ Team Analysis Dashboard")
                
                # Team selector for detailed analysis
                selected_team = st.selectbox(
                    "Select a team for detailed analysis:",
                    options=['All Teams'] + sorted(analysis_df['club_standard'].unique())
                )
                
                if selected_team == 'All Teams':
                    # Overall team efficiency analysis
                    st.subheader("Team Efficiency Overview")
                    
                    team_analysis = analysis_df.groupby('club_standard').agg({
                        'guaranteed_compensation': ['count', 'sum', 'mean'],
                        'performance_score': 'mean',
                        'value_ratio': 'mean'
                    }).round(2)
                    
                    team_analysis.columns = ['Player Count', 'Total Payroll', 'Avg Salary', 
                                            'Avg Performance', 'Avg Value Ratio']
                    team_analysis = team_analysis.sort_values('Avg Value Ratio', ascending=False)
                    
                    # Format currency columns
                    team_analysis['Total Payroll'] = team_analysis['Total Payroll'].apply(lambda x: f"${x:,.0f}")
                    team_analysis['Avg Salary'] = team_analysis['Avg Salary'].apply(lambda x: f"${x:,.0f}")
                    
                    st.dataframe(team_analysis, use_container_width=True)
                    
                    # Team scatter plot
                    team_scatter_data = analysis_df.groupby('club_standard').agg({
                        'guaranteed_compensation': 'sum',
                        'performance_score': 'mean'
                    }).reset_index()
                    
                    fig_team = px.scatter(
                        team_scatter_data,
                        x='guaranteed_compensation',
                        y='performance_score',
                        text='club_standard',
                        title='Team Total Payroll vs Average Performance',
                        labels={
                            'guaranteed_compensation': 'Total Team Payroll ($)',
                            'performance_score': 'Average Performance Score'
                        }
                    )
                    
                    fig_team.update_traces(textposition='top center')
                    fig_team.update_layout(height=500)
                    st.plotly_chart(fig_team, use_container_width=True)
                    
                else:
                    # Detailed team analysis
                    team_data = analysis_df[analysis_df['club_standard'] == selected_team].copy()
                    
                    # Create three columns for the new features
                    st.subheader(f"📊 {selected_team} - Detailed Analysis")
                    
                    # Feature 1: Player Movement Analysis
                    st.markdown("### 🔄 Player Movement (2024 vs 2025)")
                    
                    # Get unique players for each year from salary data
                    salary_2024_team = salary_df[(salary_df['club_standard'] == selected_team) & 
                                                 (salary_df['salary_year'] == 2024)]
                    salary_2025_team = salary_df[(salary_df['club_standard'] == selected_team) & 
                                                 (salary_df['salary_year'] == 2025)]
                    
                    players_2024 = set(salary_2024_team['full_name'].dropna())
                    players_2025 = set(salary_2025_team['full_name'].dropna())
                    
                    # Calculate movements
                    stayed = players_2024.intersection(players_2025)
                    left = players_2024 - players_2025
                    joined = players_2025 - players_2024
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("✅ Stayed", len(stayed))
                        with st.expander("View Players"):
                            for player in sorted(stayed)[:10]:
                                st.write(f"• {player}")
                            if len(stayed) > 10:
                                st.write(f"... and {len(stayed)-10} more")
                    
                    with col2:
                        st.metric("➡️ Left Team", len(left))
                        with st.expander("View Players"):
                            for player in sorted(left)[:10]:
                                st.write(f"• {player}")
                            if len(left) > 10:
                                st.write(f"... and {len(left)-10} more")
                    
                    with col3:
                        st.metric("⬅️ Joined Team", len(joined))
                        with st.expander("View Players"):
                            for player in sorted(joined)[:10]:
                                st.write(f"• {player}")
                            if len(joined) > 10:
                                st.write(f"... and {len(joined)-10} more")
                    
                    # Feature 2: Salary Changes for Returning Players
                    st.markdown("### 💰 Salary Changes for Returning Players")
                    
                    if len(stayed) > 0:
                        # Get salary data for players who stayed
                        salary_changes = []
                        for player in stayed:
                            sal_2024 = salary_2024_team[salary_2024_team['full_name'] == player]['guaranteed_compensation'].values
                            sal_2025 = salary_2025_team[salary_2025_team['full_name'] == player]['guaranteed_compensation'].values
                            
                            if len(sal_2024) > 0 and len(sal_2025) > 0:
                                change = sal_2025[0] - sal_2024[0]
                                pct_change = (change / sal_2024[0]) * 100 if sal_2024[0] > 0 else 0
                                salary_changes.append({
                                    'Player': player,
                                    '2024 Salary': sal_2024[0],
                                    '2025 Salary': sal_2025[0],
                                    'Change ($)': change,
                                    'Change (%)': pct_change
                                })
                        
                        if salary_changes:
                            salary_changes_df = pd.DataFrame(salary_changes)
                            salary_changes_df = salary_changes_df.sort_values('Change ($)', ascending=False)
                            
                            # Summary metrics
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                avg_change = salary_changes_df['Change ($)'].mean()
                                st.metric("Avg Salary Change", f"${avg_change:,.0f}")
                            with col2:
                                got_raise = len(salary_changes_df[salary_changes_df['Change ($)'] > 0])
                                st.metric("Got Raises", f"{got_raise}/{len(salary_changes_df)}")
                            with col3:
                                biggest_raise = salary_changes_df['Change ($)'].max()
                                st.metric("Biggest Raise", f"${biggest_raise:,.0f}")
                            with col4:
                                biggest_cut = salary_changes_df['Change ($)'].min()
                                st.metric("Biggest Cut", f"${biggest_cut:,.0f}")
                            
                            # Display top movers
                            st.write("**Top Salary Changes:**")
                            
                            # Format for display
                            display_changes = salary_changes_df.copy()
                            display_changes['2024 Salary'] = display_changes['2024 Salary'].apply(lambda x: f"${x:,.0f}")
                            display_changes['2025 Salary'] = display_changes['2025 Salary'].apply(lambda x: f"${x:,.0f}")
                            display_changes['Change ($)'] = display_changes['Change ($)'].apply(lambda x: f"${x:+,.0f}")
                            display_changes['Change (%)'] = display_changes['Change (%)'].apply(lambda x: f"{x:+.1f}%")
                            
                            st.dataframe(display_changes, use_container_width=True, hide_index=True)
                    else:
                        st.info("No players stayed with the team across both years")
                    
                    # Feature 3: Team Performance vs Salary Scatter Plot
                    st.markdown("### 📈 Team Performance vs Salary Analysis")
                    
                    if len(team_data) > 0:
                        # Create scatter plot for team
                        fig_team_scatter = px.scatter(
                            team_data,
                            x='guaranteed_compensation',
                            y='performance_score',
                            color='general_position',
                            size='Playing Time Min',
                            hover_data={
                                'display_name': True,
                                'analysis_year': True,
                                'Per 90 Minutes Gls': ':.2f',
                                'Per 90 Minutes Ast': ':.2f',
                                'value_ratio': ':.2f',
                                'guaranteed_compensation': ':$,.0f'
                            },
                            title=f'{selected_team} - Player Performance vs Salary',
                            labels={
                                'guaranteed_compensation': 'Guaranteed Compensation ($)',
                                'performance_score': 'Performance Score',
                                'general_position': 'Position'
                            },
                            color_discrete_map={'FW': '#FF6B6B', 'MF': '#4ECDC4', 'DF': '#45B7D1', 'GK': '#96CEB4'}
                        )
                        
                        # Add average lines
                        fig_team_scatter.add_hline(
                            y=team_data['performance_score'].mean(),
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Team Avg Performance: {team_data['performance_score'].mean():.1f}"
                        )
                        
                        fig_team_scatter.add_vline(
                            x=team_data['guaranteed_compensation'].mean(),
                            line_dash="dash",
                            line_color="gray",
                            annotation_text=f"Team Avg Salary: ${team_data['guaranteed_compensation'].mean():,.0f}"
                        )
                        
                        fig_team_scatter.update_layout(height=600)
                        st.plotly_chart(fig_team_scatter, use_container_width=True)
                        
                        # Team summary stats
                        st.markdown("**Team Statistics:**")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Payroll", f"${team_data['guaranteed_compensation'].sum():,.0f}")
                            st.metric("Player Count", len(team_data))
                        
                        with col2:
                            st.metric("Avg Performance Score", f"{team_data['performance_score'].mean():.1f}")
                            st.metric("Avg Value Ratio", f"{team_data['value_ratio'].mean():.2f}")
                        
                        with col3:
                            best_player = team_data.nlargest(1, 'performance_score')['display_name'].values[0] if len(team_data) > 0 else "N/A"
                            best_value = team_data.nlargest(1, 'value_ratio')['display_name'].values[0] if len(team_data) > 0 else "N/A"
                            st.metric("Top Performer", best_player)
                            st.metric("Best Value", best_value)
                    else:
                        st.warning(f"No performance data available for {selected_team}")
            
            with tab5:
                # Performance Breakdown Analysis
                st.subheader("🔍 Detailed Performance Metrics Analysis")
                
                # Select player for detailed breakdown
                selected_player = st.selectbox(
                    "Select a player for detailed metrics:",
                    options=sorted(analysis_df['display_name'].unique())
                )
                
                if selected_player:
                    player_data = analysis_df[analysis_df['display_name'] == selected_player].iloc[0]
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.markdown("### Player Info")
                        st.write(f"**Position:** {player_data['general_position']}")
                        st.write(f"**Team:** {player_data['club_standard']}")
                        st.write(f"**Salary:** ${player_data['guaranteed_compensation']:,.0f}")
                        st.write(f"**Performance Score:** {player_data['performance_score']:.1f}")
                        st.write(f"**Value Ratio:** {player_data['value_ratio']:.2f}")
                        st.write(f"**Minutes Played:** {player_data['Playing Time Min']:.0f}")
                    
                    with col2:
                        # Create radar chart based on position
                        pos = player_data['general_position']
                        
                        if pos == 'FW':
                            metrics = {
                                'Goals/90': player_data.get('Per 90 Minutes Gls', 0),
                                'Assists/90': player_data.get('Per 90 Minutes Ast', 0),
                                'xG/90': player_data.get('Per 90 Minutes xG', 0),
                                'Shots': player_data.get('Standard Sh', 0) / 10,
                                'Dribbles': player_data.get('Take-Ons Succ', 0) / 5,
                                'Key Passes': player_data.get('KP', 0) / 5
                            }
                        elif pos == 'MF':
                            metrics = {
                                'Assists/90': player_data.get('Per 90 Minutes Ast', 0),
                                'Key Passes': player_data.get('KP', 0) / 5,
                                'Pass %': player_data.get('Total Cmp%', 0) / 20,
                                'Prog Passes': player_data.get('PrgP', 0) / 10,
                                'Tackles': player_data.get('Tackles Tkl', 0) / 5,
                                'Interceptions': player_data.get('Performance Int', 0) / 5
                            }
                        elif pos == 'DF':
                            metrics = {
                                'Tackles': player_data.get('Tackles Tkl', 0) / 5,
                                'Interceptions': player_data.get('Performance Int', 0) / 5,
                                'Blocks': player_data.get('Blocks Blocks', 0) / 3,
                                'Clearances': player_data.get('Clr', 0) / 10,
                                'Aerial Won': player_data.get('Aerial Duels Won', 0) / 5,
                                'Pass %': player_data.get('Total Cmp%', 0) / 20
                            }
                        else:  # GK
                            metrics = {
                                'Games': player_data.get('Playing Time MP', 0) / 5,
                                'Minutes': player_data.get('Playing Time Min', 0) / 500,
                                'Team +/-': (player_data.get('Team Success +/-', 0) + 50) / 20,
                                'Pass %': player_data.get('Total Cmp%', 0) / 20,
                                'Goals Against': max(0, 100 - player_data.get('Team Success onGA', 0)) / 20
                            }
                        
                        # Create radar chart
                        fig_radar = go.Figure(data=go.Scatterpolar(
                            r=list(metrics.values()),
                            theta=list(metrics.keys()),
                            fill='toself',
                            name=selected_player
                        ))
                        
                        fig_radar.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 5]
                                )),
                            showlegend=False,
                            title=f"{selected_player} - Performance Metrics"
                        )
                        
                        st.plotly_chart(fig_radar, use_container_width=True)

    except Exception as e:
            st.error(f"❌ Error processing data: {str(e)}")
            st.write("Debug information:")
            st.code(str(e))
                
                # Clear cache on error
            if st.button("🔄 Clear Cache and Retry"):
                    st.session_state.processed_data = None
                    st.session_state.salary_df = None
                    st.session_state.performance_df = None
                    st.cache_data.clear()
                    st.rerun()

else:
    st.info("👆 Please upload both salary and performance CSV files to begin the analysis")
