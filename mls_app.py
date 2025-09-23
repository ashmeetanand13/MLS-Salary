import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import unicodedata

# Page configuration
st.set_page_config(
    page_title="MLS Salary vs Performance Analysis", 
    page_icon="⚽", 
    layout="wide"
)

# Title and description
st.title("⚽ MLS Salary vs Performance Analysis")
st.markdown("Analyze whether MLS player salaries are justified by their on-field performance")

# Sidebar for file uploads
st.sidebar.header("📁 Upload Data Files")
salary_file = st.sidebar.file_uploader("Upload Salary Data (CSV)", type=['csv'], key="salary")
performance_file = st.sidebar.file_uploader("Upload Performance Data (CSV)", type=['csv'], key="performance")

def normalize_name(name):
    """Enhanced normalize player names for matching - handles Excel encoding issues"""
    if pd.isna(name) or name == '':
        return ''
    
    # Convert to string and lowercase
    name = str(name).lower().strip()
    
    # Handle common Excel encoding issues
    name = name.replace('\u00a0', ' ')  # Non-breaking space
    name = name.replace('\u2013', '-')  # En dash
    name = name.replace('\u2014', '-')  # Em dash
    name = name.replace('\u2019', "'")  # Right single quote
    name = name.replace('\u201c', '"')  # Left double quote
    name = name.replace('\u201d', '"')  # Right double quote
    
    # Remove all diacritical marks more aggressively
    import unicodedata
    name = unicodedata.normalize('NFD', name)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    
    # Handle specific character mappings that Excel might mess up
    char_mappings = {
        'á': 'a', 'à': 'a', 'ä': 'a', 'â': 'a', 'ã': 'a', 'å': 'a',
        'é': 'e', 'è': 'e', 'ë': 'e', 'ê': 'e',
        'í': 'i', 'ì': 'i', 'ï': 'i', 'î': 'i',
        'ó': 'o', 'ò': 'o', 'ö': 'o', 'ô': 'o', 'õ': 'o', 'ø': 'o',
        'ú': 'u', 'ù': 'u', 'ü': 'u', 'û': 'u',
        'ñ': 'n', 'ç': 'c', 'ß': 'ss',
        'æ': 'ae', 'œ': 'oe'
    }
    
    for old_char, new_char in char_mappings.items():
        name = name.replace(old_char, new_char)
    
    # Remove special characters except spaces and hyphens
    import re
    name = re.sub(r'[^a-z\s\-]', '', name)
    
    # Clean up multiple spaces
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def normalize_team_name(team):
    """Normalize team names for consistent matching"""
    if pd.isna(team) or team == '':
        return ''
    
    # Convert to lowercase and remove accents
    team = str(team).lower()
    team = unicodedata.normalize('NFD', team)
    team = re.sub(r'[\u0300-\u036f]', '', team)
    
    # Common team name standardizations
    team_mappings = {
        'atlanta utd': 'atlanta united',
        'atlanta united fc': 'atlanta united',
        'charlotte fc': 'charlotte',
        'inter miami cf': 'inter miami',
        'new york city fc': 'new york city',
        'new york red bulls': 'ny red bulls',
        'new england revolution': 'new england',
        'san jose earthquakes': 'san jose',
        'seattle sounders fc': 'seattle sounders',
        'sporting kansas city': 'sporting kc',
        'portland timbers': 'portland',
        'lafc': 'los angeles fc',
        'la galaxy': 'los angeles galaxy',
        'minnesota united fc': 'minnesota united',
        'real salt lake': 'real salt lake',
        'colorado rapids': 'colorado',
        'fc cincinnati': 'cincinnati',
        'columbus crew': 'columbus',
        'nashville sc': 'nashville',
        'orlando city sc': 'orlando city',
        'toronto fc': 'toronto',
        'cf montreal': 'montreal',
        'vancouver whitecaps fc': 'vancouver',
        'chicago fire fc': 'chicago fire',
        'dc united': 'dc united',
        'philadelphia union': 'philadelphia',
        'fc dallas': 'dallas',
        'houston dynamo fc': 'houston dynamo',
        'austin fc': 'austin'
    }
    
    # Remove common suffixes
    team = re.sub(r'\b(fc|cf|sc|united fc|city fc)\b', '', team).strip()
    
    # Apply mappings
    if team in team_mappings:
        team = team_mappings[team]
    
    # Clean up extra spaces
    team = re.sub(r'\s+', ' ', team).strip()
    
    return team

def clean_year_data(df, year_column):
    """Clean and validate year data"""
    if year_column not in df.columns:
        return df
    
    # Convert to string first to handle various formats
    df[year_column] = df[year_column].astype(str)
    
    # Remove commas and other formatting
    df[year_column] = df[year_column].str.replace(',', '').str.replace('.0', '')
    
    # Convert to integer, handling errors
    try:
        df[year_column] = pd.to_numeric(df[year_column], errors='coerce').astype('Int64')
    except:
        # If conversion fails, try to extract 4-digit years
        df[year_column] = df[year_column].str.extract(r'(\d{4})').astype('Int64')
    
    # Filter for reasonable years (2020-2030)
    valid_years = (df[year_column] >= 2020) & (df[year_column] <= 2030)
    df = df[valid_years].copy()
    
    return df

def calculate_performance_score(row):
    """Calculate position-specific performance score"""
    pos = row.get('general_position')
    
    # Helper function to safely get numeric values
    def safe_get(key, default=0):
        val = row.get(key, default)
        return float(val) if pd.notna(val) else default
    
    if pos == 'FW':
        # Forwards: prioritize goals and assists
        score = (safe_get('Per 90 Minutes Gls') * 4 + 
                safe_get('Per 90 Minutes Ast') * 3 +
                safe_get('Per 90 Minutes xG') * 2 +
                safe_get('Team Success +/-') / 10)
    elif pos == 'MF':
        # Midfielders: balanced scoring
        score = (safe_get('Per 90 Minutes Gls') * 2.5 + 
                safe_get('Per 90 Minutes Ast') * 3 +
                safe_get('Per 90 Minutes xAG') * 2.5 +
                safe_get('Team Success +/-') / 8)
    elif pos == 'DF':
        # Defenders: team success and defensive actions
        score = (safe_get('Per 90 Minutes Gls') * 2 + 
                safe_get('Per 90 Minutes Ast') * 2.5 +
                safe_get('Team Success +/-') / 4 +
                safe_get('Performance Int') / 15 +
                safe_get('Tackles Tkl') / 12)
    elif pos == 'GK':
        # Goalkeepers: team success based
        score = safe_get('Team Success +/-') / 2 + 5  # Base score for GK
    else:
        score = 0
    
    return max(0, score)

def fuzzy_match_single_player(player_name, performance_df):
    """Fuzzy match a single player against performance data"""
    from difflib import SequenceMatcher
    
    best_match = None
    best_score = 0
    
    for _, row in performance_df.iterrows():
        score = SequenceMatcher(None, player_name, row['normalized_name']).ratio()
        if score > best_score and score > 0.8:  # 80% similarity threshold
            best_score = score
            best_match = row
    
    if best_match is not None:
        return pd.DataFrame([best_match])
    else:
        return pd.DataFrame()  # Empty DataFrame if no match found

def analyze_performance_salary_correlation(comprehensive_df, merged_df):
    """Analyze how performance correlates with NEXT year's salary changes"""
    team_analysis = {}
    
    # Use normalized team names for consistency
    teams = comprehensive_df['normalized_team'].unique()
    
    for team in teams:
        # Filter by normalized team name
        team_data = comprehensive_df[comprehensive_df['normalized_team'] == team]
        
        if len(team_data) == 0:
            continue
        
        # Extract salary data from merged_df for this team
        team_salary_data = merged_df[merged_df['normalized_team'] == team]
        
        # Separate salary data by year
        salary_2024 = team_salary_data[team_salary_data['salary_year'] == 2024]
        salary_2025 = team_salary_data[team_salary_data['salary_year'] == 2025]
        
        # Track roster changes between 2024 and 2025
        players_2024 = set(salary_2024['normalized_name'].unique())
        players_2025 = set(salary_2025['normalized_name'].unique())
        stayed_players = players_2024.intersection(players_2025)
        left_players = players_2024 - players_2025
        new_players = players_2025 - players_2024
        
        # Separate performance data by season
        perf_2023 = team_data[team_data['performance_season'] == 2023]
        perf_2024 = team_data[team_data['performance_season'] == 2024]
        
        # Find salary changes: 2024 → 2025 (for players who stayed)
        salary_changes_2025 = []
        
        for player_name in stayed_players:
            # Get salary data
            sal_2024_data = salary_2024[salary_2024['normalized_name'] == player_name]
            sal_2025_data = salary_2025[salary_2025['normalized_name'] == player_name]
            
            if len(sal_2024_data) > 0 and len(sal_2025_data) > 0:
                sal_24 = sal_2024_data.iloc[0]
                sal_25 = sal_2025_data.iloc[0]
                
                # Get 2024 performance data that should justify 2025 salary
                perf_2024_player = perf_2024[perf_2024['normalized_name'] == player_name]
                
                # Use fuzzy matching if exact match fails
                if len(perf_2024_player) == 0:
                    perf_2024_player = fuzzy_match_single_player(player_name, perf_2024)
                
                salary_change_record = {
                    'player': sal_24['full_name'],
                    'normalized_name': player_name,
                    'salary_2024': sal_24['guaranteed_compensation'],
                    'salary_2025': sal_25['guaranteed_compensation'],
                    'salary_change': sal_25['guaranteed_compensation'] - sal_24['guaranteed_compensation'],
                    'salary_change_pct': ((sal_25['guaranteed_compensation'] - sal_24['guaranteed_compensation']) / sal_24['guaranteed_compensation'] * 100) if sal_24['guaranteed_compensation'] > 0 else 0,
                    'performance_2024': perf_2024_player.iloc[0]['performance_score'] if len(perf_2024_player) > 0 else 0,
                    'value_ratio_2024': perf_2024_player.iloc[0]['value_ratio'] if len(perf_2024_player) > 0 else 0,
                    'minutes_2024': perf_2024_player.iloc[0]['Playing Time Min'] if len(perf_2024_player) > 0 else 0,
                    'has_performance_data': len(perf_2024_player) > 0,
                    'correlation_type': '2024 Performance → 2025 Salary'
                }
                salary_changes_2025.append(salary_change_record)
        
        # Calculate team totals
        total_2024_salary = salary_2024['guaranteed_compensation'].sum() if len(salary_2024) > 0 else 0
        total_2025_salary = salary_2025['guaranteed_compensation'].sum() if len(salary_2025) > 0 else 0
        
        # Get the display team name (original, not normalized)
        display_team_name = team_data['Squad'].iloc[0] if len(team_data) > 0 else team
        
        team_analysis[display_team_name] = {
            'normalized_team': team,
            'stayed_players': stayed_players,
            'left_players': left_players,
            'new_players': new_players,
            'salary_changes_2025': salary_changes_2025,  # Main focus: performance → next salary
            'total_2024_salary': total_2024_salary,
            'total_2025_salary': total_2025_salary,
            'payroll_change': total_2025_salary - total_2024_salary,
            'payroll_change_pct': ((total_2025_salary - total_2024_salary) / total_2024_salary * 100) if total_2024_salary > 0 else 0,
            'players_with_salary_changes': len(salary_changes_2025),
            'players_with_performance_data': len([c for c in salary_changes_2025 if c['has_performance_data']])
        }
    
    return team_analysis

def process_data(salary_df, performance_df):
    """Process and merge salary and performance data with comprehensive cleaning"""
    
    # Position mapping
    position_mapping = {
        'CB': 'DF', 'LB': 'DF', 'RB': 'DF',
        'CDM': 'MF', 'CM': 'MF', 'CAM': 'MF', 'LM': 'MF', 'RM': 'MF',
        'ST': 'FW', 'LW': 'FW', 'RW': 'FW',
        'GK': 'GK'
    }
    
    # Clean year data in salary
    salary_df = clean_year_data(salary_df, 'salary_year')
    
    # Normalize player and team names
    salary_df['normalized_name'] = salary_df['full_name'].apply(normalize_name)
    salary_df['normalized_team'] = salary_df['club_standard'].apply(normalize_team_name)
    salary_df['general_position'] = salary_df['position_code'].map(position_mapping)
    
    # Remove duplicates in salary data (same player, same team, same year)
    salary_df = salary_df.drop_duplicates(
        subset=['normalized_name', 'normalized_team', 'salary_year'], 
        keep='first'
    ).copy()
    
    # Clean year data in performance
    performance_df = clean_year_data(performance_df, 'Season')
    
    # Normalize player and team names
    performance_df['normalized_name'] = performance_df['Player'].apply(normalize_name)
    performance_df['normalized_team'] = performance_df['Squad'].apply(normalize_team_name)
    
    # Remove duplicates in performance data (same player, same team, same season)
    performance_df = performance_df.drop_duplicates(
        subset=['normalized_name', 'normalized_team', 'Season'], 
        keep='last'  # Keep the last entry in case of multiple records
    ).copy()
    
    # Create comprehensive player tracking
    all_players_data = []
    match_summary = {'perfect_matches': 0, 'name_only_matches': 0, 'no_matches': 0}
    
    # Get all unique players from salary data
    for _, salary_player in salary_df.iterrows():
        player_name = salary_player['normalized_name']
        player_team = salary_player['normalized_team']
        
        # Find performance records for this player
        # First try: exact name and team match
        exact_matches = performance_df[
            (performance_df['normalized_name'] == player_name) & 
            (performance_df['normalized_team'] == player_team)
        ]
        
        # Second try: name match only (for transfers)
        name_matches = performance_df[performance_df['normalized_name'] == player_name]
        
        # Use exact matches if available, otherwise name matches
        player_performances = exact_matches if len(exact_matches) > 0 else name_matches
        
        if len(player_performances) > 0:
            if len(exact_matches) > 0:
                match_summary['perfect_matches'] += 1
            else:
                match_summary['name_only_matches'] += 1
            
            # Process each season for this player
            for season in player_performances['Season'].unique():
                season_performances = player_performances[player_performances['Season'] == season]
                
                # Handle multiple teams in one season (transfers)
                if len(season_performances) > 1:
                    # Use the team with most minutes played
                    main_team_perf = season_performances.loc[
                        season_performances['Playing Time Min'].idxmax()
                    ].copy()
                    
                    # Mark as transfer case
                    main_team_perf['had_transfer'] = True
                    main_team_perf['transfer_teams'] = ', '.join(
                        season_performances['Squad'].unique()
                    )
                    main_team_perf['transfer_count'] = len(season_performances)
                else:
                    main_team_perf = season_performances.iloc[0].copy()
                    main_team_perf['had_transfer'] = False
                    main_team_perf['transfer_teams'] = main_team_perf['Squad']
                    main_team_perf['transfer_count'] = 1
                
                # Combine salary and performance data
                combined_record = {**salary_player.to_dict(), **main_team_perf.to_dict()}
                combined_record['performance_season'] = int(season)
                combined_record['salary_year'] = int(salary_player.get('salary_year', 2025))
                
                # Create meaningful season context
                perf_year = int(season)
                sal_year = int(salary_player.get('salary_year', 2025))
                combined_record['season_context'] = f"{perf_year} Performance → {sal_year} Salary"
                
                # Check for team consistency
                salary_team = salary_player['normalized_team']
                perf_team = main_team_perf['normalized_team']
                combined_record['team_consistent'] = (salary_team == perf_team)
                
                all_players_data.append(combined_record)
        else:
            match_summary['no_matches'] += 1
    
    if len(all_players_data) == 0:
        st.error("No matches found between salary and performance data!")
        return pd.DataFrame()
    
    comprehensive_df = pd.DataFrame(all_players_data)
    
    # Final duplicate removal based on key identifying fields
    comprehensive_df = comprehensive_df.drop_duplicates(
        subset=['normalized_name', 'normalized_team', 'performance_season', 'salary_year'],
        keep='first'
    ).copy()
    
    # Filter qualified players and calculate metrics
    qualified = comprehensive_df[
        (comprehensive_df['Playing Time MP'] >= 5) & 
        (comprehensive_df['Playing Time Min'] >= 450) & 
        (comprehensive_df['guaranteed_compensation'] > 0)
    ].copy()
    
    if len(qualified) == 0:
        st.error("No players meet the qualification criteria!")
        return pd.DataFrame()
    
    # Calculate performance scores
    qualified['performance_score'] = qualified.apply(calculate_performance_score, axis=1)
    qualified['value_ratio'] = qualified['performance_score'] / (qualified['guaranteed_compensation'] / 1000000)
    
    # Final cleaning - ensure years are integers
    qualified['performance_season'] = qualified['performance_season'].astype(int)
    qualified['salary_year'] = qualified['salary_year'].astype(int)
    
    return qualified

if salary_file and performance_file:
    # Load data
    try:
        salary_df = pd.read_csv(salary_file)
        performance_df = pd.read_csv(performance_file)
        
        # Validate required columns
        required_salary_cols = ['full_name', 'guaranteed_compensation', 'position_code', 'club_standard']
        required_perf_cols = ['Player', 'Playing Time MP', 'Playing Time Min', 'Season']
        
        missing_salary_cols = [col for col in required_salary_cols if col not in salary_df.columns]
        missing_perf_cols = [col for col in required_perf_cols if col not in performance_df.columns]
        
        if missing_salary_cols:
            st.error(f"Missing columns in salary data: {missing_salary_cols}")
            st.stop()
        
        if missing_perf_cols:
            st.error(f"Missing columns in performance data: {missing_perf_cols}")
            st.stop()
        
        # Process data
        with st.spinner("Processing and merging data..."):
            merged_df = process_data(salary_df, performance_df)
        
        if len(merged_df) == 0:
            st.error("No players could be matched between salary and performance data. Please check player name formats.")
            st.stop()
        
        # Sidebar team filter - use original team names for display
        st.sidebar.header("Team Analysis")
        available_teams = sorted(merged_df['Squad'].unique())
        selected_team = st.sidebar.selectbox(
            "Select Team for Detailed Analysis", 
            options=['All Teams'] + available_teams
        )
        
        # Season filter - ensure clean integer years
        available_seasons = sorted([int(s) for s in merged_df['performance_season'].unique()])
        selected_seasons = st.sidebar.multiselect(
            "Select Seasons to Compare",
            options=available_seasons,
            default=available_seasons
        )
        
        # Filter data based on selections using original team names
        if selected_team != 'All Teams':
            team_filtered_df = merged_df[merged_df['Squad'] == selected_team]
        else:
            team_filtered_df = merged_df
        
        season_filtered_df = team_filtered_df[team_filtered_df['performance_season'].isin(selected_seasons)]
        
        # Remove any remaining duplicates after filtering
        season_filtered_df = season_filtered_df.drop_duplicates(
            subset=['normalized_name', 'performance_season', 'normalized_team'],
            keep='first'
        ).copy()
        
        # Main dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Players", len(season_filtered_df))
        with col2:
            avg_salary = season_filtered_df['guaranteed_compensation'].mean()
            st.metric("Avg Salary", f"${avg_salary:,.0f}")
        with col3:
            avg_performance = season_filtered_df['performance_score'].mean()
            st.metric("Avg Performance Score", f"{avg_performance:.2f}")
        with col4:
            avg_value = season_filtered_df['value_ratio'].mean()
            st.metric("Avg Value Ratio", f"{avg_value:.2f}")
        
        # Multi-season timeline view
        if len(selected_seasons) > 1 and len(season_filtered_df) > 0:
            st.header("Multi-Season Timeline Analysis")
            
            try:
                # Show performance progression over seasons
                season_comparison = season_filtered_df.groupby(['performance_season', 'general_position']).agg({
                    'performance_score': 'mean',
                    'guaranteed_compensation': 'mean',
                    'value_ratio': 'mean'
                }).reset_index()
                
                if len(season_comparison) > 0:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_timeline1 = px.line(
                            season_comparison, 
                            x='performance_season', 
                            y='performance_score',
                            color='general_position',
                            title='Performance Score Trends by Position',
                            markers=True
                        )
                        st.plotly_chart(fig_timeline1, use_container_width=True)
                    
                    with col2:
                        fig_timeline2 = px.line(
                            season_comparison, 
                            x='performance_season', 
                            y='guaranteed_compensation',
                            color='general_position',
                            title='Salary Trends by Position',
                            markers=True
                        )
                        st.plotly_chart(fig_timeline2, use_container_width=True)
                else:
                    st.warning("Insufficient data for timeline analysis with current filters.")
            except Exception as e:
                st.warning(f"Timeline analysis unavailable: {str(e)}")
        
        # Transfer analysis
        st.header("Transfer Impact Analysis")
        
        # Use proper filtered data for transfer analysis
        if selected_team != 'All Teams':
            transfer_players = season_filtered_df[season_filtered_df.get('had_transfer', False) == True]
        else:
            transfer_players = merged_df[merged_df.get('had_transfer', False) == True]
            
        if len(transfer_players) > 0:
            st.subheader(f"Players with Mid-Season Transfers ({len(transfer_players)} cases)")
            
            transfer_display = transfer_players[['full_name', 'position_code', 'transfer_teams', 
                                              'performance_season', 'performance_score', 'guaranteed_compensation']].copy()
            transfer_display['guaranteed_compensation'] = transfer_display['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
            transfer_display.columns = ['Player', 'Position', 'Teams Played For', 'Season', 'Performance Score', 'Salary']
            
            st.dataframe(transfer_display, use_container_width=True, hide_index=True)
        else:
            st.info("No mid-season transfers detected in the filtered data.")
        
        # Position analysis (updated to use filtered data)
        st.header("Position-Based Analysis")
        
        # Create position summary using filtered data
        position_data = []
        
        for pos in season_filtered_df['general_position'].unique():
            pos_players = season_filtered_df[season_filtered_df['general_position'] == pos]
            
            if len(pos_players) > 0:
                row = {
                    'Position': pos,
                    'Count': len(pos_players),
                    'Avg Salary': f"${pos_players['guaranteed_compensation'].mean():,.0f}",
                    'Median Salary': f"${pos_players['guaranteed_compensation'].median():,.0f}",
                    'Min Salary': f"${pos_players['guaranteed_compensation'].min():,.0f}",
                    'Max Salary': f"${pos_players['guaranteed_compensation'].max():,.0f}",
                    'Avg Performance': round(pos_players['performance_score'].mean(), 2),
                    'Avg Value Ratio': round(pos_players['value_ratio'].mean(), 2)
                }
                
                # Add performance metrics if available
                if 'Per 90 Minutes Gls' in season_filtered_df.columns:
                    row['Goals/90'] = round(pos_players['Per 90 Minutes Gls'].mean(), 2)
                if 'Per 90 Minutes Ast' in season_filtered_df.columns:
                    row['Assists/90'] = round(pos_players['Per 90 Minutes Ast'].mean(), 2)
                if 'Team Success +/-' in season_filtered_df.columns:
                    row['Team +/-'] = round(pos_players['Team Success +/-'].mean(), 1)
                
                position_data.append(row)
        
        position_summary = pd.DataFrame(position_data)
        st.dataframe(position_summary, use_container_width=True, hide_index=True)
        
        # Visualizations
        st.header("Visualizations")
        
        # Salary distribution by position
        fig1 = px.box(season_filtered_df, x='general_position', y='guaranteed_compensation',
                     title=f'Salary Distribution by Position ({", ".join(map(str, selected_seasons))})')
        fig1.update_layout(yaxis_title='Guaranteed Compensation ($)')
        st.plotly_chart(fig1, use_container_width=True)
        
        # Performance vs Salary scatter plot with season context
        fig2 = px.scatter(season_filtered_df, x='guaranteed_compensation', y='performance_score',
                         color='performance_season', 
                         symbol='general_position',
                         size='Playing Time Min',
                         hover_data=['full_name', 'Squad', 'season_context', 'value_ratio'],
                         title='Performance Score vs Salary (Multi-Season View)',
                         labels={'guaranteed_compensation': 'Salary ($)', 'performance_score': 'Performance Score'})
        fig2.update_layout(xaxis_type='log')
        st.plotly_chart(fig2, use_container_width=True)
        
        # Value ratio analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Best Value Players")
            best_value = season_filtered_df[season_filtered_df['performance_score'] > 1].nlargest(10, 'value_ratio')[
                ['full_name', 'position_code', 'Squad', 'performance_season', 'guaranteed_compensation', 
                 'performance_score', 'value_ratio', 'season_context']
            ]
            if len(best_value) > 0:
                best_value = best_value.copy()
                best_value['guaranteed_compensation'] = best_value['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
                best_value['performance_score'] = best_value['performance_score'].round(2)
                best_value['value_ratio'] = best_value['value_ratio'].round(2)
                best_value.columns = ['Player', 'Pos', 'Team', 'Season', 'Salary', 'Perf Score', 'Value Ratio', 'Context']
                st.dataframe(best_value, use_container_width=True, hide_index=True)
            else:
                st.write("No players meet the criteria in the filtered data.")
        
        with col2:
            st.subheader("Most Overpaid Players")
            overpaid = season_filtered_df[season_filtered_df['guaranteed_compensation'] > 800000].nsmallest(10, 'value_ratio')[
                ['full_name', 'position_code', 'Squad', 'performance_season', 'guaranteed_compensation', 
                 'performance_score', 'value_ratio', 'season_context']
            ]
            if len(overpaid) > 0:
                overpaid = overpaid.copy()
                overpaid['guaranteed_compensation'] = overpaid['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
                overpaid['performance_score'] = overpaid['performance_score'].round(2)
                overpaid['value_ratio'] = overpaid['value_ratio'].round(2)
                overpaid.columns = ['Player', 'Pos', 'Team', 'Season', 'Salary', 'Perf Score', 'Value Ratio', 'Context']
                st.dataframe(overpaid, use_container_width=True, hide_index=True)
            else:
                st.write("No high-salary players found in the filtered data.")
        
        # Team analysis
        st.header("Team Analysis")
        
        if selected_team != 'All Teams':
            # Detailed team analysis
            st.subheader(f"{selected_team} - Detailed Roster Analysis")
            
            # Analyze team changes
            team_changes = analyze_performance_salary_correlation(merged_df, merged_df)
            
            if selected_team in team_changes:
                team_data = team_changes[selected_team]
                
                # Team overview metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Players Stayed", len(team_data['stayed_players']))
                with col2:
                    st.metric("Players Left", len(team_data['left_players']))
                with col3:
                    st.metric("New Players", len(team_data['new_players']))
                with col4:
                    payroll_change = team_data['payroll_change']
                    st.metric("Payroll Change", f"${payroll_change:,.0f}")
                
                # Roster changes breakdown
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.subheader("Players Who Stayed")
                    if team_data.get('salary_changes_2025'):
                        stayed_df = pd.DataFrame(team_data['salary_changes_2025'])
                        stayed_df['Salary Change'] = stayed_df['salary_change'].apply(lambda x: f"${x:,.0f}")
                        stayed_df['Change Pct'] = stayed_df['salary_change_pct'].apply(lambda x: f"{x:.1f}%")
                        stayed_df['Performance 2024'] = stayed_df['performance_2024'].round(2)
                        
                        display_stayed = stayed_df[['player', 'Salary Change', 'Change Pct', 'Performance 2024']].copy()
                        display_stayed.columns = ['Player', 'Salary Change $', 'Salary Change %', 'Performance Score']
                        
                        st.dataframe(display_stayed, hide_index=True, use_container_width=True)
                    else:
                        st.write("No players stayed between seasons")
                
                with col2:
                    st.subheader("Players Who Left")
                    if team_data['left_players']:
                        left_players_data = []
                        normalized_selected_team = normalize_team_name(selected_team)
                        
                        for player in team_data['left_players']:
                            player_info = merged_df[
                                (merged_df['normalized_name'] == player) & 
                                (merged_df['normalized_team'] == normalized_selected_team) & 
                                (merged_df['salary_year'] == 2024)
                            ]
                            if len(player_info) > 0:
                                p = player_info.iloc[0]
                                left_players_data.append({
                                    'Player': p['full_name'],
                                    'Position': p['position_code'],
                                    'Last Salary': f"${p['guaranteed_compensation']:,.0f}",
                                    'Performance': round(p['performance_score'], 2),
                                    'Value Ratio': round(p['value_ratio'], 2)
                                })
                        
                        if left_players_data:
                            st.dataframe(pd.DataFrame(left_players_data), hide_index=True, use_container_width=True)
                        else:
                            st.write("No detailed data available for players who left")
                    else:
                        st.write("No players left")
                
                with col3:
                    st.subheader("New Players")
                    if team_data['new_players']:
                        new_players_data = []
                        normalized_selected_team = normalize_team_name(selected_team)
                        
                        for player in team_data['new_players']:
                            player_info = merged_df[
                                (merged_df['normalized_name'] == player) & 
                                (merged_df['normalized_team'] == normalized_selected_team) & 
                                (merged_df['salary_year'] == 2025)
                            ]
                            if len(player_info) > 0:
                                p = player_info.iloc[0]
                                new_players_data.append({
                                    'Player': p['full_name'],
                                    'Position': p['position_code'],
                                    'Salary': f"${p['guaranteed_compensation']:,.0f}",
                                    'Performance': round(p.get('performance_score', 0), 2),
                                    'Value Ratio': round(p.get('value_ratio', 0), 2)
                                })
                        
                        if new_players_data:
                            st.dataframe(pd.DataFrame(new_players_data), hide_index=True, use_container_width=True)
                        else:
                            st.write("No detailed data available for new players")
                    else:
                        st.write("No new players")
                
                # Show salary changes for 2024 → 2025
                if team_data.get('salary_changes_2025'):
                    st.subheader("2024 Performance → 2025 Salary Changes")
                    
                    salary_changes_df = pd.DataFrame(team_data['salary_changes_2025'])
                    
                    # Create display dataframe
                    display_salary_changes = salary_changes_df.copy()
                    display_salary_changes['Salary 2024'] = display_salary_changes['salary_2024'].apply(lambda x: f"${x:,.0f}")
                    display_salary_changes['Salary 2025'] = display_salary_changes['salary_2025'].apply(lambda x: f"${x:,.0f}")
                    display_salary_changes['Change $'] = display_salary_changes['salary_change'].apply(lambda x: f"${x:,.0f}")
                    display_salary_changes['Change %'] = display_salary_changes['salary_change_pct'].apply(lambda x: f"{x:.1f}%")
                    display_salary_changes['2024 Performance'] = display_salary_changes['performance_2024'].round(2)
                    display_salary_changes['2024 Value Ratio'] = display_salary_changes['value_ratio_2024'].round(2)
                    
                    display_cols = ['player', 'Salary 2024', 'Salary 2025', 'Change $', 'Change %', '2024 Performance', '2024 Value Ratio']
                    final_display = display_salary_changes[display_cols].copy()
                    final_display.columns = ['Player', 'Salary 2024', 'Salary 2025', 'Change $', 'Change %', 'Performance Score', 'Value Ratio']
                    
                    # Sort by salary change amount (descending)
                    final_display_sorted = final_display.copy()
                    final_display_sorted['sort_key'] = salary_changes_df['salary_change']
                    final_display_sorted = final_display_sorted.sort_values('sort_key', ascending=False).drop('sort_key', axis=1)
                    
                    st.dataframe(final_display_sorted, hide_index=True, use_container_width=True)
                    
                    # Summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        avg_change = salary_changes_df['salary_change'].mean()
                        st.metric("Average Salary Change", f"${avg_change:,.0f}")
                    with col2:
                        raises_count = (salary_changes_df['salary_change'] > 0).sum()
                        st.metric("Players with Raises", raises_count)
                    with col3:
                        cuts_count = (salary_changes_df['salary_change'] < 0).sum()
                        st.metric("Players with Cuts", cuts_count)
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Salary change bar chart
                        fig_salary = px.bar(
                            salary_changes_df.sort_values('salary_change'), 
                            x='salary_change', 
                            y='player',
                            title=f'{selected_team} - Salary Changes (2024 → 2025)',
                            labels={'salary_change': 'Salary Change ($)', 'player': 'Player'},
                            color='salary_change',
                            color_continuous_scale=['red', 'white', 'green']
                        )
                        fig_salary.update_layout(height=max(400, len(salary_changes_df) * 25))
                        st.plotly_chart(fig_salary, use_container_width=True)
                    
                    with col2:
                        # Performance vs salary change correlation
                        fig_correlation = px.scatter(
                            salary_changes_df,
                            x='performance_2024',
                            y='salary_change',
                            hover_data=['player', 'salary_change_pct'],
                            title='2024 Performance vs 2025 Salary Change',
                            labels={
                                'performance_2024': '2024 Performance Score', 
                                'salary_change': 'Salary Change ($)',
                                'salary_change_pct': 'Salary Change %'
                            }
                        )
                        st.plotly_chart(fig_correlation, use_container_width=True)
                    
                    # Identify interesting cases
                    st.subheader("Notable Cases")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Biggest Raises:**")
                        biggest_raises = salary_changes_df.nlargest(3, 'salary_change')[['player', 'salary_change', 'performance_2024']]
                        for _, row in biggest_raises.iterrows():
                            st.write(f"• {row['player']}: +${row['salary_change']:,.0f} (Performance: {row['performance_2024']:.2f})")
                    
                    with col2:
                        st.write("**Performance vs Salary Mismatches:**")
                        # Find players with high performance but low/no raises
                        high_perf_low_raise = salary_changes_df[
                            (salary_changes_df['performance_2024'] > salary_changes_df['performance_2024'].median()) & 
                            (salary_changes_df['salary_change'] < salary_changes_df['salary_change'].median())
                        ].nlargest(3, 'performance_2024')[['player', 'salary_change', 'performance_2024']]
                        
                        if len(high_perf_low_raise) > 0:
                            for _, row in high_perf_low_raise.iterrows():
                                st.write(f"• {row['player']}: +${row['salary_change']:,.0f} (Performance: {row['performance_2024']:.2f})")
                        else:
                            st.write("No clear mismatches found")
                
                else:
                    st.info("No salary change data available for this team")
            
            else:
                st.warning(f"No analysis data found for {selected_team}")
        
        else:
            # League-wide team analysis
            team_summary = season_filtered_df.groupby('Squad').agg({
                'guaranteed_compensation': ['count', 'sum', 'mean'],
                'performance_score': 'mean',
                'value_ratio': 'mean',
                'Team Success +/-': 'mean'
            }).round(2)
            
            team_summary.columns = ['Player Count', 'Total Payroll', 'Avg Salary', 'Avg Performance', 'Avg Value Ratio', 'Avg Team +/-']
            team_summary = team_summary.sort_values('Avg Value Ratio', ascending=False)
            
            st.dataframe(team_summary, use_container_width=True)
        
        # Interactive filters
        st.header("Interactive Player Explorer")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            position_filter = st.multiselect(
                "Select Positions", 
                options=season_filtered_df['general_position'].unique(),
                default=season_filtered_df['general_position'].unique()
            )
        
        with col2:
            salary_range = st.slider(
                "Salary Range (millions)",
                min_value=0.0,
                max_value=float(season_filtered_df['guaranteed_compensation'].max() / 1000000),
                value=(0.0, float(season_filtered_df['guaranteed_compensation'].max() / 1000000)),
                step=0.1
            )
        
        with col3:
            min_performance = st.slider(
                "Minimum Performance Score",
                min_value=0.0,
                max_value=float(season_filtered_df['performance_score'].max()),
                value=0.0,
                step=0.1
            )
        
        # Apply additional filters
        filtered_df = season_filtered_df[
            (season_filtered_df['general_position'].isin(position_filter)) &
            (season_filtered_df['guaranteed_compensation'] >= salary_range[0] * 1000000) &
            (season_filtered_df['guaranteed_compensation'] <= salary_range[1] * 1000000) &
            (season_filtered_df['performance_score'] >= min_performance)
        ]
        
        st.subheader(f"Filtered Results ({len(filtered_df)} players)")
        
        display_cols = ['full_name', 'position_code', 'Squad', 'performance_season', 'season_context',
                       'guaranteed_compensation', 'Per 90 Minutes Gls', 'Per 90 Minutes Ast', 
                       'Team Success +/-', 'performance_score', 'value_ratio']
        
        # Filter for columns that exist
        existing_cols = [col for col in display_cols if col in filtered_df.columns]
        display_df = filtered_df[existing_cols].copy()
        
        if 'guaranteed_compensation' in display_df.columns:
            display_df['guaranteed_compensation'] = display_df['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
        
        # Round numeric columns
        for col in display_df.columns:
            if display_df[col].dtype in ['float64', 'int64'] and col != 'guaranteed_compensation':
                display_df[col] = display_df[col].round(2)
        
        # Rename columns for display
        display_df.columns = [col.replace('_', ' ').title() for col in display_df.columns]
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Download processed data
        st.header("Download Results")
        
        @st.cache_data
        def convert_df(df):
            return df.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(season_filtered_df)
        
        st.download_button(
            label="Download Filtered Analysis as CSV",
            data=csv,
            file_name=f'mls_salary_performance_analysis_{selected_team.replace(" ", "_")}_{"-".join(map(str, selected_seasons))}.csv',
            mime='text/csv',
        )
        
        # Key insights
        st.header("Key Insights")
        
        insights_col1, insights_col2 = st.columns(2)
        
        with insights_col1:
            st.subheader("Findings")
            
            try:
                if len(position_summary) > 0:
                    # Find best value position
                    best_value_idx = position_summary['Avg Value Ratio'].idxmax()
                    best_value_position = position_summary.loc[best_value_idx, 'Position']
                    best_value_ratio = position_summary.loc[best_value_idx, 'Avg Value Ratio']
                    
                    # Find worst value position
                    worst_value_idx = position_summary['Avg Value Ratio'].idxmin()
                    worst_value_position = position_summary.loc[worst_value_idx, 'Position']
                    worst_value_ratio = position_summary.loc[worst_value_idx, 'Avg Value Ratio']
                    
                    # Find highest paid position - use raw salary data
                    pos_salaries = season_filtered_df.groupby('general_position')['guaranteed_compensation'].mean()
                    highest_paid_position = pos_salaries.idxmax()
                    
                    st.write(f"• **Best value position**: {best_value_position} (Ratio: {best_value_ratio})")
                    st.write(f"• **Worst value position**: {worst_value_position} (Ratio: {worst_value_ratio})")
                    st.write(f"• **Highest paid position**: {highest_paid_position}")
                
                st.write(f"• **Total salary analyzed**: ${season_filtered_df['guaranteed_compensation'].sum():,.0f}")
                st.write(f"• **Players analyzed**: {len(season_filtered_df)}")
                st.write(f"• **Average value ratio**: {season_filtered_df['value_ratio'].mean():.2f}")
                st.write(f"• **Seasons covered**: {', '.join(map(str, selected_seasons))}")
                
                if selected_team != 'All Teams':
                    st.write(f"• **Team focus**: {selected_team}")
                
            except Exception as e:
                st.write("• **Analysis Summary:**")
                st.write(f"  - Total players: {len(season_filtered_df)}")
                st.write(f"  - Total salary: ${season_filtered_df['guaranteed_compensation'].sum():,.0f}")
                st.write(f"  - Avg performance score: {season_filtered_df['performance_score'].mean():.2f}")
                st.write(f"  - Avg value ratio: {season_filtered_df['value_ratio'].mean():.2f}")
        
        with insights_col2:
            st.subheader("Recommendations")
            
            if selected_team != 'All Teams':
                st.write("**Team-Specific Recommendations:**")
                st.write("• Review salary increases for players with declining performance")
                st.write("• Consider performance bonuses tied to team success metrics")
                st.write("• Analyze transfer patterns to optimize roster construction")
                st.write("• Focus on retaining high-value ratio players")
            else:
                st.write("**League-Wide Recommendations:**")
                st.write("• Focus on young player development for better value")
                st.write("• Consider team impact metrics, not just individual stats")
                st.write("• Reassess spending on high-salary, low-impact players")
                st.write("• Invest more in goalkeeping - highest team impact per dollar")
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.write("Please ensure your CSV files have the correct column names and format.")

else:
    st.info("Please upload both salary and performance CSV files to begin the analysis")
    
    with st.expander("Required Data Format"):
        st.subheader("Salary Data CSV should include:")
        st.code("""
        - full_name: Player's full name
        - guaranteed_compensation: Total compensation
        - position_code: Position (CB, LW, ST, etc.)
        - club_standard: Team name
        - salary_tier: Salary tier classification
        """)
        
        st.subheader("Performance Data CSV should include:")
        st.code("""
        - Player: Player's name
        - Squad: Team name
        - Pos: Position
        - Playing Time MP: Matches played
        - Playing Time Min: Minutes played
        - Per 90 Minutes Gls: Goals per 90 minutes
        - Per 90 Minutes Ast: Assists per 90 minutes
        - Team Success +/-: Plus/minus rating
        - Season: Year of data
        """)