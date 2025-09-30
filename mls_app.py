import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import unicodedata
from difflib import SequenceMatcher

# Page configuration
st.set_page_config(
    page_title="MLS Salary vs Performance Analysis", 
    page_icon="⚽", 
    layout="wide"
)

# Title and description
st.title("⚽ MLS Salary vs Performance Analysis")
st.markdown("Analyze how MLS player performance in one season affects their salary in the next season")

# Function to load data from GitHub
@st.cache_data
def load_data_from_github():
    """Load CSV files directly from GitHub"""
    try:
        # GitHub raw URLs for the CSV files
        performance_url = "https://raw.githubusercontent.com/ashmeetanand13/MLS-Salary/main/mls_23_24.csv"
        salary_url = "https://raw.githubusercontent.com/ashmeetanand13/MLS-Salary/main/mlspa_salary.csv"
        
        # Load the data
        performance_file = pd.read_csv(performance_url)
        salary_file = pd.read_csv(salary_url)
        
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
    
    # Handle common name variations
    name = name.replace('william', 'willy')  # William Agada -> Willy Agada
    name = name.replace('joseph', 'joe')
    name = name.replace('alexander', 'alex')
    name = name.replace('nicholas', 'nick')
    name = name.replace('matthew', 'matt')
    
    return name.strip()

def fuzzy_match_names(name1, name2, threshold=0.85):
    """Use fuzzy matching for names that don't match exactly"""
    return SequenceMatcher(None, name1, name2).ratio() >= threshold

def create_name_mapping(salary_df, performance_df):
    """Create a mapping between salary and performance names"""
    salary_names = salary_df['full_name'].dropna().unique()
    performance_names = performance_df['Player'].dropna().unique()
    
    name_mapping = {}
    
    for perf_name in performance_names:
        norm_perf = normalize_name(perf_name)
        best_match = None
        best_score = 0
        
        for sal_name in salary_names:
            norm_sal = normalize_name(sal_name)
            
            # Check exact match first
            if norm_perf == norm_sal:
                name_mapping[perf_name] = sal_name
                break
            
            # Check fuzzy match
            score = SequenceMatcher(None, norm_perf, norm_sal).ratio()
            if score > best_score and score >= 0.85:
                best_score = score
                best_match = sal_name
        
        if perf_name not in name_mapping and best_match:
            name_mapping[perf_name] = best_match
    
    return name_mapping

def calculate_performance_score(row):
    """Calculate position-specific performance score"""
    pos = row.get('general_position', 'UNKNOWN')
    
    # Get key metrics with safe defaults
    goals_90 = row.get('Per 90 Minutes Gls', 0) or 0
    assists_90 = row.get('Per 90 Minutes Ast', 0) or 0
    xg_90 = row.get('Per 90 Minutes xG', 0) or 0
    xag_90 = row.get('Per 90 Minutes xAG', 0) or 0
    team_plus_minus = row.get('Team Success +/-', 0) or 0
    minutes = row.get('Playing Time Min', 0) or 0
    
    # Adjust for playing time (minimum 450 minutes)
    if minutes < 450:
        return 0
    
    if pos == 'FW':
        # Forwards: prioritize goals and assists
        score = (goals_90 * 100 + 
                assists_90 * 60 +
                xg_90 * 40 +
                team_plus_minus * 0.5)
    elif pos == 'MF':
        # Midfielders: balanced scoring
        score = (goals_90 * 60 + 
                assists_90 * 80 +
                xag_90 * 50 +
                team_plus_minus * 0.8)
    elif pos == 'DF':
        # Defenders: team success and defensive actions
        tackles = row.get('Tackles Tkl', 0) or 0
        interceptions = row.get('Performance Int', 0) or 0
        score = (goals_90 * 40 + 
                assists_90 * 50 +
                team_plus_minus * 1.5 +
                (tackles + interceptions) * 0.5)
    elif pos == 'GK':
        # Goalkeepers: team success and save-based metrics
        score = team_plus_minus * 2 + 50  # Base score for GK
    else:
        score = 0
    
    return max(0, score)

def process_data(salary_df, performance_df):
    """Process and merge salary and performance data with year-to-year matching"""
    
    # Position mapping
    position_mapping = {
        'CB': 'DF', 'LB': 'DF', 'RB': 'DF', 'LWB': 'DF', 'RWB': 'DF',
        'CDM': 'MF', 'CM': 'MF', 'CAM': 'MF', 'LM': 'MF', 'RM': 'MF', 'AM': 'MF',
        'ST': 'FW', 'LW': 'FW', 'RW': 'FW', 'CF': 'FW',
        'GK': 'GK'
    }
    
    # Create name mapping
    name_mapping = create_name_mapping(salary_df, performance_df)
    
    # Apply name mapping to performance data
    performance_df['matched_name'] = performance_df['Player'].map(name_mapping).fillna(performance_df['Player'])
    
    # Prepare salary data
    salary_df['normalized_name'] = salary_df['full_name'].apply(normalize_name)
    salary_df['general_position'] = salary_df['position_code'].map(position_mapping).fillna('UNKNOWN')
    
    # Prepare performance data
    performance_df['normalized_name'] = performance_df['matched_name'].apply(normalize_name)
    
    # Map positions from performance data
    pos_mapping_perf = {'DF': 'DF', 'MF': 'MF', 'FW': 'FW', 'GK': 'GK', 
                        'DF,MF': 'DF', 'MF,FW': 'MF', 'FW,MF': 'FW'}
    performance_df['general_position'] = performance_df['Pos'].map(pos_mapping_perf).fillna('UNKNOWN')
    
    # Create two datasets: 2023 performance -> 2024 salary, 2024 performance -> 2025 salary
    merged_data = []
    
    # 1. Match 2023 performance with 2024 salary
    perf_2023 = performance_df[performance_df['Season'] == 2023].copy()
    salary_2024 = salary_df[salary_df['salary_year'] == 2024].copy()
    
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
            (merged['Playing Time Min'] >= 450) & 
            (merged['guaranteed_compensation'] > 0)
        ].copy()
        
        # Calculate performance scores
        qualified['performance_score'] = qualified.apply(calculate_performance_score, axis=1)
        qualified['value_ratio'] = qualified['performance_score'] / (qualified['guaranteed_compensation'] / 100000)
        
        # Add display name
        qualified['display_name'] = qualified['matched_name']
        
        return qualified
    else:
        return pd.DataFrame()

# Main app logic
if salary_df and performance_df:
    try:
        # Load data
        salary_df = pd.read_csv(salary_file)
        performance_df = pd.read_csv(performance_file)
        
        # Display initial data info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 Loaded {len(salary_df)} salary records")
            st.caption(f"Years: {sorted(salary_df['salary_year'].unique())}")
        with col2:
            st.info(f"⚽ Loaded {len(performance_df)} performance records")
            st.caption(f"Seasons: {sorted(performance_df['Season'].unique())}")
        
        # Process data
        with st.spinner("Processing and matching player data..."):
            merged_df = process_data(salary_df, performance_df)
        
        if len(merged_df) == 0:
            st.error("No matching players found between datasets. Please check your data files.")
        else:
            st.success(f"✅ Successfully matched and analyzed {len(merged_df)} player-season combinations")
            
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
            
            # Position analysis
            st.header("🏃 Position-Based Analysis")
            
            position_stats = analysis_df.groupby('general_position').agg({
                'guaranteed_compensation': ['count', 'mean', 'median'],
                'Per 90 Minutes Gls': 'mean',
                'Per 90 Minutes Ast': 'mean',
                'performance_score': 'mean',
                'value_ratio': 'mean'
            }).round(2)
            
            position_stats.columns = ['Count', 'Avg Salary', 'Median Salary', 
                                     'Goals/90', 'Assists/90', 'Perf Score', 'Value Ratio']
            
            # Format salary columns
            for col in ['Avg Salary', 'Median Salary']:
                position_stats[col] = position_stats[col].apply(lambda x: f"${x:,.0f}")
            
            st.dataframe(position_stats, use_container_width=True)
            
            # Visualizations
            st.header("📈 Performance vs Salary Analysis")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Scatter Plot", "Position Distribution", "Top Players", "Team Analysis"])
            
            with tab1:
                # Performance vs Salary scatter plot
                fig = px.scatter(
                    analysis_df,
                    x='guaranteed_compensation',
                    y='performance_score',
                    color='general_position',
                    size='Playing Time Min',
                    hover_data={
                        'display_name': True,
                        'club_standard': True,
                        'analysis_year': True,
                        'value_ratio': ':.2f',
                        'guaranteed_compensation': ':$,.0f',
                        'performance_score': ':.1f'
                    },
                    title='Performance Score vs Salary by Position',
                    labels={
                        'guaranteed_compensation': 'Guaranteed Compensation ($)',
                        'performance_score': 'Performance Score',
                        'general_position': 'Position'
                    },
                    color_discrete_map={'FW': '#FF6B6B', 'MF': '#4ECDC4', 'DF': '#45B7D1', 'GK': '#96CEB4'}
                )
                
                # Update axis to log scale
                fig.update_layout(
                    xaxis_type='log',
                    xaxis_title='Guaranteed Compensation ($) - Log Scale',
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # Position distribution
                col1, col2 = st.columns(2)
                
                with col1:
                    # Salary distribution by position
                    fig_salary = px.box(
                        analysis_df,
                        x='general_position',
                        y='guaranteed_compensation',
                        title='Salary Distribution by Position',
                        labels={'guaranteed_compensation': 'Guaranteed Compensation ($)'},
                        color='general_position',
                        color_discrete_map={'FW': '#FF6B6B', 'MF': '#4ECDC4', 'DF': '#45B7D1', 'GK': '#96CEB4'}
                    )
                    fig_salary.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_salary, use_container_width=True)
                
                with col2:
                    # Performance score distribution by position
                    fig_perf = px.box(
                        analysis_df,
                        x='general_position',
                        y='performance_score',
                        title='Performance Score Distribution by Position',
                        labels={'performance_score': 'Performance Score'},
                        color='general_position',
                        color_discrete_map={'FW': '#FF6B6B', 'MF': '#4ECDC4', 'DF': '#45B7D1', 'GK': '#96CEB4'}
                    )
                    fig_perf.update_layout(showlegend=False, height=400)
                    st.plotly_chart(fig_perf, use_container_width=True)
            
            with tab3:
                # Top players analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Best Value Players")
                    st.caption("High performance relative to salary")
                    
                    best_value = analysis_df[analysis_df['performance_score'] > 10].nlargest(10, 'value_ratio')[
                        ['display_name', 'position_code', 'club_standard', 'guaranteed_compensation', 
                         'performance_score', 'value_ratio', 'analysis_year']
                    ].copy()
                    
                    best_value['guaranteed_compensation'] = best_value['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
                    best_value['performance_score'] = best_value['performance_score'].round(1)
                    best_value['value_ratio'] = best_value['value_ratio'].round(2)
                    
                    st.dataframe(best_value, use_container_width=True, hide_index=True)
                
                with col2:
                    st.subheader("💰 Highest Impact Players")
                    st.caption("Highest performance scores overall")
                    
                    top_performers = analysis_df.nlargest(10, 'performance_score')[
                        ['display_name', 'position_code', 'club_standard', 'guaranteed_compensation', 
                         'performance_score', 'value_ratio', 'analysis_year']
                    ].copy()
                    
                    top_performers['guaranteed_compensation'] = top_performers['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
                    top_performers['performance_score'] = top_performers['performance_score'].round(1)
                    top_performers['value_ratio'] = top_performers['value_ratio'].round(2)
                    
                    st.dataframe(top_performers, use_container_width=True, hide_index=True)
            
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
                            
                            # Visualize salary changes
                            fig_changes = px.bar(
                                salary_changes_df.sort_values('Change ($)'),
                                x='Change ($)',
                                y='Player',
                                orientation='h',
                                title='Salary Changes for Returning Players',
                                color='Change ($)',
                                color_continuous_scale=['red', 'yellow', 'green'],
                                color_continuous_midpoint=0
                            )
                            fig_changes.update_layout(height=400 + len(salary_changes_df) * 15)
                            st.plotly_chart(fig_changes, use_container_width=True)
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
                        
                        # Position breakdown for team
                        st.markdown("**Position Breakdown:**")
                        position_breakdown = team_data.groupby('general_position').agg({
                            'display_name': 'count',
                            'guaranteed_compensation': 'mean',
                            'performance_score': 'mean'
                        }).round(2)
                        position_breakdown.columns = ['Count', 'Avg Salary', 'Avg Performance']
                        position_breakdown['Avg Salary'] = position_breakdown['Avg Salary'].apply(lambda x: f"${x:,.0f}")
                        st.dataframe(position_breakdown, use_container_width=True)
                    else:
                        st.warning(f"No performance data available for {selected_team}")
            
            # Interactive filters
            st.header("🔍 Player Explorer")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                position_filter = st.multiselect(
                    "Position",
                    options=analysis_df['general_position'].unique(),
                    default=analysis_df['general_position'].unique()
                )
            
            with col2:
                team_filter = st.multiselect(
                    "Team",
                    options=sorted(analysis_df['club_standard'].unique()),
                    default=[]
                )
            
            with col3:
                salary_range = st.slider(
                    "Salary Range ($K)",
                    min_value=int(analysis_df['guaranteed_compensation'].min() / 1000),
                    max_value=int(analysis_df['guaranteed_compensation'].max() / 1000),
                    value=(int(analysis_df['guaranteed_compensation'].min() / 1000), 
                           int(analysis_df['guaranteed_compensation'].max() / 1000)),
                    step=100
                )
            
            with col4:
                min_minutes = st.number_input(
                    "Min. Minutes Played",
                    min_value=0,
                    max_value=int(analysis_df['Playing Time Min'].max()),
                    value=450,
                    step=100
                )
            
            # Apply filters
            filtered_df = analysis_df[
                (analysis_df['general_position'].isin(position_filter)) &
                (analysis_df['guaranteed_compensation'] >= salary_range[0] * 1000) &
                (analysis_df['guaranteed_compensation'] <= salary_range[1] * 1000) &
                (analysis_df['Playing Time Min'] >= min_minutes)
            ]
            
            if team_filter:
                filtered_df = filtered_df[filtered_df['club_standard'].isin(team_filter)]
            
            # Display filtered results
            st.subheader(f"Filtered Results ({len(filtered_df)} records)")
            
            display_cols = ['display_name', 'position_code', 'club_standard', 'analysis_year',
                           'guaranteed_compensation', 'Playing Time Min', 'Per 90 Minutes Gls', 
                           'Per 90 Minutes Ast', 'performance_score', 'value_ratio']
            
            display_df = filtered_df[display_cols].copy()
            display_df['guaranteed_compensation'] = display_df['guaranteed_compensation'].apply(lambda x: f"${x:,.0f}")
            display_df = display_df.round(2).sort_values('performance_score', ascending=False)
            
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            
            # Download section
            st.header("💾 Export Data")
            
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(merged_df)
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📥 Download Complete Analysis (CSV)",
                    data=csv,
                    file_name='mls_salary_performance_analysis.csv',
                    mime='text/csv',
                )
            
            with col2:
                # Create summary statistics
                summary_stats = {
                    'Total Players Analyzed': len(merged_df),
                    'Average Salary': f"${merged_df['guaranteed_compensation'].mean():,.0f}",
                    'Average Performance Score': f"{merged_df['performance_score'].mean():.1f}",
                    'Best Value Position': position_stats['Value Ratio'].idxmax(),
                    'Highest Paid Position': position_stats.index[position_stats['Avg Salary'].str.replace('$', '').str.replace(',', '').astype(float).argmax()]
                }
                
                summary_text = "\n".join([f"{k}: {v}" for k, v in summary_stats.items()])
                st.download_button(
                    label="📊 Download Summary Report (TXT)",
                    data=summary_text,
                    file_name='mls_analysis_summary.txt',
                    mime='text/plain',
                )
    
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.write("Debug information:")
        st.code(str(e))
        
        with st.expander("🐛 Troubleshooting Guide"):
            st.write("""
            **Common issues and solutions:**
            
            1. **Column name mismatch**: Ensure your CSV files have the expected column names
            2. **Data type issues**: Check that numeric columns don't contain text values
            3. **Missing data**: Some players might not have complete statistics
            4. **Name matching**: Player names might differ between datasets (e.g., 'William' vs 'Willy')
            
            **Required columns:**
            - Salary CSV: full_name, guaranteed_compensation, position_code, club_standard, salary_year
            - Performance CSV: Player, Squad, Pos, Season, Playing Time MP, Playing Time Min, performance metrics
            """)

else:
    st.info("👆 Please upload both salary and performance CSV files to begin the analysis")
    
    with st.expander("📋 Data Requirements"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Salary Data CSV")
            st.write("""
            **Required columns:**
            - `full_name`: Player's full name
            - `guaranteed_compensation`: Total compensation
            - `position_code`: Position (CB, LW, ST, etc.)
            - `club_standard`: Team name
            - `salary_year`: Year of salary (2024 or 2025)
            
            **Optional columns:**
            - `base_salary`: Base salary amount
            - `bonus_amount`: Bonus compensation
            """)
        
        with col2:
            st.subheader("Performance Data CSV")
            st.write("""
            **Required columns:**
            - `Player`: Player's name
            - `Squad`: Team name
            - `Pos`: Position
            - `Season`: Year (2023 or 2024)
            - `Playing Time MP`: Matches played
            - `Playing Time Min`: Minutes played
            - `Per 90 Minutes Gls`: Goals per 90
            - `Per 90 Minutes Ast`: Assists per 90
            
            **Optional metrics:**
            - `Per 90 Minutes xG`: Expected goals
            - `Team Success +/-`: Plus/minus rating
            - Various other performance metrics
            """)
    
    with st.expander("🔄 How the Analysis Works"):
        st.write("""
        **Year-to-Year Matching:**
        
        1. **2023 Performance → 2024 Salary**: We analyze how a player's 2023 season performance influenced their 2024 salary
        2. **2024 Performance → 2025 Salary**: We analyze how a player's 2024 season performance influenced their 2025 salary
        
        **Performance Score Calculation:**
        
        The app calculates position-specific performance scores:
        - **Forwards (FW)**: Heavily weighted on goals and assists
        - **Midfielders (MF)**: Balanced between goals, assists, and playmaking
        - **Defenders (DF)**: Team success, defensive actions, and clean sheets
        - **Goalkeepers (GK)**: Team success and save metrics
        
        **Value Ratio:**
        
        Value Ratio = Performance Score / (Salary in $100K)
        
        Higher value ratios indicate players providing more performance per dollar spent.
        """)

# Footer
st.markdown("---")
st.markdown("**MLS Salary vs Performance Analysis** | Data: 2023-2025 | Built with Streamlit & Plotly")
