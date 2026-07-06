import os
import pandas as pd
import numpy as np

# Absolute path resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DF1_PATH = os.path.join(BASE_DIR, "Daily_Power_Gen_Source_march_23.csv")
DF2_PATH = os.path.join(BASE_DIR, "Daily_Power_Gen_States_march_23.csv")

# Load datasets globally for service caching
df1 = pd.read_csv(DF1_PATH)
df1_source = df1.drop(df1[df1['source'] == 'Total'].index).copy()
df1_source['date'] = pd.to_datetime(df1_source['date'])

df2 = pd.read_csv(DF2_PATH)
df2['date'] = pd.to_datetime(df2['date'])
df2['Energy Met (MU)'] = df2['Energy Met (MU)'].fillna(0)
df2['Shortage during maximum Demand(MW)'] = df2['Shortage during maximum Demand(MW)'].fillna(0)

# Extract lists for validation
states_list = sorted(df2['States'].dropna().unique().tolist())
sources_list = sorted(df1_source['source'].unique().tolist())
regions_list = ['NR', 'WR', 'SR', 'ER', 'NER']

def get_monthly_series(state='All India', source='All Sources'):
    """
    Downscale and aggregate historical power data into a monthly series (date, demand) in MU.
    If state is 'All India' and source is 'All Sources', returns All India monthly total.
    If source is specific, downscales the region's source-generation by the state's share.
    """
    state = state.strip() if state else 'All India'
    source = source.strip() if source else 'All Sources'
    
    # 1. State: All India / All, Source: All Sources / All
    if state in ['All India', 'All'] and source in ['All Sources', 'All']:
        df_group = df2.groupby('date')['Energy Met (MU)'].sum().reset_index()
        df_monthly = df_group.resample('ME', on='date').sum().reset_index()
        df_monthly.rename(columns={'Energy Met (MU)': 'demand'}, inplace=True)
        return df_monthly[['date', 'demand']]
        
    # 2. State: All India / All, Source: Specific Source
    elif state in ['All India', 'All']:
        df_src = df1_source[df1_source['source'] == source]
        if df_src.empty:
            # Fallback if source name slightly different
            matching = [s for s in sources_list if source.lower() in s.lower()]
            if matching:
                df_src = df1_source[df1_source['source'] == matching[0]]
                
        if df_src.empty:
            return pd.DataFrame(columns=['date', 'demand'])
            
        df_monthly = df_src.resample('ME', on='date')['All India'].sum().reset_index()
        df_monthly.rename(columns={'All India': 'demand'}, inplace=True)
        return df_monthly[['date', 'demand']]
        
    # 3. State: Specific State, Source: All Sources / All
    elif source in ['All Sources', 'All']:
        df_st = df2[df2['States'] == state]
        if df_st.empty:
            return pd.DataFrame(columns=['date', 'demand'])
        df_monthly = df_st.resample('ME', on='date')['Energy Met (MU)'].sum().reset_index()
        df_monthly.rename(columns={'Energy Met (MU)': 'demand'}, inplace=True)
        return df_monthly[['date', 'demand']]
        
    # 4. State: Specific State, Source: Specific Source
    else:
        st_rows = df2[df2['States'] == state]
        if st_rows.empty:
            return pd.DataFrame(columns=['date', 'demand'])
            
        region = st_rows['Region'].iloc[0]
        if pd.isna(region) or region not in regions_list:
            region = 'NR' # Default fallback
            
        # Get regional source generation
        df_src = df1_source[df1_source['source'] == source]
        if df_src.empty:
            matching = [s for s in sources_list if source.lower() in s.lower()]
            if matching:
                df_src = df1_source[df1_source['source'] == matching[0]]
                
        if df_src.empty or region not in df_src.columns:
            # Try All India as a fallback
            target_col = 'All India' if 'All India' in df_src.columns else df_src.columns[2]
        else:
            target_col = region
            
        df_src_monthly = df_src.resample('ME', on='date')[target_col].sum().reset_index()
        df_src_monthly.rename(columns={target_col: 'source_gen'}, inplace=True)
        
        # Get state's monthly demand
        df_st_monthly = df2[df2['States'] == state].resample('ME', on='date')['Energy Met (MU)'].sum().reset_index()
        df_st_monthly.rename(columns={'Energy Met (MU)': 'state_demand'}, inplace=True)
        
        # Get region's monthly demand
        df_reg_monthly = df2[df2['Region'] == region].resample('ME', on='date')['Energy Met (MU)'].sum().reset_index()
        df_reg_monthly.rename(columns={'Energy Met (MU)': 'region_demand'}, inplace=True)
        
        # Merge datasets by date
        merged = pd.merge(df_st_monthly, df_reg_monthly, on='date', how='outer').fillna(0)
        merged = pd.merge(merged, df_src_monthly, on='date', how='outer').fillna(0)
        
        # Compute dynamic month-by-month share
        merged['share'] = np.where(
            merged['region_demand'] > 0,
            merged['state_demand'] / merged['region_demand'],
            0.0
        )
        # Handle outliers in share
        merged['share'] = merged['share'].clip(0, 1.0)
        
        merged['demand'] = merged['source_gen'] * merged['share']
        
        # Clean up columns and sort
        merged = merged.sort_values('date').reset_index(drop=True)
        return merged[['date', 'demand']]
