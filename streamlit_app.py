"""
Streamlit Dashboard for HRP Allocator.
Displays HRP portfolio weights and cluster dendrograms.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.figure_factory as ff
from huggingface_hub import HfApi, hf_hub_download
import json
import numpy as np
import config

st.set_page_config(
    page_title="P2Quant HRP Allocator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_latest_weights():
    """Fetch the most recent result file from HF dataset."""
    try:
        api = HfApi(token=config.HF_TOKEN)
        files = api.list_repo_files(repo_id=config.HF_OUTPUT_REPO, repo_type="dataset")
        json_files = sorted([f for f in files if f.endswith('.json')], reverse=True)
        if not json_files:
            return None
        local_path = hf_hub_download(
            repo_id=config.HF_OUTPUT_REPO,
            filename=json_files[0],
            repo_type="dataset",
            token=config.HF_TOKEN,
            cache_dir="./hf_cache"
        )
        with open(local_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return None

def create_dendrogram(linkage: list, labels: list):
    """Create a dendrogram from linkage matrix. Returns (fig, error_message)."""
    if linkage is None or len(linkage) == 0:
        return None, "Linkage matrix is empty."
    if labels is None or len(labels) == 0:
        return None, "Labels list is empty."

    expected_rows = len(labels) - 1
    if len(linkage) != expected_rows:
        msg = f"Linkage matrix has {len(linkage)} rows, expected {expected_rows}."
        if len(linkage) > expected_rows:
            linkage = linkage[:expected_rows]
            msg += " Trimmed to match."
        else:
            return None, msg + " Cannot fix."

    try:
        fig = ff.create_dendrogram(
            np.array(linkage),
            labels=labels,
            orientation='bottom',
            colorscale='Viridis'
        )
        fig.update_layout(
            title="Hierarchical Clustering Dendrogram",
            xaxis_title="ETF Ticker",
            yaxis_title="Distance",
            height=400
        )
        return fig, None
    except Exception as e:
        return None, str(e)

# --- Sidebar ---
st.sidebar.markdown("## ⚙️ Configuration")
st.sidebar.markdown(f"**Data Source:** `{config.HF_DATA_REPO}`")
st.sidebar.markdown(f"**Results Repo:** `{config.HF_OUTPUT_REPO}`")
st.sidebar.divider()

st.sidebar.markdown("### 📊 HRP Parameters")
st.sidebar.markdown(f"- Lookback Window: **{config.LOOKBACK_WINDOW} days**")
st.sidebar.markdown(f"- Linkage Method: **{config.LINKAGE_METHOD}**")
st.sidebar.divider()

data = load_latest_weights()
if data:
    st.sidebar.markdown(f"**Run Date:** {data.get('run_date', 'Unknown')}")
else:
    st.sidebar.markdown("*No data available*")

st.sidebar.divider()
st.sidebar.markdown("### 📖 About")
st.sidebar.markdown("""
**HRP Allocator** generates robust portfolio weights using Hierarchical Risk Parity.

- No expected return estimates required
- Uses covariance clustering to group similar assets
- Distributes risk equally across the hierarchy
""")

# --- Main Content ---
st.markdown('<div class="main-header">📊 P2Quant HRP Allocator</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hierarchical Risk Parity – Daily Rebalanced Weights</div>', unsafe_allow_html=True)

if data is None:
    st.warning("No data available. Please run the daily pipeline first.")
    st.stop()

# --- Debug expander (can be removed later) ---
with st.expander("🔍 Debug: Data Structure", expanded=False):
    st.write("Data keys:", list(data.keys()))
    st.write("cluster_info keys:", list(data.get('cluster_info', {}).keys()))
    for key in data.get('cluster_info', {}):
        st.write(f"{key}: linkage rows = {len(data['cluster_info'][key].get('linkage', []))}, tickers count = {len(data['cluster_info'][key].get('original_tickers', []))}")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📊 Combined", "📈 Equity Sectors", "💰 FI/Commodities"])
universe_keys = ["COMBINED", "EQUITY_SECTORS", "FI_COMMODITIES"]

for tab, universe_key in zip([tab1, tab2, tab3], universe_keys):
    with tab:
        weights = data['weights'].get(universe_key, {})
        
        if not weights:
            st.info(f"No weights available for {universe_key} universe.")
            continue
        
        df = pd.DataFrame(weights.items(), columns=['Ticker', 'Weight'])
        df = df.sort_values('Weight', ascending=False)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Portfolio Allocation")
            fig_pie = go.Figure(go.Pie(
                labels=df['Ticker'],
                values=df['Weight'],
                hole=0.4,
                textinfo='label+percent',
                marker=dict(colors=['#1f77b4' if i == 0 else '#a0aec0' for i in range(len(df))])
            ))
            fig_pie.update_layout(height=450)
            st.plotly_chart(fig_pie, use_container_width=True, key=f"pie_{universe_key}")
        
        with col2:
            st.markdown("### Weight Distribution")
            fig_bar = go.Figure(go.Bar(
                x=df['Ticker'],
                y=df['Weight'],
                marker_color=['#1f77b4' if i == 0 else '#a0aec0' for i in range(len(df))],
                text=df['Weight'].apply(lambda x: f'{x:.2%}'),
                textposition='outside'
            ))
            fig_bar.update_layout(
                xaxis_title="ETF Ticker",
                yaxis_title="Weight",
                height=450
            )
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_{universe_key}")
        
        st.markdown("### Detailed Weights")
        df_display = df.copy()
        df_display['Weight'] = df_display['Weight'].apply(lambda x: f'{x:.2%}')
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        # Dendrogram with error reporting
        st.markdown("### Hierarchical Clustering")
        cluster_entry = data.get('cluster_info', {}).get(universe_key)
        if cluster_entry:
            linkage = cluster_entry.get('linkage')
            original_tickers = cluster_entry.get('original_tickers')
            if linkage and original_tickers and len(original_tickers) > 2:
                fig_dendro, error_msg = create_dendrogram(linkage, original_tickers)
                if fig_dendro:
                    st.plotly_chart(fig_dendro, use_container_width=True, key=f"dendro_{universe_key}")
                else:
                    st.warning(f"Dendrogram could not be created: {error_msg}")
            else:
                st.info(f"Missing linkage or tickers. Linkage: {linkage is not None}, Tickers: {original_tickers is not None}, Count: {len(original_tickers) if original_tickers else 0}")
        else:
            st.info("No cluster information available for this universe.")
