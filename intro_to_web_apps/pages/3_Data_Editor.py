import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent.parent
data = pd.read_csv(BASE / "data" / "car_info.csv")

# ── Color util ────────────────────────────────────────────────────────────────
def build_colors(data, cmap='gist_heat'):
    cmap = plt.get_cmap(cmap)
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    step = max(1, int(len(colors) / len(data)))
    return {cls: colors[i * step] for i, cls in enumerate(data)}

DEFAULT_COLOR_MAP = 'gist_heat'

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding: 0 2rem 2rem 2rem !important; }
    .page-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(36px, 5vw, 64px); letter-spacing: 3px; color: #F0EDE8; margin: 32px 0 4px 0; line-height: 1; }
    .page-title .red { color: #E8302A; }
    .section-label { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #E8302A; margin-bottom: 8px; }
    .red-divider { width: 40px; height: 2px; background: #E8302A; margin: 8px 0 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">DATA <span class="red">EDITOR</span></div>', unsafe_allow_html=True)
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">// Filters</div>', unsafe_allow_html=True)

    years = data['year'].sort_values().unique()
    start_year, end_year = st.select_slider(
        "Year Range",
        options=years,
        value=(years[0], years[-1])
    )

    brands = st.multiselect("Brands", sorted(data['make_name'].unique()))
    decades = st.multiselect("Decades", sorted(data['decade'].unique()))

    st.markdown('<div class="section-label" style="margin-top:16px;">// Chart Options</div>', unsafe_allow_html=True)
    color_map = st.selectbox("Color Theme", ['gist_heat', 'plasma', 'viridis', 'cool', 'inferno'], index=0)
    chart_height = st.slider("Chart Height", min_value=300, max_value=800, value=450, step=50)

# ── Filter data ───────────────────────────────────────────────────────────────
filtered = data[(data['year'] >= start_year) & (data['year'] <= end_year)]
if brands:
    filtered = filtered[filtered['make_name'].isin(brands)]
if decades:
    filtered = filtered[filtered['decade'].isin(decades)]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(['Production Trends', 'Brand Analysis', 'Raw Data'])

# ── Tab 1: Production Trends ──────────────────────────────────────────────────
with tab1:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('<div class="section-label">// Models Produced Per Year</div>', unsafe_allow_html=True)
        temp = filtered.groupby('year')['unique_models'].sum().reset_index()
        fig = px.line(temp, x='year', y='unique_models',
                      title='Total Unique Models Per Year',
                      labels={'year': 'Year', 'unique_models': 'Unique Models'})
        fig.update_traces(line=dict(color='#E8302A'), marker=dict(color='#E8302A'))
        fig.update_layout(
            paper_bgcolor='#0D0D0D', plot_bgcolor='#141414',
            font_color='#888', height=chart_height,
            title_font_color='#F0EDE8'
        )
        fig.update_xaxes(gridcolor='#1e1e1e')
        fig.update_yaxes(gridcolor='#1e1e1e')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">// Year Summary</div>', unsafe_allow_html=True)
        summary = filtered.groupby('year')['unique_models'].sum().reset_index()
        summary.columns = ['Year', 'Models']
        st.dataframe(summary, use_container_width=True, height=chart_height)

# ── Tab 2: Brand Analysis ─────────────────────────────────────────────────────
with tab2:
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-label">// Production by Decade</div>', unsafe_allow_html=True)

        chart_type = st.radio(
            "Chart Type",
            ["Bar", "Scatter", "Line"],
            horizontal=True
        )

        temp = filtered.groupby(['decade', 'make_name'])['total_entries'].sum().reset_index()
        color_map_dict = build_colors(temp['make_name'].unique(), color_map)

        if chart_type == "Bar":
            fig = px.bar(temp, x='decade', y='total_entries', color='make_name',
                         title='Total Entries by Brand per Decade',
                         labels={'decade': 'Decade', 'total_entries': 'Total Entries', 'make_name': 'Brand'},
                         color_discrete_map=color_map_dict)
        elif chart_type == "Scatter":
            fig = px.scatter(temp, x='decade', y='total_entries', color='make_name',
                             title='Total Entries by Brand per Decade',
                             labels={'decade': 'Decade', 'total_entries': 'Total Entries', 'make_name': 'Brand'},
                             color_discrete_map=color_map_dict)
        else:
            fig = px.line(temp, x='decade', y='total_entries', color='make_name',
                          title='Total Entries by Brand per Decade',
                          labels={'decade': 'Decade', 'total_entries': 'Total Entries', 'make_name': 'Brand'},
                          color_discrete_map=color_map_dict,
                          markers=True)

        fig.update_layout(
            paper_bgcolor='#0D0D0D', plot_bgcolor='#141414',
            font_color='#888', height=chart_height,
            title_font_color='#F0EDE8',
            legend=dict(bgcolor='#141414', bordercolor='#1e1e1e')
        )
        fig.update_xaxes(gridcolor='#1e1e1e')
        fig.update_yaxes(gridcolor='#1e1e1e')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">// Brand Totals</div>', unsafe_allow_html=True)
        brand_summary = filtered.groupby('make_name')['total_entries'].sum().reset_index()
        brand_summary.columns = ['Brand', 'Total']
        brand_summary.sort_values('Total', ascending=False, inplace=True)
        st.dataframe(brand_summary, use_container_width=True, height=chart_height)

# ── Tab 3: Raw Data ───────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-label">// Raw Data</div>', unsafe_allow_html=True)
    st.dataframe(filtered, use_container_width=True)