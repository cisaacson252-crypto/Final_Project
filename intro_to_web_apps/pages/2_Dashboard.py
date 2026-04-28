import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

def build_colors(data, cmap='Reds'):
    cmap = plt.get_cmap(cmap)
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]
    step = int(len(colors) / len(data))
    return {cls: colors[i * step] for i, cls in enumerate(data)}

DEFAULT_COLOR_MAP = 'Reds'

st.set_page_config(page_title='CAR DATA', layout='wide')

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding: 0 2rem 2rem 2rem !important; }
    .page-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(40px, 6vw, 72px); letter-spacing: 3px; color: #F0EDE8; margin: 32px 0 4px 0; line-height: 1; }
    .page-title .red { color: #E8302A; }
    .section-label { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #E8302A; margin-bottom: 8px; }
    .red-divider { width: 40px; height: 2px; background: #E8302A; margin: 8px 0 24px 0; }
    .stat-box { background: #141414; border: 1px solid #1e1e1e; padding: 20px; text-align: center; margin-bottom: 8px; }
    .stat-box-val { font-family: 'Bebas Neue', sans-serif; font-size: 38px; color: #E8302A; line-height: 1; margin-bottom: 4px; }
    .stat-box-label { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #555; }
</style>
""", unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
data = pd.read_csv("C:\\Users\\cisaa\\OneDrive - Marquette University\\s26\\AIM 4420\\car_info.csv")
data_orig = data.copy()

# ── Page title ───────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">DRIVING <span class="red">DATA</span></div>', unsafe_allow_html=True)
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="section-label">// Filters</div>', unsafe_allow_html=True)

    brand = st.multiselect("Brand", sorted(data['make_name'].unique()))

    decade = st.multiselect("Decade", sorted(data['decade'].unique()))

    year_min, year_max = int(data['year'].min()), int(data['year'].max())
    year_range = st.slider("Year Range", year_min, year_max, (year_min, year_max))

# ── Apply filters ─────────────────────────────────────────────────────────────
if brand:
    data = data[data['make_name'].isin(brand)]
if decade:
    data = data[data['decade'].isin(decade)]
data = data[(data['year'] >= year_range[0]) & (data['year'] <= year_range[1])]

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(['Dashboard', 'Raw Data'])

with tab1:

    # ── Metrics ──────────────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="stat-box"><div class="stat-box-val">{data["total_entries"].sum():,}</div><div class="stat-box-label">Total Entries</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="stat-box"><div class="stat-box-val">{data["make_name"].nunique():,}</div><div class="stat-box-label">Brands</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="stat-box"><div class="stat-box-val">{data["model_name"].nunique():,}</div><div class="stat-box-label">Unique Models</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="stat-box"><div class="stat-box-val">{data["year"].nunique():,}</div><div class="stat-box-label">Years Covered</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: two charts side by side ───────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-label">// Total Entries by Model</div>', unsafe_allow_html=True)
        temp = data.groupby(['model_name', 'model_id'])['total_entries'].sum().reset_index()
        temp.sort_values('total_entries', ascending=False, inplace=True)
        fig = px.bar(temp, x='model_name', y='total_entries',
                     color='model_name',
                     color_discrete_map=build_colors(temp['model_name'].unique(), DEFAULT_COLOR_MAP),
                     labels={'model_name': 'Model', 'total_entries': 'Total Entries'})
        fig.update_layout(paper_bgcolor='#0D0D0D', plot_bgcolor='#141414', font_color='#888',
                          showlegend=False, margin=dict(t=20, b=0))
        fig.update_xaxes(showticklabels=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-label">// # of Entries per Brand</div>', unsafe_allow_html=True)
        temp = data.groupby('make_name')['total_entries'].sum().reset_index()
        temp.sort_values('total_entries', ascending=False, inplace=True)
        fig = px.bar(temp, x='make_name', y='total_entries',
                     color='make_name',
                     color_discrete_map=build_colors(temp['make_name'].unique(), DEFAULT_COLOR_MAP),
                     labels={'make_name': 'Brand', 'total_entries': 'Total Entries'})
        fig.update_layout(paper_bgcolor='#0D0D0D', plot_bgcolor='#141414', font_color='#888',
                          showlegend=False, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: two charts side by side ───────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-label">// Production of Models per Decade</div>', unsafe_allow_html=True)
        temp = data.groupby('decade')['unique_models'].sum().reset_index()
        temp.sort_values('decade', inplace=True)
        fig = px.line(temp, x='decade', y='unique_models',
                      markers=True,
                      labels={'decade': 'Decade', 'unique_models': 'Unique Models'})
        fig.update_traces(line_color='#E8302A', marker=dict(color='#E8302A', size=7))
        fig.update_layout(paper_bgcolor='#0D0D0D', plot_bgcolor='#141414', font_color='#888',
                          margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.markdown('<div class="section-label">// Brand & Make ID</div>', unsafe_allow_html=True)
        temp = data[['make_name', 'make_id']].drop_duplicates().sort_values('make_id')
        fig = px.bar(temp, x='make_name', y='make_id',
                     color='make_name',
                     color_discrete_map=build_colors(temp['make_name'].unique(), DEFAULT_COLOR_MAP),
                     labels={'make_name': 'Brand', 'make_id': 'Make ID'})
        fig.update_layout(paper_bgcolor='#0D0D0D', plot_bgcolor='#141414', font_color='#888',
                          showlegend=False, margin=dict(t=20, b=0))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.dataframe(data)
