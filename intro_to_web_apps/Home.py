import streamlit as st
import pandas as pd

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="TRAYCED", layout="wide", page_icon="🏎️")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .block-container { padding: 0 2rem 2rem 2rem !important; }

    .hero-wrapper {
        border-bottom: 1px solid #1e1e1e;
        padding: 64px 0 48px 0;
        position: relative;
        overflow: hidden;
    }
    .hero-wrapper::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 50% 100% at 80% 50%, rgba(232,48,42,0.10) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-eyebrow { font-family: 'DM Mono', monospace; font-size: 11px; letter-spacing: 3px; text-transform: uppercase; color: #E8302A; margin-bottom: 14px; }
    .hero-title { font-family: 'Bebas Neue', sans-serif; font-size: clamp(60px, 8vw, 100px); line-height: 0.92; letter-spacing: 2px; margin-bottom: 20px; }
    .hero-title .red { color: #E8302A; }
    .hero-sub { font-size: 15px; line-height: 1.75; color: #888; max-width: 420px; margin-bottom: 32px; }

    .stButton > button { background: #E8302A !important; color: white !important; border: none !important; border-radius: 0 !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 10px 24px !important; }
    .stButton > button:hover { background: #c4241e !important; }

    .red-divider { width: 40px; height: 2px; background: #E8302A; margin: 12px 0 20px 0; }
    .section-label { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: #E8302A; margin-bottom: 8px; }

    .feed-card { background: #141414; border: 1px solid #1e1e1e; overflow: hidden; transition: border-color 0.2s; margin-bottom: 8px; }
    .feed-card:hover { border-color: #333; }
    .feed-card img { width: 100%; height: 180px; object-fit: cover; filter: brightness(0.8); }
    .feed-card-body { padding: 14px; }
    .feed-tag { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; background: #E8302A; color: white; padding: 2px 8px; display: inline-block; margin-bottom: 8px; }
    .feed-card-title { font-size: 13px; font-weight: 500; line-height: 1.4; margin-bottom: 6px; }

    .stat-box { background: #141414; border: 1px solid #1e1e1e; padding: 24px 20px; text-align: center; }
    .stat-box-val { font-family: 'Bebas Neue', sans-serif; font-size: 42px; color: #E8302A; line-height: 1; margin-bottom: 6px; }
    .stat-box-label { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 2px; text-transform: uppercase; color: #555; }

    .ticker { background: #E8302A; padding: 10px 0; overflow: hidden; white-space: nowrap; margin: 0 -2rem; }
    .ticker-inner { display: inline-block; animation: ticker 24s linear infinite; }
    .ticker-inner span { font-family: 'DM Mono', monospace; font-size: 10px; letter-spacing: 3px; text-transform: uppercase; color: rgba(255,255,255,0.85); margin-right: 40px; }
    @keyframes ticker { from { transform: translateX(0); } to { transform: translateX(-50%); } }

    .feature-card { background: #141414; border: 1px solid #1e1e1e; padding: 24px 20px; height: 100%; transition: background 0.2s; }
    .feature-card:hover { background: #1c1c1c; }
    .feature-icon { font-size: 22px; margin-bottom: 12px; }
    .feature-title { font-family: 'Bebas Neue', sans-serif; font-size: 20px; letter-spacing: 1px; margin-bottom: 8px; }
    .feature-desc { font-size: 12px; color: #666; line-height: 1.7; }

    .footer { border-top: 1px solid #1a1a1a; padding: 32px 0 16px 0; display: flex; justify-content: space-between; align-items: center; margin-top: 64px; }
    .footer-logo { font-family: 'Bebas Neue', sans-serif; font-size: 22px; letter-spacing: 4px; color: #E8302A; }
    .footer-logo span { color: #F0EDE8; }
    .footer-text { font-family: 'DM Mono', monospace; font-size: 10px; color: #333; letter-spacing: 2px; }
</style>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
  <div class="hero-eyebrow">// Car Social + Data Platform</div>
  <div class="hero-title">YOUR BUILD.<br><span class="red">YOUR WAY.</span></div>
  <p class="hero-sub">Share your build, track your mods, and dig into real data from thousands of enthusiasts — all in one place.</p>
</div>
""", unsafe_allow_html=True)

hero_col1, hero_col2 = st.columns([1, 1])
with hero_col1:
    if st.button("Join the Community", key="cta_join"):
        st.success("Welcome to TRAYCED.")
with hero_col2:
    explore = st.toggle("Show Community Stats", value=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TICKER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="ticker">
  <div class="ticker-inner">
    <span>BMW E30</span><span>Nissan R34</span><span>Toyota Supra MK4</span><span>Honda NSX</span>
    <span>Porsche 911</span><span>Mazda RX-7 FD</span><span>Subaru STI</span><span>Mitsubishi Evo IX</span>
    <span>BMW E30</span><span>Nissan R34</span><span>Toyota Supra MK4</span><span>Honda NSX</span>
    <span>Porsche 911</span><span>Mazda RX-7 FD</span><span>Subaru STI</span><span>Mitsubishi Evo IX</span>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── COMMUNITY STATS (conditional on toggle) ───────────────────────────────────
if explore:
    st.markdown('<div class="section-label">// Platform Stats</div>', unsafe_allow_html=True)
    st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

    stat1, stat2, stat3, stat4 = st.columns(4)
    with stat1:
        st.markdown('<div class="stat-box"><div class="stat-box-val">14,302</div><div class="stat-box-label">Active Builds</div></div>', unsafe_allow_html=True)
    with stat2:
        st.markdown('<div class="stat-box"><div class="stat-box-val">43,221</div><div class="stat-box-label">Mod Records</div></div>', unsafe_allow_html=True)
    with stat3:
        st.markdown('<div class="stat-box"><div class="stat-box-val">15,800+</div><div class="stat-box-label">Forum Posts</div></div>', unsafe_allow_html=True)
    with stat4:
        st.markdown('<div class="stat-box"><div class="stat-box-val">6,044</div><div class="stat-box-label">Track Days</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── LATEST BUILDS FEED ───────────────────────────────────────────────────────
st.markdown('<div class="section-label">// Community Feed</div>', unsafe_allow_html=True)
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

FEED_POSTS = [
    {"tag": "Featured", "title": "1991 BMW E30 M3 — Full Track Build",
     "img": "https://cdn.dealeraccelerate.com/farland/1/171/6263/1920x1440/1991-bmw-m3"},
    {"tag": "Turbo", "title": "FD RX-7 Single Turbo Swap",
     "img": "https://images.fineartamerica.com/images/artworkimages/mediumlarge/3/5-legendary-air-cooled-flat-6-engine-porsche-911-964-turbo-36-s-coupe-flachbau-vladyslav-shapovalenko.jpg"},
    {"tag": "Stance", "title": "Porsche 964 Air-Cooled Build",
     "img": "https://www.gorillafivemcars.com/cdn/shop/files/GTA5_2024-03-07_02-47-05.png?v=1714952586&width=1206"},
    {"tag": "Drag", "title": "Supra A90 Drag Build — 9s Pass",
     "img": "https://i.ebayimg.com/images/g/hDAAAOSwJKVlHgPr/s-l1200.jpg"},
    {"tag": "JDM", "title": "EK9 Civic Type R — Full Strip",
     "img": "https://i.ebayimg.com/images/g/hDAAAOSwJKVlHgPr/s-l1200.jpg"},
    {"tag": "Track", "title": "A Widebody Subaru WRX Hatchback",
     "img": "https://stanceauto.co.uk/uploads/images/202409/image_870x_66d43f0e4a5d4.webp"},
]

category_filter = st.radio(
    "Filter by Category:",
    ["All", "Featured", "Track", "Turbo", "Drag", "Stance", "JDM"],
    index=0,
    horizontal=True
)

st.markdown("<br>", unsafe_allow_html=True)

if category_filter == "All":
    filtered_posts = FEED_POSTS
else:
    filtered_posts = [p for p in FEED_POSTS if p["tag"] == category_filter]

feed_col1, feed_col2, feed_col3 = st.columns(3)
cols = [feed_col1, feed_col2, feed_col3]

for i, post in enumerate(filtered_posts):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="feed-card">
            <img src="{post['img']}" alt="build photo">
            <div class="feed-card-body">
                <span class="feed-tag">{post['tag']}</span>
                <div class="feed-card-title">{post['title']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── FEATURES ─────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">// Platform Features</div>', unsafe_allow_html=True)
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

feat1, feat2, feat3 = st.columns(3)
features = [
    ("🚗", "My Garage", "Log every mod, part number, and dollar spent. Your full build history in one place."),
    ("📈", "Data Explorer", "Run queries across community data. Compare builds, benchmark HP, find trends by make and year."),
    ("🤖", "AI Assistant", "Ask questions about your build or the data. Powered by RAG over real forum threads and build logs."),
]
for col, (icon, title, desc) in zip([feat1, feat2, feat3], features):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

feat4, feat5, feat6 = st.columns(3)
features2 = [
    ("🏆", "Leaderboards", "See who's running the quickest 1/4 mile, highest HP, or most mods. Weekly and all-time rankings."),
    ("📷", "Build Feed", "Photo-forward posts, tagged by make, mod type, and category. Follow what you care about."),
    ("🔧", "Parts Tracker", "Track installs, budget, and condition over time. Know what you've done and what's next."),
]
for col, (icon, title, desc) in zip([feat4, feat5, feat6], features2):
    with col:
        st.markdown(f"""
        <div class="feature-card">
            <div class="feature-icon">{icon}</div>
            <div class="feature-title">{title}</div>
            <div class="feature-desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-logo">TRAY<span>CED</span></div>
    <div class="footer-text">© 2026 TRAYCED — CAR SOCIAL PLATFORM</div>
    <div class="footer-text">BUILT WITH STREAMLIT</div>
</div>
""", unsafe_allow_html=True)