import sys
import uuid
from pathlib import Path
import importlib.util
import streamlit as st

# Load supabase directly from file path
_client_path = Path(__file__).parent.parent / "backend" / "supabase.py"
_spec = importlib.util.spec_from_file_location("supabase", _client_path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

create_post = _module.create_post
get_posts = _module.get_posts
like_post = _module.like_post
upload_image = _module.upload_image

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
    .post-card { background: #141414; border: 1px solid #1e1e1e; border-left: 3px solid #E8302A; padding: 20px; margin-bottom: 12px; }
    .post-tag { font-family: 'DM Mono', monospace; font-size: 9px; letter-spacing: 2px; text-transform: uppercase; background: #E8302A; color: white; padding: 2px 8px; display: inline-block; margin-bottom: 8px; }
    .post-title { font-size: 16px; font-weight: 500; margin-bottom: 6px; color: #F0EDE8; }
    .post-meta { font-family: 'DM Mono', monospace; font-size: 10px; color: #555; margin-bottom: 10px; }
    .post-body { font-size: 13px; color: #888; line-height: 1.7; }
    .stButton > button { background: #E8302A !important; color: white !important; border: none !important; border-radius: 0 !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important; letter-spacing: 2px !important; text-transform: uppercase !important; padding: 10px 24px !important; }
    .stButton > button:hover { background: #c4241e !important; }
    .stTextInput > div > input, .stTextArea > div > textarea, .stSelectbox > div { background: #141414 !important; border: 1px solid #1e1e1e !important; color: #F0EDE8 !important; }
</style>
""", unsafe_allow_html=True)

# ── Page title ────────────────────────────────────────────────────────────────
st.markdown('<div class="page-title">BUILD <span class="red">FEED</span></div>', unsafe_allow_html=True)
st.markdown('<div class="red-divider"></div>', unsafe_allow_html=True)

# ── Two column layout ─────────────────────────────────────────────────────────
post_col, feed_col = st.columns([1, 2])

with post_col:
    st.markdown('<div class="section-label">// Submit a Build Post</div>', unsafe_allow_html=True)

    with st.form("post_form"):
        username = st.text_input("Username")
        title = st.text_input("Post Title")
        col1, col2 = st.columns(2)
        with col1:
            make = st.text_input("Make")
            year = st.number_input("Year", min_value=1900, max_value=2025, step=1, value=2000)
        with col2:
            model = st.text_input("Model")
            tag = st.selectbox("Category", ["Track", "Turbo", "Drag", "Stance", "JDM", "Featured", "General"])
        content = st.text_area("Your Post", height=200, placeholder="Tell us about your build...")
        uploaded_file = st.file_uploader("Upload Build Photo", type=["jpg", "jpeg", "png", "webp"])
        submitted = st.form_submit_button("Submit Post")

        if submitted:
            if not username or not title or not content:
                st.error("Username, title, and content are required.", icon="🚨")
            else:
                try:
                    image_url = None
                    if uploaded_file:
                        filename = f"{uuid.uuid4()}.{uploaded_file.name.split('.')[-1]}"
                        image_url = upload_image(uploaded_file, filename)
                    create_post(username, title, content, make, model, int(year), tag, image_url)
                    st.success("Post submitted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to submit: {e}", icon="🚨")

with feed_col:
    st.markdown('<div class="section-label">// Community Posts</div>', unsafe_allow_html=True)

    tag_filter = st.radio(
        "Filter:",
        ["All", "Track", "Turbo", "Drag", "Stance", "JDM", "Featured", "General"],
        horizontal=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    try:
        posts = get_posts(tag=tag_filter if tag_filter != "All" else None)

        if not posts:
            st.markdown('<p style="font-family: DM Mono, monospace; font-size: 11px; color: #444;">No posts yet. Be the first to post!</p>', unsafe_allow_html=True)
        else:
            for post in posts:
                if post.get('image_url'):
                    st.markdown(f"<img src='{post['image_url']}' style='width:100%;height:200px;object-fit:cover;margin-bottom:8px;filter:brightness(0.8);'>", unsafe_allow_html=True)

                car_info = " ".join(filter(None, [
                    str(post.get('year', '')) if post.get('year') else '',
                    post.get('make', ''),
                    post.get('model', '')
                ])).strip()

                st.markdown(f"""
                <div class="post-card">
                    <span class="post-tag">{post.get('tag') or 'General'}</span>
                    <div class="post-title">{post['title']}</div>
                    <div class="post-meta">@{post['username']} · {car_info} · {str(post['created_at'])[:10]}</div>
                    <div class="post-body">{post['content']}</div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"♥  {post['likes']}", key=f"like_{post['id']}"):
                    like_post(post['id'], post['likes'])
                    st.rerun()

                st.markdown("<br>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Could not load posts: {e}", icon="🚨")