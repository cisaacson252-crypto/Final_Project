import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

client = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_posts(tag=None, limit=20):
    query = client.table("posts").select("*").order("created_at", desc=True).limit(limit)
    if tag and tag != "All":
        query = query.eq("tag", tag)
    return query.execute().data

def create_post(username, title, content, make=None, model=None, year=None, tag=None, image_url=None):
    return client.table("posts").insert({
        "username": username,
        "title": title,
        "content": content,
        "make": make,
        "model": model,
        "year": year,
        "tag": tag,
        "image_url": image_url
    }).execute()

def like_post(post_id, current_likes):
    return client.table("posts").update({"likes": current_likes + 1}).eq("id", post_id).execute()

def upload_image(file, filename):
    res = client.storage.from_("post-images").upload(
        path=filename,
        file=file.read(),
        file_options={"content-type": file.type}
    )
    # Get public URL
    url = client.storage.from_("post-images").get_public_url(filename)
    return url