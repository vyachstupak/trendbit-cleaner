from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

import pandas as pd
import numpy as np
from datetime import datetime, timezone

app = FastAPI()

@app.get("/")
def health():
    return {"ok": True}

# ---------- helpers (copied from your script) ----------
def to_int(x):
    try:
        if pd.isna(x):
            return np.nan
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return int(float(x))
    except Exception:
        return np.nan

def parse_timestamp(x):
    if pd.isna(x):
        return pd.NaT
    try:
        xv = float(x)
        if xv > 1e12:
            return pd.to_datetime(int(xv), unit="ms", utc=True, errors="coerce")
        if xv > 1e9:
            return pd.to_datetime(int(xv), unit="s", utc=True, errors="coerce")
    except Exception:
        pass
    return pd.to_datetime(x, utc=True, errors="coerce")

def join_hashtags(row, hashtag_cols):
    tags = []
    for c in hashtag_cols:
        v = row.get(c, None)
        if pd.notna(v) and str(v).strip() != "":
            tags.append(str(v).strip().lstrip("#"))
    seen = set()
    out = []
    for t in tags:
        if t.lower() not in seen:
            out.append(t)
            seen.add(t.lower())
    return ", ".join(out)

def clean_instagram_items(items: list[dict], category: str):
    NOW_UTC = datetime.now(timezone.utc)

    df = pd.DataFrame(items)

    hashtag_cols = [c for c in df.columns if c.startswith("hashtags/")]

    caption = df.get("caption", pd.Series([None]*len(df)))
    creator = df.get("ownerUsername", pd.Series([None]*len(df)))
    creator_fullname = df.get("ownerFullName", pd.Series([None]*len(df)))
    url = df.get(
        "url",
        df.get("displayUrl", df.get("videoUrl", pd.Series([None]*len(df))))
    )

    likes = df.get("likesCount", pd.Series([np.nan]*len(df))).apply(to_int)
    comments = df.get("commentsCount", pd.Series([np.nan]*len(df))).apply(to_int)

    plays_raw = None
    for col in ["videoPlayCount", "igPlayCount", "fbPlayCount"]:
        if col in df.columns:
            plays_raw = df[col]
            break
    plays = plays_raw.apply(to_int) if plays_raw is not None else pd.Series([0]*len(df))

    audio_name = df.get("musicInfo/song_name", pd.Series([None]*len(df)))
    ts = df.get("timestamp", pd.Series([None]*len(df))).apply(parse_timestamp)

    hashtags = df.apply(lambda r: join_hashtags(r, hashtag_cols), axis=1) if hashtag_cols else pd.Series([""]*len(df))

    shares = pd.Series([0]*len(df))
    saves = pd.Series([0]*len(df))
    upvotes = pd.Series([0]*len(df))
    creator_followers = pd.Series([np.nan]*len(df))

    hours_since_post = (NOW_UTC - ts).dt.total_seconds() / 3600
    hours_since_post = hours_since_post.clip(lower=0.01)

    denom = plays.fillna(0).replace(0, 1)
    engagement_score = (
        likes.fillna(0)
        + 2 * comments.fillna(0)
        + 3 * shares.fillna(0)
        + 2 * saves.fillna(0)
        + upvotes.fillna(0)
    ) / denom

    total_eng = (
        likes.fillna(0)
        + comments.fillna(0)
        + shares.fillna(0)
        + saves.fillna(0)
        + upvotes.fillna(0)
    )
    velocity = total_eng / hours_since_post

    clean = pd.DataFrame({
        "platform": "instagram",
        "caption": caption.fillna(""),
        "text": caption.fillna(""),
        "hashtags": hashtags,
        "creator": creator.fillna(""),
        "creator_fullname": creator_fullname.fillna(""),
        "creator_followers": creator_followers,
        "timestamp": ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url.fillna(""),
        "likes": likes.fillna(0).astype(int),
        "comments": comments.fillna(0).astype(int),
        "shares": shares.astype(int),
        "plays": plays.fillna(0).astype(int),
        "saves": saves.astype(int),
        "upvotes": upvotes.astype(int),
        "engagement_score": engagement_score.fillna(0),
        "hours_since_post": hours_since_post.fillna(np.nan),
        "velocity": velocity.fillna(0),
        "audio_name": audio_name.fillna(""),
        "category": category
    })

    clean = clean.drop_duplicates(subset=["url"], keep="first")

    final_cols = [
        "platform","caption","text","hashtags","creator","creator_fullname","creator_followers",
        "timestamp","url","likes","comments","shares","plays","saves","upvotes","engagement_score",
        "hours_since_post","velocity","audio_name","category"
    ]
    clean = clean[final_cols]
    return clean.to_dict(orient="records")


# ---------- API ----------
class CleanRequest(BaseModel):
    items: list[dict]
    category: str = "beauty"

@app.post("/clean/instagram")
def clean_instagram(req: CleanRequest):
    rows = clean_instagram_items(req.items, req.category)
    return {"rows": rows, "count": len(rows)}
