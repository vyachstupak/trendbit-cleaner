import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

DEDUP_BY_URL = True

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
    """
    Handles:
    - ISO strings (createTimeISO)
    - unix seconds (createTime)
    - unix milliseconds (createTime)
    """
    if pd.isna(x):
        return pd.NaT

    # numeric epoch?
    try:
        xv = float(x)
        if xv > 1e12:
            return pd.to_datetime(int(xv), unit="ms", utc=True, errors="coerce")
        if xv > 1e9:
            return pd.to_datetime(int(xv), unit="s", utc=True, errors="coerce")
    except Exception:
        pass

    # string timestamp
    return pd.to_datetime(x, utc=True, errors="coerce")

def join_list_fields(row, cols, strip_hash=False):
    vals = []
    for c in cols:
        v = row.get(c, None)
        if pd.notna(v) and str(v).strip() != "":
            s = str(v).strip()
            if strip_hash:
                s = s.lstrip("#")
            vals.append(s)

    # unique while preserving order (case-insensitive)
    seen = set()
    out = []
    for v in vals:
        key = v.lower()
        if key not in seen:
            out.append(v)
            seen.add(key)
    return ", ".join(out)

def clean_tiktok_df(df: pd.DataFrame, category: str, now_utc: Optional[datetime] = None) -> pd.DataFrame:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    if df.empty:
        return pd.DataFrame(columns=[
            "platform","caption","text","hashtags","creator","creator_fullname","creator_followers",
            "timestamp","url","likes","comments","shares","plays","saves","upvotes","engagement_score",
            "hours_since_post","velocity","audio_name","category"
        ])

    # --- hashtags: hashtags/0/name ... ---
    hashtag_cols = [c for c in df.columns if c.startswith("hashtags/") and c.endswith("/name")]

    # --- core text ---
    text = df.get("text", pd.Series([""] * len(df))).fillna("")
    caption = text  # TikTok doesn't really separate "caption" vs "text" in your schema

    # --- author ---
    creator = df.get("authorMeta/name", pd.Series([""] * len(df))).fillna("")
    creator_fullname = df.get("authorMeta/nickName", pd.Series([""] * len(df))).fillna("")
    creator_followers = df.get("authorMeta/fans", pd.Series([np.nan] * len(df))).apply(to_int)

    # --- url ---
    url = df.get("webVideoUrl", pd.Series([""] * len(df))).fillna("")
    # (fallback if needed)
    if (url == "").all() and "id" in df.columns:
        url = df["id"].astype(str).fillna("")

    # --- timestamp ---
    # Prefer createTimeISO, fallback to createTime
    ts_source = df.get("createTimeISO", None)
    if ts_source is None:
        ts_source = df.get("createTime", pd.Series([None] * len(df)))
    ts = ts_source.apply(parse_timestamp)
    timestamp_str = ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")

    # --- engagement fields ---
    likes = df.get("diggCount", pd.Series([np.nan] * len(df))).apply(to_int).fillna(0)
    comments = df.get("commentCount", pd.Series([np.nan] * len(df))).apply(to_int).fillna(0)
    shares = df.get("shareCount", pd.Series([np.nan] * len(df))).apply(to_int).fillna(0)
    plays = df.get("playCount", pd.Series([np.nan] * len(df))).apply(to_int).fillna(0)
    saves = df.get("collectCount", pd.Series([np.nan] * len(df))).apply(to_int).fillna(0)

    # Not used for TikTok in your schema, keep 0 for standardization
    upvotes = pd.Series([0] * len(df))

    # --- hashtags combined ---
    hashtags = (
        df.apply(lambda r: join_list_fields(r, hashtag_cols, strip_hash=True), axis=1)
        if hashtag_cols else pd.Series([""] * len(df))
    )

    # --- audio name ---
    audio_name = df.get("musicMeta/musicName", pd.Series([""] * len(df))).fillna("")

    # --- recency metrics ---
    hours_since_post = (now_utc - ts).dt.total_seconds() / 3600
    hours_since_post = hours_since_post.fillna(0).clip(lower=0.01)

    denom = plays.replace(0, 1)
    engagement_score = (
        likes
        + 2 * comments
        + 3 * shares
        + 2 * saves
        + upvotes.fillna(0)
    ) / denom

    total_eng = likes + comments + shares + saves + upvotes.fillna(0)
    velocity = total_eng / hours_since_post

    clean = pd.DataFrame({
        "platform": "tiktok",
        "caption": caption,
        "text": text,
        "hashtags": hashtags.fillna(""),
        "creator": creator,
        "creator_fullname": creator_fullname,
        "creator_followers": creator_followers.fillna(0).astype(int),
        "timestamp": timestamp_str,
        "url": url,
        "likes": likes.astype(int),
        "comments": comments.astype(int),
        "shares": shares.astype(int),
        "plays": plays.astype(int),
        "saves": saves.astype(int),
        "upvotes": upvotes.astype(int),
        "engagement_score": engagement_score.fillna(0),
        "hours_since_post": hours_since_post,
        "velocity": velocity.fillna(0),
        "audio_name": audio_name,
        "category": category
    })

    if DEDUP_BY_URL:
        clean = clean.drop_duplicates(subset=["url"], keep="first")

    clean = clean.replace({np.nan: None, pd.NaT: None})
    final_cols = [
        "platform","caption","text","hashtags","creator","creator_fullname","creator_followers",
        "timestamp","url","likes","comments","shares","plays","saves","upvotes","engagement_score",
        "hours_since_post","velocity","audio_name","category"
    ]
    return clean[final_cols]

def clean_tiktok_items(items: List[Dict[str, Any]], category: str):
    df = pd.DataFrame(items)
    if df.empty:
        return []
    clean_df = clean_tiktok_df(df, category)
    clean_df = clean_df.replace({np.nan: None, pd.NaT: None})
    return clean_df.to_dict(orient="records")