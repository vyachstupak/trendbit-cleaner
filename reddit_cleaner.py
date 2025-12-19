import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

PLATFORM = "reddit"

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
    - ISO strings (common for Reddit exports)
    - unix seconds / ms if numeric
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

def norm_str(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(x).strip()

def join_tags(*vals: str) -> str:
    tags = []
    for v in vals:
        v = norm_str(v)
        if not v:
            continue
        # remove leading # if present
        v = v.lstrip("#")
        tags.append(v)

    # unique (case-insensitive) preserving order
    seen = set()
    out = []
    for t in tags:
        key = t.lower()
        if key not in seen:
            out.append(t)
            seen.add(key)
    return ", ".join(out)

def clean_reddit_df(df: pd.DataFrame, category: str, now_utc: Optional[datetime] = None) -> pd.DataFrame:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # core text fields
    title = df.get("title", pd.Series([None] * len(df)))
    body = df.get("body", pd.Series([None] * len(df)))
    description = df.get("description", pd.Series([None] * len(df)))

    # caption: prefer title
    caption = title.fillna("").astype(str)
    caption = caption.where(caption.str.strip() != "", df.get("displayName", pd.Series([""] * len(df))).fillna("").astype(str))
    caption = caption.where(caption.str.strip() != "", df.get("communityName", pd.Series([""] * len(df))).fillna("").astype(str))

    # text: title + body (fallback to description)
    t_title = title.fillna("").astype(str)
    t_body = body.fillna("").astype(str)
    text = (t_title.str.strip() + "\n\n" + t_body.str.strip()).str.strip()
    text = text.where(text != "", description.fillna("").astype(str))

    # hashtags: from flair + community
    flair = df.get("flair", pd.Series([None] * len(df)))
    community = df.get("communityName", pd.Series([None] * len(df)))
    hashtags = pd.DataFrame({"flair": flair, "community": community}).apply(
        lambda r: join_tags(r.get("flair"), r.get("community")),
        axis=1
    )

    # creator
    creator = df.get("username", pd.Series([None] * len(df))).fillna("").astype(str)
    creator_fullname = pd.Series([""] * len(df))  # not available
    creator_followers = pd.Series([np.nan] * len(df))  # not available

    # url
    url = df.get("url", pd.Series([None] * len(df)))
    if "link" in df.columns:
        url = url.where(url.notna() & (url.astype(str).str.strip() != ""), df["link"])
    url = url.fillna("").astype(str)

    # timestamp
    created = df.get("createdAt", pd.Series([None] * len(df)))
    ts = created.apply(parse_timestamp)
    if "scrapedAt" in df.columns:
        ts2 = df["scrapedAt"].apply(parse_timestamp)
        ts = ts.where(~ts.isna(), ts2)

    # metrics
    upvotes = df.get("upVotes", pd.Series([0] * len(df))).apply(to_int).fillna(0).astype(int)
    likes = upvotes.copy()
    comments = df.get("numberOfComments", pd.Series([0] * len(df))).apply(to_int).fillna(0).astype(int)

    shares = pd.Series([0] * len(df)).astype(int)
    plays = pd.Series([0] * len(df)).astype(int)
    saves = pd.Series([0] * len(df)).astype(int)
    audio_name = pd.Series([""] * len(df))

    # hours since post
    hours_since_post = (now_utc - ts).dt.total_seconds() / 3600
    hours_since_post = hours_since_post.clip(lower=0.01)

    # engagement_score (no plays for reddit; use denom=1)
    denom = plays.replace(0, 1)
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
        "platform": PLATFORM,
        "caption": caption.fillna(""),
        "text": text.fillna(""),
        "hashtags": hashtags.fillna(""),
        "creator": creator.fillna(""),
        "creator_fullname": creator_fullname.fillna(""),
        "creator_followers": creator_followers,
        "timestamp": ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url,
        "likes": likes,
        "comments": comments,
        "shares": shares,
        "plays": plays,
        "saves": saves,
        "upvotes": upvotes,
        "engagement_score": engagement_score.fillna(0),
        "hours_since_post": hours_since_post,
        "velocity": velocity.fillna(0),
        "audio_name": audio_name.fillna(""),
        "category": category,
    })

    final_cols = [
        "platform","caption","text","hashtags","creator","creator_fullname","creator_followers",
        "timestamp","url","likes","comments","shares","plays","saves","upvotes","engagement_score",
        "hours_since_post","velocity","audio_name","category"
    ]
    return clean[final_cols]

def clean_reddit_items(items: List[Dict[str, Any]], category: str):
    df = pd.DataFrame(items)
    if df.empty:
        return []
    clean_df = clean_reddit_df(df, category)
    clean_df = clean_df.replace({np.nan: None, pd.NaT: None})
    return clean_df.to_dict(orient="records")