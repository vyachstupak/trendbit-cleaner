import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ---------- CONFIG ----------
INPUT_CSV = "instagram_raw.csv"        # your Apify/Make export
OUTPUT_CSV = "instagram_clean.csv"
PLATFORM = "instagram"
CATEGORY = "beauty"                    # set per vertical (beauty, fitness, etc.)
NOW_UTC = datetime.now(timezone.utc)   # used to compute hours_since_post/velocity

# Optional: if you want to dedup by URL within this file
DEDUP_BY_URL = True


def to_int(x):
    try:
        if pd.isna(x):
            return np.nan
        # handle strings like "1,234"
        if isinstance(x, str):
            x = x.replace(",", "").strip()
        return int(float(x))
    except Exception:
        return np.nan


def parse_timestamp(x):
    """
    Handles:
    - ISO strings
    - unix seconds
    - unix milliseconds
    """
    if pd.isna(x):
        return pd.NaT

    # numeric epoch?
    try:
        xv = float(x)
        # heuristic: ms if > 1e12
        if xv > 1e12:
            return pd.to_datetime(int(xv), unit="ms", utc=True, errors="coerce")
        if xv > 1e9:
            return pd.to_datetime(int(xv), unit="s", utc=True, errors="coerce")
    except Exception:
        pass

    # string timestamp
    return pd.to_datetime(x, utc=True, errors="coerce")


def join_hashtags(row, hashtag_cols):
    tags = []
    for c in hashtag_cols:
        v = row.get(c, None)
        if pd.notna(v) and str(v).strip() != "":
            tags.append(str(v).strip().lstrip("#"))
    # unique while preserving order
    seen = set()
    out = []
    for t in tags:
        if t.lower() not in seen:
            out.append(t)
            seen.add(t.lower())
    return ", ".join(out)


def clean_instagram_df(df: pd.DataFrame, category: str, now_utc: datetime | None = None) -> pd.DataFrame:
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    # find hashtag columns like hashtags/0 ... hashtags/29 (or more)
    hashtag_cols = [c for c in df.columns if c.startswith("hashtags/")]

    # core fields
    caption = df.get("caption", pd.Series([None]*len(df)))
    creator = df.get("ownerUsername", pd.Series([None]*len(df)))
    creator_fullname = df.get("ownerFullName", pd.Series([None]*len(df)))

    # best URL to use (prefer post URL, fallback to displayUrl/videoUrl)
    if "url" in df.columns:
        url = df["url"]
    elif "displayUrl" in df.columns:
        url = df["displayUrl"]
    elif "videoUrl" in df.columns:
        url = df["videoUrl"]
    else:
        url = pd.Series([None]*len(df))

    likes = df.get("likesCount", pd.Series([np.nan]*len(df))).apply(to_int)
    comments = df.get("commentsCount", pd.Series([np.nan]*len(df))).apply(to_int)

    # shares: IG sometimes has reshareCount (use it if present)
    shares = df.get("reshareCount", pd.Series([0]*len(df))).apply(to_int).fillna(0).astype(int)

    # saves: often not available; keep 0 unless you later capture it
    saves = pd.Series([0]*len(df))

    # plays: pick best available
    plays_raw = None
    for col in ["videoPlayCount", "igPlayCount", "fbPlayCount"]:
        if col in df.columns:
            plays_raw = df[col]
            break
    plays = plays_raw.apply(to_int) if plays_raw is not None else pd.Series([0]*len(df))
    plays = plays.fillna(0)

    # audio name
    audio_name = df.get("musicInfo/song_name", pd.Series([None]*len(df)))

    # timestamp
    ts = df.get("timestamp", pd.Series([None]*len(df))).apply(parse_timestamp)

    # hashtags combined
    hashtags = df.apply(lambda r: join_hashtags(r, hashtag_cols), axis=1) if hashtag_cols else pd.Series([""]*len(df))

    # not in IG export
    upvotes = pd.Series([0]*len(df))
    creator_followers = pd.Series([np.nan]*len(df))

    # hours since post
    hours_since_post = (now_utc - ts).dt.total_seconds() / 3600
    hours_since_post = hours_since_post.clip(lower=0.01)

    # engagement score
    denom = plays.replace(0, 1)
    engagement_score = (
        likes.fillna(0)
        + 2 * comments.fillna(0)
        + 3 * shares.fillna(0)
        + 2 * saves.fillna(0)
        + upvotes.fillna(0)
    ) / denom

    # velocity: engagements per hour
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
        "hashtags": hashtags.fillna(""),
        "creator": creator.fillna(""),
        "creator_fullname": creator_fullname.fillna(""),
        "creator_followers": creator_followers,
        "timestamp": ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "url": url.fillna(""),
        "likes": likes.fillna(0).astype(int),
        "comments": comments.fillna(0).astype(int),
        "shares": shares.fillna(0).astype(int),
        "plays": plays.fillna(0).astype(int),
        "saves": saves.fillna(0).astype(int),
        "upvotes": upvotes.fillna(0).astype(int),
        "engagement_score": engagement_score.fillna(0),
        "hours_since_post": hours_since_post,
        "velocity": velocity.fillna(0),
        "audio_name": audio_name.fillna(""),
        "category": category
    })

    if DEDUP_BY_URL:
        clean = clean.drop_duplicates(subset=["url"], keep="first")

    final_cols = [
        "platform","caption","text","hashtags","creator","creator_fullname","creator_followers",
        "timestamp","url","likes","comments","shares","plays","saves","upvotes","engagement_score",
        "hours_since_post","velocity","audio_name","category"
    ]
    return clean[final_cols]


def main():
    df = pd.read_csv(INPUT_CSV)
    clean = clean_instagram_df(df, CATEGORY, now_utc=NOW_UTC)
    clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  ({len(clean)} rows)")

if __name__ == "__main__":
    main()