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


def main():
    df = pd.read_csv(INPUT_CSV)

    # find hashtag columns like hashtags/0 ... hashtags/29 (or more)
    hashtag_cols = [c for c in df.columns if c.startswith("hashtags/")]

    # core fields (safe gets)
    caption = df.get("caption", pd.Series([None]*len(df)))
    creator = df.get("ownerUsername", df.get("ownerUsername", pd.Series([None]*len(df))))
    creator_fullname = df.get("ownerFullName", pd.Series([None]*len(df)))
    url = df.get("url", df.get("displayUrl", pd.Series([None]*len(df))))

    likes = df.get("likesCount", pd.Series([np.nan]*len(df))).apply(to_int)
    comments = df.get("commentsCount", pd.Series([np.nan]*len(df))).apply(to_int)

    # plays can be in videoPlayCount OR igPlayCount OR fbPlayCount; pick best available
    plays_raw = None
    for col in ["videoPlayCount", "igPlayCount", "fbPlayCount"]:
        if col in df.columns:
            plays_raw = df[col]
            break
    if plays_raw is None:
        plays = pd.Series([np.nan]*len(df))
    else:
        plays = plays_raw.apply(to_int)

    # audio name
    audio_name = df.get("musicInfo/song_name", pd.Series([None]*len(df)))

    # timestamp
    ts = df.get("timestamp", pd.Series([None]*len(df))).apply(parse_timestamp)

    # hashtags combined
    hashtags = df.apply(lambda r: join_hashtags(r, hashtag_cols), axis=1) if hashtag_cols else ""

    # metrics not in IG export -> set to 0 or NA (your choice)
    shares = pd.Series([0]*len(df))
    saves = pd.Series([0]*len(df))
    upvotes = pd.Series([0]*len(df))
    creator_followers = pd.Series([np.nan]*len(df))  # IG export usually doesn't include this

    # hours since post
    hours_since_post = (NOW_UTC - ts).dt.total_seconds() / 3600
    # avoid division by 0 or negative
    hours_since_post = hours_since_post.clip(lower=0.01)

    # engagement score: simple, tunable
    # (likes + 2*comments + 3*shares + 2*saves + upvotes) / max(plays,1)
    denom = plays.fillna(0).replace(0, 1)
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
        "platform": PLATFORM,
        "caption": caption.fillna(""),
        "text": caption.fillna(""),  # for IG, text = caption (you can change later)
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
        "category": CATEGORY
    })

    if DEDUP_BY_URL:
        clean = clean.drop_duplicates(subset=["url"], keep="first")

    # final column order EXACTLY as you requested
    final_cols = [
        "platform","caption","text","hashtags","creator","creator_fullname","creator_followers",
        "timestamp","url","likes","comments","shares","plays","saves","upvotes","engagement_score",
        "hours_since_post","velocity","audio_name","category"
    ]
    clean = clean[final_cols]

    clean.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV}  ({len(clean)} rows)")


if __name__ == "__main__":
    main()
