import pandas as pd
import numpy as np
import kagglehub
from googleapiclient.discovery import build
from dotenv import load_dotenv
import os

def load_data():
    """Load and merge Kaggle dataset with YouTube API data"""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")
    youtube = build("youtube", "v3", developerKey=api_key)

    path = kagglehub.dataset_download("datasnaek/youtube-new")
    df_kaggle = pd.read_csv(f"{path}/USvideos.csv")

    video_ids = df_kaggle["video_id"].dropna().unique().tolist()
    rows = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i+50]
        response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch)
        ).execute()
        for item in response["items"]:
            rows.append({
                "video_id": item["id"],
                "title":    item["snippet"]["title"],
                "channel":  item["snippet"]["channelTitle"],
                "published": item["snippet"]["publishedAt"],
                "views":    int(item["statistics"].get("viewCount", 0)),
                "likes":    int(item["statistics"].get("likeCount", 0)),
                "comments": int(item["statistics"].get("commentCount", 0)),
            })
    df_api = pd.DataFrame(rows)
    return pd.merge(df_kaggle, df_api, on="video_id", how="inner")


def clean_data(df):
    """Rename columns, fix types, drop duplicates"""
    df = df.drop(columns=["title_y", "channel"])
    df = df.rename(columns={
        "title_x":       "title",
        "views_x":       "views_2017",
        "likes_x":       "likes_2017",
        "views_y":       "views_current",
        "likes_y":       "likes_current",
        "comment_count": "comments_2017",
        "comments":      "comments_current"
    })
    df["trending_date"] = pd.to_datetime(df["trending_date"], format="%y.%d.%m")
    df["publish_time"] = pd.to_datetime(df["publish_time"])
    df["published"] = pd.to_datetime(df["published"])
    return df


def run_cleaning_pipeline():
    """Main pipeline — load and clean data, return final df"""
    print("Running cleaning pipeline...")
    df = load_data()
    df = clean_data(df)
    print(f"Done! Shape: {df.shape}")
    return df