import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

category_map = {
    1: "Film & Animation", 2: "Autos & Vehicles", 10: "Music",
    15: "Pets & Animals", 17: "Sports", 19: "Travel & Events",
    20: "Gaming", 22: "People & Blogs", 23: "Comedy",
    24: "Entertainment", 25: "News & Politics", 26: "Howto & Style",
    27: "Education", 28: "Science & Technology", 29: "Nonprofits & Activism"
}

def growth_analysis(df):
    """Analyze view, like, and comment growth from 2017 to current"""
    df = df.copy()
    df_unique = df.drop_duplicates(subset="video_id")
    df_unique["view_growth"] = df_unique["views_current"] - df_unique["views_2017"]
    df_unique["like_growth"] = df_unique["likes_current"] - df_unique["likes_2017"]
    df_unique["comment_growth"] = df_unique["comments_current"] - df_unique["comments_2017"]

    print(df_unique[["view_growth", "like_growth", "comment_growth"]].describe())
    print(df_unique[["title", "view_growth"]].sort_values("view_growth", ascending=False).head(10))

    plt.figure(figsize=(10, 5))
    sns.histplot(df_unique["view_growth"], log_scale=True, bins=50)
    plt.title("Distribution of View Growth (2017 to Current)")
    plt.xlabel("View Growth")
    plt.show()

    return df_unique

def trending_patterns(df):
    """Analyze trending patterns by day of week and month"""
    df = df.copy()
    df["day_of_week"] = df["trending_date"].dt.day_name()
    df["month"] = df["trending_date"].dt.month_name()

    plt.figure(figsize=(10, 5))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.countplot(data=df, x="day_of_week", order=day_order)
    plt.title("Number of Trending Videos by Day of Week")
    plt.xlabel("Day of Week")
    plt.ylabel("Count")
    plt.show()

    plt.figure(figsize=(10, 5))
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    sns.countplot(data=df, x="month", order=month_order)
    plt.title("Number of Trending Videos by Month")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

def category_analysis(df):
    """Analyze views and engagement by category"""
    df = df.copy()
    df["category"] = df["category_id"].map(category_map)

    plt.figure(figsize=(12, 6))
    cat_views = df.groupby("category")["views_current"].mean().sort_values(ascending=False)
    sns.barplot(x=cat_views.values, y=cat_views.index)
    plt.title("Average Current Views by Category")
    plt.xlabel("Average Views")
    plt.show()

def engagement_analysis(df):
    """Analyze like rate and comment rate by category"""
    df = df.copy()
    df_unique = df.drop_duplicates(subset="video_id")
    df_unique["category"] = df_unique["category_id"].map(category_map)
    df_unique["like_rate"] = df_unique["likes_current"] / df_unique["views_current"]
    df_unique["comment_rate"] = df_unique["comments_current"] / df_unique["views_current"]

    plt.figure(figsize=(12, 6))
    cat_likes = df_unique.groupby("category")["like_rate"].mean().sort_values(ascending=False)
    sns.barplot(x=cat_likes.values, y=cat_likes.index)
    plt.title("Average Like Rate by Category")
    plt.xlabel("Likes / Views")
    plt.show()

def time_to_trend_analysis(df):
    """Analyze how long after publishing a video starts trending"""
    df = df.copy()
    df_unique = df.drop_duplicates(subset="video_id")
    df_unique["time_to_trend"] = (df_unique["trending_date"] - df_unique["publish_time"].dt.tz_localize(None)).dt.days

    print(df_unique["time_to_trend"].describe())

    plt.figure(figsize=(10, 5))
    sns.histplot(df_unique["time_to_trend"].clip(0, 30), bins=30)
    plt.title("Days from Publishing to Trending")
    plt.xlabel("Days")
    plt.show()

def run_analysis_pipeline(df):
    """Run all analysis functions"""
    print("Running analysis pipeline...")
    growth_analysis(df)
    trending_patterns(df)
    category_analysis(df)
    engagement_analysis(df)
    time_to_trend_analysis(df)
    print("Analysis complete!")

def add(a, b):
    return a + b