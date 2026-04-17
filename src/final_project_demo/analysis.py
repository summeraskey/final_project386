import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

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
    df_unique = df.drop_duplicates(subset="video_id").copy()
    df_unique["view_growth"] = df_unique["views_current"] - df_unique["views_2017"]
    df_unique["like_growth"] = df_unique["likes_current"] - df_unique["likes_2017"]
    df_unique["comment_growth"] = df_unique["comments_current"] - df_unique["comments_2017"]

    print(df_unique[["view_growth", "like_growth", "comment_growth"]].describe())
    print(df_unique[["title", "view_growth"]].sort_values("view_growth", ascending=False).head(10))

    plt.figure(figsize=(10, 5))
    sns.histplot(df_unique["view_growth"], log_scale=True, bins=50)
    plt.title("Distribution of View Growth (2017 to Current)")
    plt.xlabel("View Growth")
    plt.tight_layout()
    plt.savefig("growth_distribution.png")
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
    plt.tight_layout()
    plt.savefig("trending_by_day.png")
    plt.show()

    plt.figure(figsize=(10, 5))
    month_order = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    sns.countplot(data=df, x="month", order=month_order)
    plt.title("Number of Trending Videos by Month")
    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("trending_by_month.png")
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
    plt.tight_layout()
    plt.savefig("views_by_category.png")
    plt.show()

def engagement_analysis(df):
    """Analyze like rate and comment rate by category"""
    df = df.copy()
    df_unique = df.drop_duplicates(subset="video_id").copy()
    df_unique["category"] = df_unique["category_id"].map(category_map)
    df_unique["like_rate"] = df_unique["likes_current"] / df_unique["views_current"]
    df_unique["comment_rate"] = df_unique["comments_current"] / df_unique["views_current"]

    plt.figure(figsize=(12, 6))
    cat_likes = df_unique.groupby("category")["like_rate"].mean().sort_values(ascending=False)
    sns.barplot(x=cat_likes.values, y=cat_likes.index)
    plt.title("Average Like Rate by Category")
    plt.xlabel("Likes / Views")
    plt.tight_layout()
    plt.savefig("like_rate_by_category.png")
    plt.show()

def time_to_trend_analysis(df):
    """Analyze how long after publishing a video starts trending"""
    df = df.copy()
    df_unique = df.drop_duplicates(subset="video_id").copy()
    df_unique["time_to_trend"] = (df_unique["trending_date"] - df_unique["publish_time"].dt.tz_localize(None)).dt.days

    print(df_unique["time_to_trend"].describe())

    plt.figure(figsize=(10, 5))
    sns.histplot(df_unique["time_to_trend"].clip(0, 30), bins=30)
    plt.title("Days from Publishing to Trending")
    plt.xlabel("Days")
    plt.tight_layout()
    plt.savefig("time_to_trend.png")
    plt.show()

def predict_current_views(df):
    """Model 1: Predict current views from 2017 engagement metrics"""
    df_model = df.drop_duplicates(subset="video_id").copy()

    features = ["likes_2017", "dislikes", "comments_2017", "category_id",
                "comments_disabled", "ratings_disabled"]
    X = df_model[features].copy()
    X["comments_disabled"] = X["comments_disabled"].astype(int)
    X["ratings_disabled"] = X["ratings_disabled"].astype(int)
    y = np.log1p(df_model["views_current"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model 1 R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"Model 1 MAE (log scale): {mean_absolute_error(y_test, y_pred):.3f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Views (log scale)")
    plt.ylabel("Predicted Views (log scale)")
    plt.title("Model 1: Predicted vs Actual Current Views")
    plt.tight_layout()
    plt.savefig("model1_predicted_vs_actual.png")
    plt.show()

    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Model 1: Feature Importance for Predicting Current Views")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("model1_feature_importance.png")
    plt.show()

    return model

def predict_time_to_trend(df):
    """Model 2: Predict how quickly a video trends after publishing"""
    df_model = df.drop_duplicates(subset="video_id").copy()
    df_model["time_to_trend"] = (df_model["trending_date"] - df_model["publish_time"].dt.tz_localize(None)).dt.days
    df_model = df_model[df_model["time_to_trend"] >= 0]
    df_model["publish_hour"] = df_model["publish_time"].dt.hour
    df_model["publish_day"] = df_model["publish_time"].dt.dayofweek

    features = ["likes_2017", "views_2017", "comments_2017", "category_id",
                "publish_hour", "publish_day", "comments_disabled", "ratings_disabled"]
    X = df_model[features].copy()
    X["comments_disabled"] = X["comments_disabled"].astype(int)
    X["ratings_disabled"] = X["ratings_disabled"].astype(int)
    y = np.log1p(df_model["time_to_trend"])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model 2 R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"Model 2 MAE (log scale): {mean_absolute_error(y_test, y_pred):.3f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Time to Trend (log scale)")
    plt.ylabel("Predicted Time to Trend (log scale)")
    plt.title("Model 2: Predicted vs Actual Time to Trend")
    plt.tight_layout()
    plt.savefig("model2_predicted_vs_actual.png")
    plt.show()

    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Model 2: Feature Importance for Predicting Time to Trend")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("model2_feature_importance.png")
    plt.show()

    return model

def predict_view_growth(df):
    """Model 3: Predict long term view growth from 2017 to current"""
    df_model = df.drop_duplicates(subset="video_id").copy()
    df_model["time_to_trend"] = (df_model["trending_date"] - df_model["publish_time"].dt.tz_localize(None)).dt.days
    df_model = df_model[df_model["time_to_trend"] >= 0]
    df_model["view_growth"] = df_model["views_current"] - df_model["views_2017"]

    features = ["views_2017", "likes_2017", "comments_2017", "category_id", "time_to_trend"]
    X = df_model[features].copy()
    y = np.log1p(df_model["view_growth"].clip(lower=0))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Model 3 R² Score: {r2_score(y_test, y_pred):.3f}")
    print(f"Model 3 MAE (log scale): {mean_absolute_error(y_test, y_pred):.3f}")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual View Growth (log scale)")
    plt.ylabel("Predicted View Growth (log scale)")
    plt.title("Model 3: Predicted vs Actual View Growth")
    plt.tight_layout()
    plt.savefig("model3_predicted_vs_actual.png")
    plt.show()

    importances = pd.Series(model.feature_importances_, index=features).sort_values()
    plt.figure(figsize=(8, 5))
    importances.plot(kind="barh")
    plt.title("Model 3: Feature Importance for Predicting View Growth")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("model3_feature_importance.png")
    plt.show()

    return model

def run_analysis_pipeline(df):
    """Run all EDA and modeling functions"""
    print("Running analysis pipeline...")
    growth_analysis(df)
    trending_patterns(df)
    category_analysis(df)
    engagement_analysis(df)
    time_to_trend_analysis(df)
    predict_current_views(df)
    predict_time_to_trend(df)
    predict_view_growth(df)
    print("Analysis complete!")

def add(a, b):
    return a + b