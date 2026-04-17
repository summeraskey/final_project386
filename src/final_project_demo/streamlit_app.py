import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from final_project_demo import run_cleaning_pipeline
from final_project_demo.analysis import (
    predict_current_views,
    predict_time_to_trend,
    predict_view_growth
)

load_dotenv()

st.set_page_config(page_title="YouTube Video Predictor", layout="wide")
st.title("YouTube Video Performance Predictor")
st.markdown("Explore YouTube trending video data and predict how a video will perform based on its early engagement metrics.")

category_map = {
    "Film & Animation": 1, "Autos & Vehicles": 2, "Music": 10,
    "Pets & Animals": 15, "Sports": 17, "Travel & Events": 19,
    "Gaming": 20, "People & Blogs": 22, "Comedy": 23,
    "Entertainment": 24, "News & Politics": 25, "Howto & Style": 26,
    "Education": 27, "Science & Technology": 28, "Nonprofits & Activism": 29
}
category_map_reverse = {v: k for k, v in category_map.items()}

@st.cache_resource
def load_models():
    df = run_cleaning_pipeline()
    model1 = predict_current_views(df)
    model2 = predict_time_to_trend(df)
    model3 = predict_view_growth(df)
    return df, model1, model2, model3

with st.spinner("Loading data and training models... this may take a minute!"):
    df, model1, model2, model3 = load_models()

st.success("Models ready!")
st.divider()

tab1, tab2, tab3 = st.tabs(["EDA", "Model Visualizations", "Predictor"])

# ---- TAB 1: EDA ----
with tab1:
    st.header("Exploratory Data Analysis")

    df_unique = df.drop_duplicates(subset="video_id").copy()
    df_unique["category"] = df_unique["category_id"].map(category_map_reverse)
    df_unique["view_growth"] = df_unique["views_current"] - df_unique["views_2017"]
    df_unique["like_rate"] = df_unique["likes_current"] / df_unique["views_current"]
    df_unique["time_to_trend"] = (df_unique["trending_date"] - df_unique["publish_time"].dt.tz_localize(None)).dt.days

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Unique Videos", f"{df_unique.shape[0]:,}")
    with col2:
        st.metric("Median View Growth", f"{df_unique['view_growth'].median():,.0f}")
    with col3:
        st.metric("Median Days to Trend", f"{df_unique['time_to_trend'].clip(0).median():.0f}")

    st.subheader("View Growth Distribution")
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(df_unique["view_growth"], log_scale=True, bins=50, ax=ax1)
    ax1.set_xlabel("View Growth")
    ax1.set_title("Distribution of View Growth (2017 to Current)")
    st.pyplot(fig1)
    plt.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Average Views by Category")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        cat_views = df_unique.groupby("category")["views_current"].mean().sort_values(ascending=False)
        sns.barplot(x=cat_views.values, y=cat_views.index, ax=ax2)
        ax2.set_xlabel("Average Views")
        st.pyplot(fig2)
        plt.close()

    with col2:
        st.subheader("Average Like Rate by Category")
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        cat_likes = df_unique.groupby("category")["like_rate"].mean().sort_values(ascending=False)
        sns.barplot(x=cat_likes.values, y=cat_likes.index, ax=ax3)
        ax3.set_xlabel("Likes / Views")
        st.pyplot(fig3)
        plt.close()

    st.subheader("Days from Publishing to Trending")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    sns.histplot(df_unique["time_to_trend"].clip(0, 30), bins=30, ax=ax4)
    ax4.set_xlabel("Days")
    ax4.set_title("Days from Publishing to Trending")
    st.pyplot(fig4)
    plt.close()

    st.subheader("Trending Videos by Day of Week")
    df["day_of_week"] = df["trending_date"].dt.day_name()
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    sns.countplot(data=df, x="day_of_week", order=day_order, ax=ax5)
    ax5.set_xlabel("Day of Week")
    ax5.set_ylabel("Count")
    st.pyplot(fig5)
    plt.close()

# ---- TAB 2: Model Visualizations ----
with tab2:
    st.header("Model Visualizations")

    features1 = ["likes_2017", "dislikes", "comments_2017", "category_id", "comments_disabled", "ratings_disabled"]
    features2 = ["likes_2017", "views_2017", "comments_2017", "category_id", "publish_hour", "publish_day", "comments_disabled", "ratings_disabled"]
    features3 = ["views_2017", "likes_2017", "comments_2017", "category_id", "time_to_trend"]

    st.subheader("Model 1: Feature Importance for Predicting Current Views")
    fig6, ax6 = plt.subplots(figsize=(8, 4))
    imp1 = pd.Series(model1.feature_importances_, index=features1).sort_values()
    imp1.plot(kind="barh", ax=ax6)
    ax6.set_xlabel("Importance")
    st.pyplot(fig6)
    plt.close()

    st.subheader("Model 2: Feature Importance for Predicting Time to Trend")
    fig7, ax7 = plt.subplots(figsize=(8, 5))
    imp2 = pd.Series(model2.feature_importances_, index=features2).sort_values()
    imp2.plot(kind="barh", ax=ax7)
    ax7.set_xlabel("Importance")
    st.pyplot(fig7)
    plt.close()

    st.subheader("Model 3: Feature Importance for Predicting View Growth")
    fig8, ax8 = plt.subplots(figsize=(8, 4))
    imp3 = pd.Series(model3.feature_importances_, index=features3).sort_values()
    imp3.plot(kind="barh", ax=ax8)
    ax8.set_xlabel("Importance")
    st.pyplot(fig8)
    plt.close()

    st.subheader("Model Performance Summary")
    perf_df = pd.DataFrame({
        "Model": ["Model 1: Current Views", "Model 2: Time to Trend", "Model 3: View Growth"],
        "R² Score": [0.679, 0.274, 0.669],
        "Top Feature": ["likes_2017 (71%)", "comments_2017 (29%)", "likes_2017 (62%)"]
    })
    st.dataframe(perf_df, use_container_width=True)

# ---- TAB 3: Predictor ----
with tab3:
    st.header("Predict Video Performance")
    st.markdown("Enter the engagement stats for a video at the time it was trending to get predictions.")

    col1, col2 = st.columns(2)

    with col1:
        likes = st.number_input("Likes at time of trending", min_value=0, value=10000, step=1000)
        dislikes = st.number_input("Dislikes at time of trending", min_value=0, value=500, step=100)
        comments = st.number_input("Comments at time of trending", min_value=0, value=1000, step=100)
        views = st.number_input("Views at time of trending", min_value=0, value=500000, step=10000)

    with col2:
        category = st.selectbox("Category", options=list(category_map.keys()))
        comments_disabled = st.selectbox("Comments disabled?", options=["No", "Yes"])
        ratings_disabled = st.selectbox("Ratings disabled?", options=["No", "Yes"])
        publish_hour = st.slider("Publish hour (0-23)", min_value=0, max_value=23, value=12)
        publish_day = st.selectbox("Publish day", options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

    category_id = category_map[category]
    comments_disabled_val = 1 if comments_disabled == "Yes" else 0
    ratings_disabled_val = 1 if ratings_disabled == "Yes" else 0
    publish_day_val = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(publish_day)

    if st.button("Predict", type="primary"):

        X1 = pd.DataFrame([{
            "likes_2017": likes,
            "dislikes": dislikes,
            "comments_2017": comments,
            "category_id": category_id,
            "comments_disabled": comments_disabled_val,
            "ratings_disabled": ratings_disabled_val
        }])
        pred1 = np.expm1(model1.predict(X1)[0])

        X2 = pd.DataFrame([{
            "likes_2017": likes,
            "views_2017": views,
            "comments_2017": comments,
            "category_id": category_id,
            "publish_hour": publish_hour,
            "publish_day": publish_day_val,
            "comments_disabled": comments_disabled_val,
            "ratings_disabled": ratings_disabled_val
        }])
        pred2 = np.expm1(model2.predict(X2)[0])

        X3 = pd.DataFrame([{
            "views_2017": views,
            "likes_2017": likes,
            "comments_2017": comments,
            "category_id": category_id,
            "time_to_trend": max(pred2, 0)
        }])
        pred3 = np.expm1(model3.predict(X3)[0])

        st.divider()
        st.subheader("Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Predicted Current Views", f"{pred1:,.0f}")
            st.caption("Model 1: Estimated total views today")

        with col2:
            st.metric("Predicted Days to Trend", f"{pred2:.1f} days")
            st.caption("Model 2: How quickly it would trend")

        with col3:
            st.metric("Predicted View Growth", f"{pred3:,.0f}")
            st.caption("Model 3: Views gained since trending")