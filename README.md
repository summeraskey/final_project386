```markdown
# YouTube Performance Analysis
**STAT 386 Final Project — Summer Price & Jane Gustafson**

This package analyzes YouTube trending video data by merging the Kaggle YouTube Trending Videos dataset with live data from the YouTube Data API. The result is a custom longitudinal dataset that tracks how trending videos from 2017 have grown over time, enabling exploratory analysis and predictive modeling of video performance.

## What This Package Does

- Downloads the Kaggle YouTube Trending Videos dataset (US, Nov 2017 - Jun 2018)
- Fetches current view, like, and comment counts for each video via the YouTube Data API
- Merges both sources into a single cleaned dataset
- Runs exploratory data analysis across 5 dimensions (growth, trending patterns, categories, engagement, time to trend)
- Trains 3 Random Forest models to predict current views, time to trend, and view growth

## Quick Start

```bash
git clone https://github.com/summeraskey/final_project386.git
cd final_project386
uv venv
source .venv/bin/activate
uv sync
```

Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_youtube_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

## Usage

```python
from final_project_demo import run_cleaning_pipeline, run_analysis_pipeline

df = run_cleaning_pipeline()
run_analysis_pipeline(df)
```

## Streamlit App

An interactive model predictor is hosted at:
https://finalproject386-qpuktjzfa562fbmkaycd9v.streamlit.app/

To run locally:
```bash
streamlit run src/final_project_demo/streamlit_app.py
```

## GitHub Pages Site

Full documentation, tutorial, and technical report are hosted at:
https://summeraskey.github.io/final_project386/

## Project Structure

```
final_project386/
├── src/final_project_demo/
│   ├── cleaning.py        # Data loading and cleaning pipeline
│   ├── analysis.py        # EDA and predictive modeling
│   └── streamlit_app.py   # Interactive Streamlit app
├── docs/                  # Generated Quarto site
├── index.qmd              # Home page
├── Documentation.qmd      # Function reference
├── Tutorial.qmd           # Usage tutorial
├── TechnicalReport.qmd    # Full technical report
└── _quarto.yml            # Quarto configuration
```

## Rebuild the Site

```bash
quarto render
```

Serve locally with:
```bash
quarto preview
```
```
