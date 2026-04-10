# NYC Airbnb — Bayesian Modeling & Content-Based Filtering

Final project for DS 4420: Machine Learning 2, Spring 2026.

This project analyzes NYC Airbnb listings through two lenses: a **Bayesian modeling pipeline in R** (price prediction and ghost listing detection, deployed as an interactive Shiny app) and a **content-based recommender system in Python** (cosine similarity over listing features).

---

## Project Structure

```
Preprocessing Original.ipynb   # Python: cleans raw data → airbnb_clean.csv
Content Based Filtering.ipynb  # Python: content-based recommender (cosine similarity)

fit_models.r                   # R: fits Bayesian price + ghost models via rstanarm
prepare_data.R                 # R: extracts model artifacts for the Shiny app
app.R                          # R: interactive Shiny web app (shinylive-compatible)
modeling_rmd.rmd               # R: local R Markdown version of the Shiny app
```

---

## Pipeline

### Python

1. **`Preprocessing Original.ipynb`** — Loads `Nov_data.csv` (raw Inside Airbnb scrape), audits missing values, drops sparse columns, and saves `airbnb_clean.csv`.
2. **`Content Based Filtering.ipynb`** — Builds a 36,353 × 25 feature matrix (one-hot borough/room type, scaled guest capacity, binary amenity flags) and ranks listings by cosine similarity to user preferences.

### R

1. **`fit_models.r`** — Fits two Bayesian regression models on `airbnb_clean.csv` using `rstanarm` (4 chains, 2,000 iterations):
   - **Price model** (Gaussian): predicts log(nightly price)
   - **Ghost model** (logistic): predicts probability a listing is inactive/fraudulent
2. **`prepare_data.R`** — Run once after model fitting. Extracts posterior draws matrices, credible intervals, and summary tables; renders the posterior predictive check plot; saves all artifacts as `.rds` files for the web app.
3. **`app.R`** — Shiny app with four tabs: project overview, price prediction, ghost detection, and a filterable table of flagged listings. Uses pre-computed `.rds` artifacts so it runs without `rstanarm` (shinylive/WebR compatible).

---

## Data

Data sourced from [Inside Airbnb](http://insideairbnb.com) — a public repository of scraped NYC Airbnb listings (November 2025 snapshot). Listings filtered to \$10–\$2,000/night.

---

## Live App

[https://your-username.github.io/your-repo-name/](https://your-username.github.io/your-repo-name/)
