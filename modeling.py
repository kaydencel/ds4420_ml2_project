#!/usr/bin/env python
# coding: utf-8

# # Bayesian Airbnb Analysis
# ### Price Prediction & Ghost Listing Detection

# ## Setup
# Run this first to force JAX backend — avoids PyTensor C compiler issues on Mac
import os
os.environ["PYTENSOR_FLAGS"] = "device=cpu,floatX=float64,optimizer=None"

import numpy as np
import pandas as pd
import pymc as pm
import bambi as bmb
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

az.style.use("arviz-darkgrid")

# ## Load & Clean

df = pd.read_csv("airbnb_clean.csv")

df["price_num"] = pd.to_numeric(
    df["price"].str.replace(r"[$,]", "", regex=True), errors="coerce"
)
df["log_price"] = np.log(df["price_num"])

df["neighbourhood_cleansed"] = df["neighbourhood_cleansed"].astype(str)
df["neigh_median_price"] = df.groupby("neighbourhood_cleansed")["log_price"].transform("median")
df["price_vs_neigh"]    = df["log_price"] - df["neigh_median_price"]

df["ghost"] = (
    (df["availability_365"] > 300) &
    (df["reviews_per_month"].fillna(0) < 0.1)
).astype(int)
df.loc[
    (df["reviews_per_month"].fillna(0) == 0) &
    (df["calculated_host_listings_count"] > 3),
    "ghost"
] = 1

df["log_reviews"]    = np.log1p(df["number_of_reviews"])
df["log_host_count"] = np.log1p(df["calculated_host_listings_count"])

review_cols = [
    "review_scores_rating", "review_scores_cleanliness",
    "review_scores_communication", "review_scores_location", "review_scores_value"
]
df["review_complete"] = df[review_cols].mean(axis=1)
df["room_type"]       = df["room_type"].astype(str)

df = df.dropna(subset=["log_price", "accommodates", "number_of_reviews",
                        "neigh_median_price", "review_complete"])
df = df[(df["price_num"] > 10) & (df["price_num"] < 2000)].reset_index(drop=True)

print(f"Listings: {len(df)} | Ghost-flagged: {df['ghost'].sum()}")

# ## Price Model
# Using bambi for R-style formula syntax.
# Priors mirror the R version: intercept N(5, 1), coefficients N(0, 0.5)

df_price = df.dropna(subset=["neigh_median_price", "room_type",
                               "accommodates", "log_reviews",
                               "review_complete", "log_price"]).reset_index(drop=True)

price_model = bmb.Model(
    "log_price ~ neigh_median_price + room_type + accommodates + log_reviews + review_complete",
    data   = df_price,
    family = "gaussian",
    priors = {
        "Intercept":          bmb.Prior("Normal", mu=5.0, sigma=1.0),
        "neigh_median_price": bmb.Prior("Normal", mu=0, sigma=0.5),
        "room_type":          bmb.Prior("Normal", mu=0, sigma=0.5),
        "accommodates":       bmb.Prior("Normal", mu=0, sigma=0.5),
        "log_reviews":        bmb.Prior("Normal", mu=0, sigma=0.5),
        "review_complete":    bmb.Prior("Normal", mu=0, sigma=0.5),
    }
)

price_trace = price_model.fit(
    draws=1500, tune=500, chains=4, random_seed=42,
    nuts_sampler="nutpie"   # pure-Python sampler, no C compiler needed
)

# ### Posterior Predictive Check

price_ppc = price_model.predict(price_trace, kind="pps")

az.plot_ppc(price_trace, num_pp_samples=50, group="posterior_predictive",
            observed_rug=True)
plt.title("Price Model — Posterior Predictive Check")
plt.xlabel("log(price)")
plt.tight_layout()
plt.show()

# ### Coefficient Credible Intervals

az.plot_forest(price_trace, var_names=["neigh_median_price", "room_type",
                                        "accommodates", "log_reviews",
                                        "review_complete"],
               hdi_prob=0.95, combined=True)
plt.title("95% Credible Intervals — Price Predictors")
plt.axvline(0, color="tomato", linestyle="--")
plt.tight_layout()
plt.show()

print(az.summary(price_trace, var_names=["neigh_median_price", "accommodates",
                                          "log_reviews", "review_complete"], hdi_prob=0.95))

# ### Predict Price for a New Listing

manhattan_median = df_price.loc[
    df_price["neighbourhood_group_cleansed"] == "Manhattan", "neigh_median_price"
].median()

new_listing = pd.DataFrame([{
    "neigh_median_price": manhattan_median,
    "room_type":          "Entire home/apt",
    "accommodates":       2,
    "log_reviews":        np.log1p(30),
    "review_complete":    4.8
}])

price_pred = price_model.predict(price_trace, kind="pps", data=new_listing)
pred_draws = np.exp(price_trace.posterior_predictive["log_price"].values.flatten())

print(f"\nMedian predicted price: ${np.median(pred_draws):.2f}")
print(f"90% Credible Interval:  ${np.percentile(pred_draws, 5):.2f} – ${np.percentile(pred_draws, 95):.2f}")

plt.figure(figsize=(8, 4))
plt.hist(pred_draws[pred_draws < 800], bins=60, color="steelblue", alpha=0.8)
plt.axvline(np.median(pred_draws), color="tomato", linewidth=1.5)
plt.title("Posterior Predictive Distribution — Nightly Price")
plt.xlabel("Predicted Price ($)")
plt.ylabel("Draw count")
plt.tight_layout()
plt.show()

# ## Ghost Listing Model

ghost_cols = ["ghost", "neighbourhood_group_cleansed", "room_type", "accommodates",
              "log_host_count", "price_vs_neigh", "review_complete", "log_reviews"]

df_ghost = df.dropna(subset=ghost_cols).reset_index(drop=True)
print(f"Ghost model rows: {len(df_ghost)}")

ghost_model = bmb.Model(
    "ghost ~ neighbourhood_group_cleansed + room_type + accommodates + "
    "log_host_count + price_vs_neigh + review_complete + log_reviews",
    data   = df_ghost,
    family = "bernoulli",
    priors = {
        "Intercept":                        bmb.Prior("Normal", mu=-3, sigma=1),
        "neighbourhood_group_cleansed":     bmb.Prior("Normal", mu=0, sigma=0.5),
        "room_type":                        bmb.Prior("Normal", mu=0, sigma=0.5),
        "accommodates":                     bmb.Prior("Normal", mu=0, sigma=0.5),
        "log_host_count":                   bmb.Prior("Normal", mu=0, sigma=0.5),
        "price_vs_neigh":                   bmb.Prior("Normal", mu=0, sigma=0.5),
        "review_complete":                  bmb.Prior("Normal", mu=0, sigma=0.5),
        "log_reviews":                      bmb.Prior("Normal", mu=0, sigma=0.5),
    }
)

ghost_trace = ghost_model.fit(
    draws=1500, tune=500, chains=4, random_seed=42,
    nuts_sampler="nutpie"
)

ghost_preds  = ghost_model.predict(ghost_trace, kind="pps", data=df_ghost, inplace=False)
df_ghost["ghost_prob"] = ghost_trace.posterior_predictive["ghost"].values.mean(axis=(0, 1))

# ### Ghost Rates by Borough

ghost_summary = (
    df_ghost.groupby("neighbourhood_group_cleansed")
    .agg(n=("ghost", "count"), n_flagged=("ghost", "sum"), avg_prob=("ghost_prob", "mean"))
    .assign(pct_flagged=lambda x: x["n_flagged"] / x["n"] * 100)
    .sort_values("avg_prob", ascending=False)
)
print(ghost_summary)

ghost_plot = ghost_summary.sort_values("avg_prob")
plt.figure(figsize=(7, 4))
plt.barh(ghost_plot.index, ghost_plot["avg_prob"], color="tomato", alpha=0.85)
for i, (borough, row) in enumerate(ghost_plot.iterrows()):
    plt.text(row["avg_prob"] + 0.002, i, f"{int(row['n_flagged'])} flagged",
             va="center", fontsize=9)
plt.xlabel("Mean P(ghost)")
plt.title("Posterior Ghost Probability by Borough")
plt.tight_layout()
plt.show()

# ### Ghost Coefficient Credible Intervals

az.plot_forest(ghost_trace,
               var_names=["neighbourhood_group_cleansed", "room_type", "accommodates",
                          "log_host_count", "price_vs_neigh", "review_complete", "log_reviews"],
               hdi_prob=0.95, combined=True)
plt.title("95% Credible Intervals — Ghost Listing Predictors")
plt.axvline(0, color="steelblue", linestyle="--")
plt.tight_layout()
plt.show()

# ### Export Flagged Listings

ghost_flagged = (
    df_ghost[df_ghost["ghost_prob"] > 0.15][[
        "id", "name", "neighbourhood_group_cleansed", "neighbourhood_cleansed",
        "room_type", "price_num", "availability_365",
        "reviews_per_month", "number_of_reviews", "ghost_prob"
    ]]
    .sort_values("ghost_prob", ascending=False)
)
ghost_flagged.to_csv("ghost_listings_flagged.csv", index=False)
print(f"Exported {len(ghost_flagged)} high-probability ghost listings.")
