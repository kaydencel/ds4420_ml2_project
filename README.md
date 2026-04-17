**NYC Airbnb | Ghost Listing Detection**

Final project for DS 4420: Machine Learning 2, Spring 2026. This project analyzes 36,353 NYC Airbnb listings to identify ghost listings, properties that appear active on the platform but show little to no genuine guest activity, using a Bayesian model and a MLP neural network. The MLP results are deployed in an interactive Shiny app.

**How It Works**

A listing is flagged as a ghost if it triggers more than 2 of 5 signals: availability of 180+ days per year, minimum stay of 180+ nights, no review in the past 180 days, price in the bottom 5th percentile for its neighborhood and room type, or an unverified host with no profile picture.

The app has two models. The Check a Listing tab runs a logistic regression that estimates ghost probability from borough, room type, guest capacity, host listing count, review score, and number of reviews, with uncertainty quantified via 4,000 posterior draws using MASS::mvrnorm. The Model Performance tab shows results for a neural network trained on 14,990 listings, achieving 82.5% accuracy and an AUC of 0.876.

**Data**

Sourced from Inside Airbnb, a public repository of scraped NYC Airbnb listings (November 2025 snapshot). 

**Live App**

https://airbnb-analysis-ds4420.shinyapps.io/Ghost-Listings-AirBNB/
