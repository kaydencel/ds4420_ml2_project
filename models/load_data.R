library(tidyverse)

df <- read_csv("../data/airbnb_clean.csv") %>%
  mutate(
    price_num  = as.numeric(str_replace_all(price, "[$,]", "")),
    log_price  = log(price_num),
    borough    = factor(neighbourhood_group_cleansed),
    room_type  = factor(room_type),
    neigh_median_price = ave(
      log(as.numeric(str_replace_all(price, "[$,]", ""))),
      neighbourhood_cleansed,
      FUN = function(x) median(x, na.rm = TRUE)
    ),
    price_vs_neigh = log_price - neigh_median_price,
    ghost = case_when(
      availability_365 > 300 & (is.na(reviews_per_month) | reviews_per_month < 0.1) ~ 1L,
      reviews_per_month == 0 & calculated_host_listings_count > 3 ~ 1L,
      TRUE ~ 0L
    ),
    log_reviews     = log1p(number_of_reviews),
    log_host_count  = log1p(calculated_host_listings_count),
    review_complete = rowMeans(
      select(., review_scores_rating, review_scores_cleanliness,
             review_scores_communication, review_scores_location,
             review_scores_value),
      na.rm = TRUE
    )
  ) %>%
  filter(
    !is.na(log_price),
    price_num > 10, price_num < 2000,
    !is.na(accommodates),
    !is.na(number_of_reviews)
  )
