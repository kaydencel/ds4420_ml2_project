library(rstanarm)
library(tidyverse)

setwd("/Users/ayushdesai/Documents/Data\ Science/Machine\ Learning\ 2/ds4420_ml2_project/") 

options(mc.cores = parallel::detectCores())

df <- read_csv("airbnb_clean.csv") %>%
  mutate(
    price_num    = as.numeric(str_replace_all(price, "[$,]", "")),
    log_price    = log(price_num),
    borough      = factor(neighbourhood_group_cleansed),
    room_type    = factor(room_type),
    neigh_median_price = ave(
      log(as.numeric(str_replace_all(price, "[$,]", ""))),
      neighbourhood_cleansed,
      FUN = function(x) median(x, na.rm = TRUE)
    ),
    price_vs_neigh  = log_price - neigh_median_price,
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

price_model <- stan_glm(
  log_price ~ neigh_median_price + room_type + accommodates +
              log_reviews + review_complete,
  data            = df,
  family          = gaussian(),
  prior_intercept = normal(5.0, 1.0),
  prior           = normal(0, 0.5),
  prior_aux       = exponential(1),
  chains = 4, iter = 2000, warmup = 500,
  seed = 42
)

ghost_vars <- c("ghost", "borough", "room_type", "accommodates",
                "log_host_count", "price_vs_neigh", "review_complete", "log_reviews")
df_ghost <- df %>% drop_na(all_of(ghost_vars))

ghost_model <- stan_glm(
  ghost ~ borough + room_type + accommodates +
          log_host_count + price_vs_neigh +
          review_complete + log_reviews,
  data            = df_ghost,
  family          = binomial(link = "logit"),
  prior_intercept = normal(-3, 1),
  prior           = normal(0, 0.5),
  chains = 4, iter = 2000, warmup = 500,
  seed = 42
)

saveRDS(price_model, "price_model.rds")
saveRDS(ghost_model, "ghost_model.rds")

cat("Done. Models saved to price_model.rds and ghost_model.rds\n")
