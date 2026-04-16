# fit bayesian models

library(brms)

options(mc.cores = parallel::detectCores())

source("load_data.R")

# model to predict price
price_model <- brm(
  log_price ~ neigh_median_price + room_type + accommodates +
              log_reviews + review_complete,
  data    = df,
  family  = gaussian(),
  prior   = c(
    prior(normal(5.0, 1.0), class = "Intercept"),
    prior(normal(0, 0.5),   class = "b"),
    prior(exponential(1),   class = "sigma")
  ),
  chains = 4, iter = 2000, warmup = 500,
  seed = 42
)

# model to predict ghost listings
ghost_vars <- c("ghost", "borough", "room_type", "accommodates",
                "log_host_count", "price_vs_neigh", "review_complete", "log_reviews")
df_ghost <- df %>% drop_na(all_of(ghost_vars))

ghost_model <- brm(
  ghost ~ borough + room_type + accommodates +
          log_host_count + price_vs_neigh +
          review_complete + log_reviews,
  data    = df_ghost,
  family  = bernoulli(link = "logit"),
  prior   = c(
    prior(normal(-3, 1),  class = "Intercept"),
    prior(normal(0, 0.5), class = "b")
  ),
  chains = 4, iter = 2000, warmup = 500,
  seed = 42
)

saveRDS(price_model, "price_model.rds")
saveRDS(ghost_model, "ghost_model.rds")

cat("Done. Models saved to price_model.rds and ghost_model.rds\n")