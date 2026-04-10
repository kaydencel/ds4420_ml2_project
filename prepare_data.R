library(rstanarm)
library(tidyverse)
library(bayesplot)

# ── Rebuild clean data ────────────────────────────────────────────────────────
df <- read_csv("airbnb_clean.csv") %>%
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
  filter(!is.na(log_price), price_num > 10, price_num < 2000,
         !is.na(accommodates), !is.na(number_of_reviews))

ghost_vars <- c("ghost", "borough", "room_type", "accommodates",
                "log_host_count", "price_vs_neigh", "review_complete", "log_reviews")
df_ghost <- df %>% drop_na(all_of(ghost_vars))

# ── Load models ───────────────────────────────────────────────────────────────
price_model <- readRDS("price_model.rds")
ghost_model <- readRDS("ghost_model.rds")

df_ghost$ghost_prob <- colMeans(posterior_epred(ghost_model))

# ── Posterior draws matrices ──────────────────────────────────────────────────
price_draws <- as.matrix(price_model)
ghost_draws <- as.matrix(ghost_model)

# ── Credible intervals ────────────────────────────────────────────────────────
price_ci <- posterior_interval(price_model, prob = 0.95)
ghost_ci  <- posterior_interval(ghost_model, prob = 0.95)

# ── PPC plot ─────────────────────────────────────────────────────────────────
dir.create("www", showWarnings = FALSE)
png("www/ppc_plot.png", width = 700, height = 220, res = 96)
print(
  pp_check(price_model, nreps = 50) +
    labs(x = "log(price)") +
    theme_minimal(base_size = 11) +
    theme(panel.grid.minor = element_blank(),
          panel.grid.major = element_line(color = "#eeeeee"))
)
dev.off()

# ── Reference lookups ─────────────────────────────────────────────────────────
borough_medians <- df %>%
  group_by(borough) %>%
  summarise(med = median(neigh_median_price, na.rm = TRUE), .groups = "drop")

factor_levels <- list(
  boroughs   = levels(df$borough),
  room_types = levels(df$room_type)
)

# ── Ghost listing table data ──────────────────────────────────────────────────
ghost_table_data <- df_ghost %>%
  select(id, name,
         neighbourhood_group_cleansed, neighbourhood_cleansed,
         room_type, price_num, availability_365,
         reviews_per_month, number_of_reviews, borough, ghost, ghost_prob)

# ── Save ──────────────────────────────────────────────────────────────────────
saveRDS(list(draws = price_draws, ci = price_ci), "price_model_data.rds")
saveRDS(list(draws = ghost_draws, ci = ghost_ci), "ghost_model_data.rds")
saveRDS(ghost_table_data, "df_ghost_data.rds")
saveRDS(borough_medians,  "borough_medians.rds")
saveRDS(factor_levels,    "factor_levels.rds")
