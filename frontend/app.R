library(scales)
library(ggplot2)
library(dplyr)
library(stringr)
library(readr)
library(tibble)
library(tidyr)
library(shiny)
library(MASS)

# generate model files on first launch, cached on subsequent runs
local({
  rds <- c("data/ghost_model_data.rds", "data/factor_levels.rds")
  if (all(file.exists(rds))) return(invisible(NULL))

  message("Building model cache...")

  raw <- read_csv("data/airbnb_clean.csv",
                  col_types = cols(id = col_character()),
                  show_col_types = FALSE) |>
    dplyr::select(-1) |>
    mutate(
      price_num = as.numeric(gsub("[$,]", "", price)),
      last_review = as.Date(last_review),
      days_since_lr = as.numeric(Sys.Date() - last_review),
      days_since_lr = ifelse(is.na(days_since_lr), 9999, days_since_lr),
      reviews_per_month = ifelse(is.na(reviews_per_month), 0, reviews_per_month)
    )

  # price percentile within neighbourhood x room type
  raw <- raw |>
    group_by(neighbourhood_cleansed, room_type) |>
    mutate(price_pct = rank(price_num, na.last = "keep", ties.method = "average") /
             sum(!is.na(price_num))) |>
    ungroup()

  # ghost label: score > 2 of 5 signals
  df <- raw |>
    mutate(
      ghost_score = (availability_365 >= 180) +
        (minimum_nights >= 180) +
        (days_since_lr >= 180) +
        (!is.na(price_pct) & price_pct < 0.05) +
        (!host_has_profile_pic & !host_identity_verified),
      ghost_label = as.integer(ghost_score > 2),
      borough = factor(neighbourhood_group_cleansed),
      room_type_f = factor(room_type),
      log_reviews = log1p(number_of_reviews),
      review_complete = review_scores_rating,
      log_host_count = log1p(host_listings_count)
    )

  # ghost model
  df_g <- df |>
    filter(!is.na(log_host_count), !is.na(review_complete)) |>
    mutate(borough = droplevels(borough), room_type_f = droplevels(room_type_f))

  fit_g <- glm(ghost_label ~ borough + room_type_f + accommodates + log_host_count +
                 review_complete + log_reviews,
               family = binomial("logit"), data = df_g)
  set.seed(42)
  drw_g <- MASS::mvrnorm(4000, coef(fit_g), vcov(fit_g))
  ci_g <- t(apply(drw_g, 2, quantile, probs = c(0.025, 0.975)))
  saveRDS(list(draws = drw_g, ci = ci_g), "data/ghost_model_data.rds")

  saveRDS(list(boroughs = levels(df_g$borough),
               room_types = levels(df_g$room_type_f)),
          "data/factor_levels.rds")
  message("Model cache written.")
})

# load model data
ghost_data <- readRDS("data/ghost_model_data.rds")
fl <- readRDS("data/factor_levels.rds")
boroughs <- fl$boroughs
room_types <- fl$room_types

# build feature vector for ghost model
make_ghost_x <- function(borough, room_type, accommodates, log_host_count,
                         review_complete, log_reviews) {
  cn <- colnames(ghost_data$draws)
  x <- setNames(numeric(length(cn)), cn)
  x["(Intercept)"] <- 1
  b_col <- paste0("borough", borough)
  if (b_col %in% cn) x[b_col] <- 1
  rt_col <- paste0("room_type_f", room_type)
  if (rt_col %in% cn) x[rt_col] <- 1
  x["accommodates"] <- accommodates
  x["log_host_count"] <- log_host_count
  x["review_complete"] <- review_complete
  x["log_reviews"] <- log_reviews
  x
}

predict_ghost <- function(x) {
  mean(plogis(as.numeric(ghost_data$draws %*% x)))
}

# css
app_css <- "
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  background: #f5f3ee;
  color: #1c2434;
  margin: 0; padding: 0;
}
.navbar {
  background-color: #1f2d3d !important;
  border: none; border-radius: 0; margin-bottom: 0;
  box-shadow: 0 1px 4px rgba(0,0,0,0.18);
}
.navbar-brand {
  color: #edeae2 !important;
  font-weight: 500;
  font-size: 15px;
  letter-spacing: 0.01em;
}
.navbar-nav > li > a {
  color: #a8b4be !important;
  font-size: 15px;
  letter-spacing: 0.01em;
  padding: 18px 18px;
}
.navbar-nav > li > a:hover,
.navbar-nav > .active > a,
.navbar-nav > .active > a:focus,
.navbar-nav > .active > a:hover {
  background-color: rgba(255,255,255,0.07) !important;
  color: #edeae2 !important;
}
.tab-content { background: #f5f3ee; }
.tab-pane { padding: 28px 28px 52px 28px; }
.stat-box {
  background: #fff;
  border-left: 3px solid #2e5b8a;
  padding: 16px 20px;
  margin-bottom: 14px;
  border-radius: 5px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
.stat-box .stat-label {
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: #8a9299;
  margin: 0 0 6px 0;
}
.stat-box .stat-value {
  font-size: 1.8em;
  font-weight: 600;
  margin: 0;
  color: #1c2434;
  line-height: 1.2;
}
.section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: #9aa4ae;
  border-bottom: 1px solid #e6e1d8;
  padding-bottom: 7px;
  margin: 0 0 14px 0;
}
.well {
  background-color: #fff;
  border: 1px solid #e6e1d8;
  border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.05);
  padding: 20px;
}
.form-control, .selectize-input {
  border-radius: 4px !important;
  border: 1px solid #d5cfc5 !important;
  font-size: 13px !important;
  background: #faf9f6 !important;
}
label {
  font-size: 12px;
  font-weight: 600;
  color: #4a5568;
  letter-spacing: 0.01em;
}
.irs--shiny .irs-bar, .irs--shiny .irs-single {
  background: #2e5b8a !important;
  border-color: #2e5b8a !important;
}
.irs--shiny .irs-handle { border-color: #2e5b8a !important; }
.btn-run {
  width: 100%;
  background-color: #1f2d3d;
  color: #edeae2;
  border: none;
  padding: 10px 0;
  border-radius: 4px;
  font-size: 13px;
  letter-spacing: 0.04em;
  cursor: pointer;
  transition: background 0.15s;
}
.btn-run:hover { background-color: #2e5b8a; color: #fff; }
.plot-panel {
  background: #fff;
  border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  padding: 20px 22px 14px 22px;
  margin-bottom: 16px;
}
.overview-section {
  max-width: 760px;
  margin: 0 auto;
  padding: 16px 0 44px 0;
  line-height: 1.78;
}
.overview-section h4 {
  color: #1c2434;
  border-bottom: 1px solid #e6e1d8;
  padding-bottom: 6px;
  margin-top: 28px;
  font-size: 1em;
  font-weight: 600;
  letter-spacing: 0.01em;
}
.overview-section p, .overview-section li {
  font-size: 1.05em;
  color: #3a4450;
  line-height: 1.82;
}
.dataTables_wrapper { font-size: 13px; }
table.dataTable thead th {
  background: #f5f3ee;
  color: #4a5568;
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0.03em;
  border-bottom: 2px solid #e6e1d8 !important;
}
"

# ui
ui <- navbarPage(
  title = "NYC Airbnb Ghost Listings",
  windowTitle = "NYC Airbnb Ghost Listings",
  header = tags$head(tags$style(HTML(app_css))),

  tabPanel("Overview",
    fluidRow(column(8, offset = 2,
      div(class = "overview-section",
        p("A ghost listing is an Airbnb property that appears active on the
          platform but shows little to no genuine guest activity. Listings are flagged based on
          signals like high availability, long gaps since the last review, unusually low pricing,
          and unverified host identity. Listings that trigger enough of these signals are labeled
          as ghost listings."),
        p("Ghost listings distort platform supply, may manipulate neighborhood pricing benchmarks,
          and can represent fraudulent or inactive inventory."),
        p("This app uses two models. The ", tags$b("Check a Listing"), " tab runs a statistical
          model that estimates ghost likelihood from listing characteristics. The ",
          tags$b("Model Performance"), " tab shows evaluation results for a neural network
          trained on 14,990 NYC listings, which achieves 82.5% accuracy and 0.876 AUC."),
        p(style = "color: #9aa4ae; font-size: 0.88em;",
          "Data source: Inside Airbnb | 36,353 New York City listings | Analysis based on November 2025 data"))
    ))
  ),

  tabPanel("Check a Listing",
    sidebarLayout(
      sidebarPanel(width = 3,
        div(class = "section-title", "Listing Details"),
        selectInput("ghost_borough", "Borough", choices = boroughs, selected = "Manhattan"),
        selectInput("room_type_ghost", "Room Type", choices = room_types, selected = "Entire home/apt"),
        selectInput("ghost_accommodates", "Guests",
                    choices = as.character(1:10), selected = "2"),
        selectInput("host_listings", "Host's Total Listings",
                    choices = as.character(1:25), selected = "1"),
        selectInput("ghost_review_score", "Review Score",
                    choices = as.character(seq(1, 5, by = 0.5)), selected = "4.5"),
        selectInput("ghost_n_reviews", "Number of Reviews",
                    choices = c("0 -10" = "5", "11 -25" = "18", "26 -50" = "38",
                                "51 -100" = "75", "101 -200" = "150", "201 -500" = "350"),
                    selected = "18"),
        br(),
        actionButton("predict_ghost", "Estimate Ghost Risk", class = "btn-run")
      ),
      mainPanel(width = 9,
        uiOutput("ghost_prob_card")
      )
    )
  ),

  tabPanel("Model Performance",
    br(),
    fluidRow(column(10, offset = 1,
      div(class = "plot-panel",
        div(class = "section-title", "Model Training & Accuracy"),
        img(src = "mlp_diagnostics.png", width = "100%", style = "border-radius:4px;")
      ),
      div(class = "plot-panel",
        div(class = "section-title", "Prediction Results"),
        img(src = "mlp_confusion_matrix.png", width = "40%",
            style = "display:block; margin:0 auto; border-radius:4px;")
      )
    ))
  ),

)

# server
server <- function(input, output, session) {

  ghost_prob_val <- eventReactive(input$predict_ghost, {
    x <- make_ghost_x(
      borough = input$ghost_borough,
      room_type = input$room_type_ghost,
      accommodates = as.integer(input$ghost_accommodates),
      log_host_count = log1p(as.numeric(input$host_listings)),
      review_complete = as.numeric(input$ghost_review_score),
      log_reviews = log1p(as.numeric(input$ghost_n_reviews))
    )
    predict_ghost(x)
  }, ignoreNULL = FALSE)

  output$ghost_prob_card <- renderUI({
    p_val <- ghost_prob_val()
    pct <- scales::percent(p_val, accuracy = 0.1)
    border_col <- if (p_val < 0.10) "#27ae60" else if (p_val < 0.30) "#e67e22" else "#c0392b"
    lbl <- if (p_val < 0.10) "Low risk — listing appears active"
           else if (p_val < 0.30) "Moderate risk — worth investigating"
           else "High risk — likely ghost listing"
    div(
      style = paste0("border-left:5px solid ", border_col,
                     "; background:#fff; padding:16px 22px; border-radius:4px;",
                     "box-shadow:0 1px 3px rgba(0,0,0,0.08); margin-bottom:16px;"),
      p(style = "margin:0; font-size:11px; text-transform:uppercase;
                 letter-spacing:0.09em; color:#8a9299;",
        "Ghost Risk Score"),
      p(style = paste0("margin:6px 0 4px 0; font-size:2.2em; font-weight:bold; color:",
                       border_col, "; line-height:1.1;"), pct),
      p(style = "margin:0; font-size:0.88em; color:#5f6b7a;", lbl)
    )
  })

  param_labels <- c(
    boroughBronx             = "Borough: Bronx",
    boroughBrooklyn          = "Borough: Brooklyn",
    boroughManhattan         = "Borough: Manhattan",
    boroughQueens            = "Borough: Queens",
    "boroughStaten Island"   = "Borough: Staten Island",
    "room_type_fHotel room"  = "Room Type: Hotel",
    "room_type_fPrivate room"= "Room Type: Private Room",
    "room_type_fShared room" = "Room Type: Shared Room",
    accommodates             = "Guests Accommodated",
    log_host_count           = "Host Listing Count",
    review_complete          = "Review Score",
    log_reviews              = "Number of Reviews"
  )

  output$ghost_coef_plot <- renderPlot({
    tryCatch({
      as.data.frame(ghost_data$ci) %>%
        rownames_to_column("param") %>%
        filter(param != "(Intercept)") %>%
        mutate(label = dplyr::coalesce(param_labels[param], param)) %>%
        ggplot(aes(y = reorder(label, `97.5%`))) +
        geom_segment(aes(x = `2.5%`, xend = `97.5%`, yend = reorder(label, `97.5%`)),
                     linewidth = 1.4, color = "#2e5b8a", alpha = 0.85) +
        geom_point(aes(x = (`2.5%` + `97.5%`) / 2), size = 3, color = "#1f2d3d") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#b83232", linewidth = 0.65) +
        labs(x = "Effect on ghost risk  (right of zero = higher risk)", y = NULL) +
        theme_minimal(base_size = 11) +
        theme(
          panel.grid.minor = element_blank(),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(color = "#ede9e0"),
          axis.text.y = element_text(size = 10, color = "#3a4450"),
          axis.title.x = element_text(size = 10, color = "#5f6b7a"),
          plot.background = element_rect(fill = "#ffffff", color = NA)
        )
    }, error = function(e) {
      ggplot() + annotate("text", x = 0.5, y = 0.5,
        label = paste("Plot unavailable:", conditionMessage(e)),
        size = 4, color = "#b83232") + theme_void()
    })
  }, res = 96)

}

shinyApp(ui, server)
