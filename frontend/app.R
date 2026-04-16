library(scales)
library(ggplot2)
library(dplyr)
library(stringr)
library(readr)
library(tibble)
library(tidyr)
library(DT)
library(shiny)

# generate model files on first launch, cached on subsequent runs
local({
  rds <- c("../data/ghost_model_data.rds", "../data/factor_levels.rds")
  if (all(file.exists(rds))) return(invisible(NULL))

  message("Building model cache...")

  raw <- read_csv("../data/airbnb_clean.csv",
                  col_types = cols(id = col_character()),
                  show_col_types = FALSE) |>
    select(-1) |>
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
  saveRDS(list(draws = drw_g, ci = ci_g), "../data/ghost_model_data.rds")

  saveRDS(list(boroughs = levels(df_g$borough),
               room_types = levels(df_g$room_type_f)),
          "../data/factor_levels.rds")
  message("Model cache written.")
})

# load model data
ghost_data <- readRDS("../data/ghost_model_data.rds")
fl <- readRDS("../data/factor_levels.rds")
boroughs <- fl$boroughs
room_types <- fl$room_types

# load MLP predictions joined with listing metadata
mlp_preds <- read_csv("../data/mlp_ghost_predictions.csv",
                      col_types = cols(id = col_character()),
                      show_col_types = FALSE)
listings <- read_csv("../data/airbnb_clean.csv",
                     col_types = cols(id = col_character()),
                     show_col_types = FALSE) |>
  select(-1)

df_ghost <- listings |>
  mutate(price_num = as.numeric(gsub("[$,]", "", price))) |>
  inner_join(mlp_preds, by = "id") |>
  rename(ghost_prob = ghost_prob_mlp)

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
body { font-family: 'Georgia', serif; background: #f4f4f4; color: #1a1a1a; margin: 0; padding: 0; }
.navbar { background-color: #1a1a2e !important; border: none; border-radius: 0; margin-bottom: 0; }
.navbar-brand, .navbar-nav > li > a { color: #e8e8e8 !important; font-family: 'Georgia', serif; font-size: 14px; letter-spacing: 0.03em; }
.navbar-nav > li > a:hover, .navbar-nav > .active > a,
.navbar-nav > .active > a:focus, .navbar-nav > .active > a:hover {
  background-color: #2c2c54 !important; color: #ffffff !important; }
.tab-content { overflow-y: auto; min-height: 0; background: #f4f4f4; }
.tab-pane { padding: 20px 24px 40px 24px; }
.stat-box { background: #fff; border-left: 4px solid #1a1a2e; padding: 14px 18px; margin-bottom: 12px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
.stat-box .stat-label { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.09em; color: #777; margin: 0 0 4px 0; font-family: 'Helvetica Neue', Arial, sans-serif; }
.stat-box .stat-value { font-size: 1.7em; font-weight: bold; margin: 0; color: #1a1a2e; line-height: 1.2; }
.section-title { font-size: 0.75em; font-weight: bold; text-transform: uppercase; letter-spacing: 0.12em; color: #666; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin: 18px 0 10px 0; font-family: 'Helvetica Neue', Arial, sans-serif; }
.well { background-color: #fff; border: 1px solid #e0e0e0; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); padding: 16px; }
.form-control, .selectize-input { border-radius: 3px !important; border: 1px solid #ccc !important; font-size: 13px !important; }
label { font-size: 12px; font-weight: 600; color: #444; letter-spacing: 0.02em; font-family: 'Helvetica Neue', Arial, sans-serif; }
.irs--shiny .irs-bar, .irs--shiny .irs-single { background: #1a1a2e !important; border-color: #1a1a2e !important; }
.irs--shiny .irs-handle { border-color: #1a1a2e !important; }
.btn-run { width: 100%; background-color: #1a1a2e; color: white; border: none; padding: 9px 0; border-radius: 4px; font-size: 13px; font-family: 'Helvetica Neue', Arial, sans-serif; letter-spacing: 0.05em; cursor: pointer; transition: background 0.2s; }
.btn-run:hover { background-color: #2c2c54; color: white; }
.plot-panel { background: #fff; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 16px 18px 10px 18px; margin-bottom: 16px; }
.overview-section { max-width: 820px; margin: 0 auto; padding: 10px 0 30px 0; line-height: 1.8; }
.overview-section h4 { color: #1a1a2e; border-bottom: 2px solid #1a1a2e; padding-bottom: 5px; margin-top: 24px; font-size: 1.05em; letter-spacing: 0.02em; }
.overview-section p, .overview-section li { font-size: 0.95em; color: #333; }
.dataTables_wrapper { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 13px; }
"

# ui
ui <- navbarPage(
  title = "NYC Airbnb — Ghost Listing Detection",
  windowTitle = "NYC Airbnb Ghost Listings",
  header = tags$head(tags$style(HTML(app_css))),

  tabPanel("Overview",
    fluidRow(column(10, offset = 1,
      div(class = "overview-section",
        h4("What Are Ghost Listings?"),
        p("A ", tags$b("ghost listing"),
          " is an Airbnb property that appears active on the platform but shows little to
          no genuine guest activity. Listings are flagged using a composite scoring heuristic
          based on the following signals:"),
        tags$ul(
          tags$li("High availability (≥ 180 days/year)"),
          tags$li("High minimum night requirement (≥ 180 nights)"),
          tags$li("Long time since last review (≥ 180 days)"),
          tags$li("Price in the bottom 5th percentile for their neighbourhood and room type"),
          tags$li("Unverified host with no profile picture")
        ),
        p("Listings scoring above 2 of these 5 signals are labeled as ghost listings.
          Ghost listings distort platform supply signals, may manipulate neighborhood
          pricing benchmarks, and can represent fraudulent or inactive inventory."),
        h4("How the Models Work"),
        p("Two models are used in this application:"),
        tags$ul(
          tags$li(tags$b("Ghost Model (Bayesian Logistic): "),
            "Predicts ghost probability from borough, room type, host listing count,
            review score, and review count. Uses weakly informative priors —
            normal(0, 1) on coefficients and normal(0, 2.5) on the intercept.
            Powers the Ghost Detection tab."),
          tags$li(tags$b("Ghost Classifier (MLP): "),
            "A four-layer neural network (128 → 64 → 32 → 1 nodes) with ReLU activations,
            30% dropout, and cosine learning rate decay. Trained on 14,990 listings.
            Achieves 82.5% accuracy, 73.9% recall, 58.1% precision, and 0.876 AUC.
            Powers the Flagged Listings tab.")
        ),
        h4("Data Source"),
        p("Data sourced from ",
          tags$a("Inside Airbnb", href = "http://insideairbnb.com", target = "_blank"),
          ", a public repository of scraped NYC Airbnb listings. The raw dataset contains
          36,353 New York City listings from November 2025. After filtering for complete
          feature data, 21,415 listings were used for modeling, of which approximately
          5% are flagged as ghost listings."),
        p(tags$em("Note: data reflects a static snapshot and does not update in real-time."))
      )
    ))
  ),

  tabPanel("Ghost Detection",
    sidebarLayout(
      sidebarPanel(width = 3,
        div(class = "section-title", "Listing Parameters"),
        selectInput("ghost_borough", "Borough", choices = boroughs, selected = "Manhattan"),
        selectInput("room_type_ghost", "Room Type", choices = room_types, selected = "Entire home/apt"),
        sliderInput("ghost_accommodates", "Guests", min = 1, max = 16, value = 2, step = 1),
        sliderInput("host_listings", "Host Listing Count", min = 1, max = 100, value = 1, step = 1),
        sliderInput("ghost_review_score", "Avg Review Score", min = 1, max = 5, value = 4.5, step = 0.1),
        sliderInput("ghost_n_reviews", "Number of Reviews", min = 0, max = 500, value = 10, step = 5),
        br(),
        actionButton("predict_ghost", "Run Prediction", class = "btn-run")
      ),
      mainPanel(width = 9,
        uiOutput("ghost_prob_card"),
        div(class = "plot-panel",
          div(class = "section-title", "95% Credible Intervals — Ghost Predictors"),
          plotOutput("ghost_coef_plot", height = "300px")
        )
      )
    )
  ),

  tabPanel("MLP Performance",
    br(),
    fluidRow(column(10, offset = 1,
      div(class = "plot-panel",
        div(class = "section-title", "Training Curve, ROC, Precision-Recall & Score Distribution"),
        img(src = "mlp_diagnostics.png", width = "100%", style = "border-radius:4px;")
      ),
      div(class = "plot-panel",
        div(class = "section-title", "Confusion Matrix (threshold = 0.747)"),
        img(src = "mlp_confusion_matrix.png", width = "40%",
            style = "display:block; margin:0 auto; border-radius:4px;")
      )
    ))
  ),

  tabPanel("Flagged Listings",
    br(),
    fluidRow(
      column(3,
        div(class = "well",
          div(class = "section-title", "Filters"),
          selectInput("tbl_borough", "Borough",
                      choices = c("All", boroughs), selected = "All"),
          sliderInput("tbl_prob_thresh", "Min MLP Ghost Probability",
                      min = 0, max = 1, value = 0.5, step = 0.05),
          br(),
          uiOutput("flag_count_card"),
          br(),
          downloadButton("dl_flagged", "Download CSV",
                         style = "background-color:#1a1a2e; color:white; border:none;
                                  border-radius:4px; padding:7px 14px; font-size:13px;")
        )
      ),
      column(9, DTOutput("ghost_table"))
    )
  )
)

# server
server <- function(input, output, session) {

  ghost_prob_val <- eventReactive(input$predict_ghost, {
    x <- make_ghost_x(
      borough = input$ghost_borough,
      room_type = input$room_type_ghost,
      accommodates = input$ghost_accommodates,
      log_host_count = log1p(input$host_listings),
      review_complete = input$ghost_review_score,
      log_reviews = log1p(input$ghost_n_reviews)
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
      p(style = "margin:0; font-size:0.72em; text-transform:uppercase;
                 letter-spacing:0.09em; color:#777; font-family:'Helvetica Neue',Arial,sans-serif;",
        "Posterior Ghost Probability"),
      p(style = paste0("margin:6px 0 4px 0; font-size:2.2em; font-weight:bold; color:",
                       border_col, "; line-height:1.1;"), pct),
      p(style = "margin:0; font-size:0.88em; color:#555;
                 font-family:'Helvetica Neue',Arial,sans-serif;", lbl)
    )
  })

  output$ghost_coef_plot <- renderPlot({
    tryCatch({
      as.data.frame(ghost_data$ci) %>%
        rownames_to_column("param") %>%
        filter(param != "(Intercept)") %>%
        ggplot(aes(y = reorder(param, `97.5%`))) +
        geom_segment(aes(x = `2.5%`, xend = `97.5%`, yend = param),
                     linewidth = 1.2, color = "#1a1a2e") +
        geom_point(aes(x = (`2.5%` + `97.5%`) / 2), size = 2.5, color = "#1a1a2e") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
        labs(x = "Effect on log-odds of ghost", y = NULL) +
        theme_minimal(base_size = 11) +
        theme(panel.grid.minor = element_blank(),
              panel.grid.major.y = element_blank(),
              panel.grid.major.x = element_line(color = "#eeeeee"),
              axis.text.y = element_text(size = 10))
    }, error = function(e) {
      ggplot() + annotate("text", x = 0.5, y = 0.5,
        label = paste("Plot unavailable:", conditionMessage(e)),
        size = 4, color = "#c0392b") + theme_void()
    })
  }, res = 96)

  flagged_data <- reactive({
    d <- df_ghost %>%
      filter(ghost_prob > input$tbl_prob_thresh) %>%
      select(
        id, name,
        borough = neighbourhood_group_cleansed,
        neighbourhood = neighbourhood_cleansed,
        room_type, price_num, availability_365,
        reviews_per_month, number_of_reviews, ghost_prob
      ) %>%
      arrange(desc(ghost_prob))
    if (input$tbl_borough != "All")
      d <- filter(d, borough == input$tbl_borough)
    d
  })

  output$flag_count_card <- renderUI({
    n <- nrow(flagged_data())
    div(class = "stat-box",
      p(class = "stat-label", "Flagged Listings"),
      p(class = "stat-value", n),
      p(style = "font-size:0.78em; color:#888; margin:4px 0 0 0;
                 font-family:'Helvetica Neue',Arial,sans-serif;",
        paste0("above ", scales::percent(input$tbl_prob_thresh, accuracy = 1), " threshold"))
    )
  })

  output$ghost_table <- renderDT({
    flagged_data() %>%
      mutate(
        ghost_prob = scales::percent(ghost_prob, accuracy = 0.1),
        price_num = scales::dollar(price_num)
      ) %>%
      rename(
        "ID" = id,
        "Name" = name,
        "Borough" = borough,
        "Neighbourhood" = neighbourhood,
        "Room Type" = room_type,
        "Price" = price_num,
        "Availability" = availability_365,
        "Reviews/mo" = reviews_per_month,
        "Total Reviews" = number_of_reviews,
        "MLP Ghost Prob" = ghost_prob
      )
  },
  options = list(pageLength = 15, scrollX = TRUE, dom = "frtip", autoWidth = TRUE),
  rownames = FALSE,
  filter = "top"
  )

  output$dl_flagged <- downloadHandler(
    filename = function() paste0("ghost_listings_", Sys.Date(), ".csv"),
    content = function(file) write_csv(flagged_data(), file)
  )
}

shinyApp(ui, server)
