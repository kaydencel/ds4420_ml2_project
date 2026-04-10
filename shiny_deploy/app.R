library(shiny)
library(tidyverse)
library(DT)

# ── Load pre-computed data ────────────────────────────────────────────────────
price_data      <- readRDS("price_model_data.rds")
ghost_data      <- readRDS("ghost_model_data.rds")
df_ghost        <- readRDS("df_ghost_data.rds")
borough_medians <- readRDS("borough_medians.rds")
fl              <- readRDS("factor_levels.rds")

boroughs   <- fl$boroughs
room_types <- fl$room_types

# ── Prediction helpers ────────────────────────────────────────────────────────

# Build design vector for price model (treatment contrasts; reference = first level).
make_price_x <- function(med_neigh, room_type, accommodates, log_reviews, review_score) {
  cn <- setdiff(colnames(price_data$draws), "sigma")
  x  <- setNames(numeric(length(cn)), cn)
  x["(Intercept)"]        <- 1
  x["neigh_median_price"] <- med_neigh
  rt_col <- paste0("room_type", room_type)
  if (rt_col %in% cn) x[rt_col] <- 1
  x["accommodates"]    <- accommodates
  x["log_reviews"]     <- log_reviews
  x["review_complete"] <- review_score
  x
}

# Build design vector for ghost (logistic) model.
make_ghost_x <- function(borough, room_type, accommodates, log_host_count,
                          price_vs_neigh, review_complete, log_reviews) {
  cn <- colnames(ghost_data$draws)
  x  <- setNames(numeric(length(cn)), cn)
  x["(Intercept)"]    <- 1
  b_col <- paste0("borough", borough)
  if (b_col %in% cn) x[b_col] <- 1
  rt_col <- paste0("room_type", room_type)
  if (rt_col %in% cn) x[rt_col] <- 1
  x["accommodates"]    <- accommodates
  x["log_host_count"]  <- log_host_count
  x["price_vs_neigh"]  <- price_vs_neigh
  x["review_complete"] <- review_complete
  x["log_reviews"]     <- log_reviews
  x
}

# Posterior predictive draws: η = Xβ + N(0,σ), back-transform to dollars.
sample_price <- function(x) {
  coef_cn <- setdiff(colnames(price_data$draws), "sigma")
  eta     <- as.numeric(price_data$draws[, coef_cn] %*% x)
  sigma   <- if ("sigma" %in% colnames(price_data$draws)) price_data$draws[, "sigma"] else 0
  exp(rnorm(length(eta), eta, sigma))
}

# Posterior mean ghost probability via logistic link.
predict_ghost <- function(x) {
  mean(plogis(as.numeric(ghost_data$draws %*% x)))
}

# ── Shared CSS ────────────────────────────────────────────────────────────────
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

# ── UI ────────────────────────────────────────────────────────────────────────
ui <- navbarPage(
  title       = "NYC Airbnb — Bayesian Analysis",
  windowTitle = "NYC Airbnb Bayesian",
  header      = tags$head(tags$style(HTML(app_css))),

  # ── Overview ─────────────────────────────────────────────────────────────────
  tabPanel("Overview",
    fluidRow(column(10, offset = 1,
      div(class = "overview-section",
        h4("Dynamic Pricing & Personalized Pricing"),
        p("Dynamic pricing adjusts prices in response to market conditions — supply and demand
          shifts, competitor pricing, and inventory levels. Originally developed in the airline
          industry in the 1980s–90s, it has expanded dramatically through big data and machine
          learning."),
        p("Today, companies use personal data — browsing behavior, device info, demographics,
          even battery level — to infer each consumer's willingness to pay. This practice,
          sometimes called ", tags$b("surveillance pricing"), " or ", tags$b("predatory pricing"),
          ", raises serious concerns around consumer protection and data privacy. Maryland's
          Predatory Pricing Act (2026) and the EU Omnibus Directive (2019/2161) represent early
          regulatory steps."),

        h4("What Are Ghost Listings?"),
        p("A ", tags$b("ghost listing"),
          " is an Airbnb property that appears active but shows little to no genuine guest
          activity. Listings are flagged using the following heuristic:"),
        tags$ul(
          tags$li("Availability > 300 days/year AND fewer than 0.1 reviews per month, OR"),
          tags$li("Zero reviews per month AND the host manages more than 3 listings")
        ),
        p("Ghost listings distort platform supply signals, may manipulate neighborhood pricing
          benchmarks, and can represent fraudulent or inactive inventory."),

        h4("How the Models Work"),
        p("Two Bayesian regression models are fit using ", tags$code("rstanarm"),
          " (4 MCMC chains, 2,000 iterations each):"),
        tags$ul(
          tags$li(tags$b("Price Model (Gaussian): "),
            "Predicts log(nightly price) from neighborhood median price, room type, guest
            capacity, review count, and composite review score."),
          tags$li(tags$b("Ghost Model (Logistic): "),
            "Predicts ghost probability from borough, room type, host listing count, price
            deviation from neighborhood median, review score, and review count.")
        ),
        p("Both models use weakly informative priors — ", tags$code("normal(0, 0.5)"),
          " on all coefficients — to regularize estimates without overwhelming the data."),

        h4("Data Source"),
        p("Data sourced from ",
          tags$a("Inside Airbnb", href = "http://insideairbnb.com", target = "_blank"),
          ", a public repository of scraped NYC Airbnb listings. Filtered to $10–$2,000/night."),
        p(tags$em("Note: data reflects a static snapshot and does not update in real-time."))
      )
    ))
  ),

  # ── Price Prediction ──────────────────────────────────────────────────────────
  tabPanel("Price Prediction",
    sidebarLayout(
      sidebarPanel(width = 3,
        div(class = "section-title", "Listing Parameters"),
        selectInput("borough",         "Borough",    choices = boroughs,   selected = "Manhattan"),
        selectInput("room_type_price", "Room Type",  choices = room_types, selected = "Entire home/apt"),
        sliderInput("accommodates",    "Guests",     min = 1,   max = 16,  value = 2,   step = 1),
        sliderInput("n_reviews",       "Reviews",    min = 0,   max = 500, value = 30,  step = 5),
        sliderInput("review_score",    "Avg Score",  min = 1,   max = 5,   value = 4.8, step = 0.1),
        br(),
        actionButton("predict_price", "Run Prediction", class = "btn-run")
      ),
      mainPanel(width = 9,
        fluidRow(
          column(4, div(class = "stat-box",
            p(class = "stat-label", "Posterior Median"),
            p(class = "stat-value", textOutput("med_price"))
          )),
          column(4, div(class = "stat-box",
            p(class = "stat-label", "90% CI — Lower"),
            p(class = "stat-value", textOutput("lo_price"))
          )),
          column(4, div(class = "stat-box",
            p(class = "stat-label", "90% CI — Upper"),
            p(class = "stat-value", textOutput("hi_price"))
          ))
        ),
        div(class = "plot-panel",
          div(class = "section-title", "Posterior Predictive Check (vs. Training Data)"),
          img(src = "ppc_plot.png", width = "100%",
              style = "display:block; border-radius:4px;")
        ),
        div(class = "plot-panel",
          div(class = "section-title", "95% Credible Intervals — Price Predictors"),
          plotOutput("price_coef_plot", height = "220px")
        )
      )
    )
  ),

  # ── Ghost Detection ───────────────────────────────────────────────────────────
  tabPanel("Ghost Detection",
    sidebarLayout(
      sidebarPanel(width = 3,
        div(class = "section-title", "Listing Parameters"),
        selectInput("ghost_borough",      "Borough",            choices = boroughs,   selected = "Manhattan"),
        selectInput("room_type_ghost",    "Room Type",          choices = room_types, selected = "Entire home/apt"),
        sliderInput("ghost_accommodates", "Guests",             min = 1,   max = 16,  value = 2,   step = 1),
        sliderInput("host_listings",      "Host Listing Count", min = 1,   max = 100, value = 1,   step = 1),
        sliderInput("price_vs_neigh_val", "Price vs. Neighborhood (log scale)", min = -2, max = 2, value = 0, step = 0.1),
        sliderInput("ghost_review_score", "Avg Review Score",   min = 1,   max = 5,   value = 4.5, step = 0.1),
        sliderInput("ghost_n_reviews",    "Number of Reviews",  min = 0,   max = 500, value = 10,  step = 5),
        br(),
        actionButton("predict_ghost", "Run Prediction", class = "btn-run")
      ),
      mainPanel(width = 9,
        uiOutput("ghost_prob_card"),
        div(class = "plot-panel",
          div(class = "section-title", "95% Credible Intervals — Ghost Predictors"),
          plotOutput("ghost_coef_plot", height = "260px")
        )
      )
    )
  ),

  # ── Flagged Listings ──────────────────────────────────────────────────────────
  tabPanel("Flagged Listings",
    br(),
    fluidRow(
      column(3,
        div(class = "well",
          div(class = "section-title", "Filters"),
          selectInput("tbl_borough",     "Borough",
                      choices = c("All", boroughs), selected = "All"),
          sliderInput("tbl_prob_thresh", "Min Ghost Probability",
                      min = 0, max = 1, value = 0.15, step = 0.05),
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
  ),

)

# ── Server ────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {

  # ── Price prediction ──────────────────────────────────────────────────────
  # ignoreNULL = FALSE so results display on load before the button is clicked
  price_draws <- eventReactive(input$predict_price, {
    med_neigh <- borough_medians$med[borough_medians$borough == input$borough]
    x <- make_price_x(
      med_neigh    = med_neigh,
      room_type    = input$room_type_price,
      accommodates = input$accommodates,
      log_reviews  = log1p(input$n_reviews),
      review_score = input$review_score
    )
    sample_price(x)
  }, ignoreNULL = FALSE)

  output$med_price <- renderText({ paste0("$", round(median(price_draws()), 0)) })
  output$lo_price  <- renderText({ paste0("$", round(quantile(price_draws(), 0.05), 0)) })
  output$hi_price  <- renderText({ paste0("$", round(quantile(price_draws(), 0.95), 0)) })

  output$price_coef_plot <- renderPlot({
    tryCatch({
      as.data.frame(price_data$ci) %>%
        rownames_to_column("param") %>%
        filter(!param %in% c("(Intercept)", "sigma")) %>%
        ggplot(aes(y = reorder(param, `97.5%`))) +
        geom_segment(aes(x = `2.5%`, xend = `97.5%`, yend = param),
                     linewidth = 1.2, color = "#1a1a2e") +
        geom_point(aes(x = (`2.5%` + `97.5%`) / 2), size = 2.5, color = "#1a1a2e") +
        geom_vline(xintercept = 0, linetype = "dashed", color = "#c0392b", linewidth = 0.7) +
        labs(x = "Effect on log(price)", y = NULL) +
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

  # ── Ghost prediction ──────────────────────────────────────────────────────
  ghost_prob_val <- eventReactive(input$predict_ghost, {
    x <- make_ghost_x(
      borough         = input$ghost_borough,
      room_type       = input$room_type_ghost,
      accommodates    = input$ghost_accommodates,
      log_host_count  = log1p(input$host_listings),
      price_vs_neigh  = input$price_vs_neigh_val,
      review_complete = input$ghost_review_score,
      log_reviews     = log1p(input$ghost_n_reviews)
    )
    predict_ghost(x)
  }, ignoreNULL = FALSE)

  # Color-coded card: green < 10%, orange 10–30%, red > 30%
  output$ghost_prob_card <- renderUI({
    p_val      <- ghost_prob_val()
    pct        <- scales::percent(p_val, accuracy = 0.1)
    border_col <- if (p_val < 0.10) "#27ae60" else if (p_val < 0.30) "#e67e22" else "#c0392b"
    lbl        <- if      (p_val < 0.10) "Low risk — listing appears active"
                  else if (p_val < 0.30) "Moderate risk — worth investigating"
                  else                   "High risk — likely ghost listing"
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

  # ── Flagged listings table ────────────────────────────────────────────────
  flagged_data <- reactive({
    d <- df_ghost %>%
      filter(ghost_prob > input$tbl_prob_thresh) %>%
      select(
        id, name,
        borough       = neighbourhood_group_cleansed,
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
        price_num  = scales::dollar(price_num)
      ) %>%
      rename(
        "ID"            = id,
        "Name"          = name,
        "Borough"       = borough,
        "Neighbourhood" = neighbourhood,
        "Room Type"     = room_type,
        "Price"         = price_num,
        "Availability"  = availability_365,
        "Reviews/mo"    = reviews_per_month,
        "Total Reviews" = number_of_reviews,
        "Ghost Prob"    = ghost_prob
      )
  },
  options  = list(pageLength = 15, scrollX = TRUE, dom = "frtip", autoWidth = TRUE),
  rownames = FALSE,
  filter   = "top"
  )

  output$dl_flagged <- downloadHandler(
    filename = function() paste0("ghost_listings_", Sys.Date(), ".csv"),
    content  = function(file) write_csv(flagged_data(), file)
  )

}

shinylive::export(appdir = ".", destdir = "docs")
shinyApp(ui, server)

