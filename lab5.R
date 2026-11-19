#Set-up
library(tidyverse)
library(dplyr)
library(ggplot2)
library(glmmTMB)
library(DHARMa)
library(ggeffects)
library(emmeans)
library(broom.mixed)
library(gt)
library(AER)

# Data loading
data(NMES1988)

# Assign reference groups of average (health), male (gender), and no (insurance)
data <- NMES1988 |>
  select(visits, chronic, health, gender, school, insurance) |> 
  mutate(health = fct_relevel(factor(health), "average"),
         gender = fct_relevel(factor(gender), "male"),
         insurance = fct_relevel(factor(insurance), "no"))

# Exploring the data
head(data) # Looking at first few rows

summary.table <- summary(data) # Basic descriptives for all variables

visits_summary <- data |> # Exploring descriptive stats for outcome
  summarise(n = n(),
    mean = mean(visits),
    var = var(visits),
    p_zero = mean(visits == 0), # Proportion of zero values
    max = max(visits))

# Creating a plot of the overall distribution of the outcome variable (number of visits)
descriptive_plot <- ggplot(data, aes(x = visits)) +
  geom_bar() +
  labs(x = "Physician's Office Visits", y = "Number of People")

# Fitting a Gaussian model to see why a linear model doesn't work for counts

m_gauss <- glmmTMB(visits ~ chronic + health + gender + school + insurance,
  data = data,
  family = gaussian(link = "identity"))

#simulation 
set.seed(9041)
res_gauss <- simulateResiduals(m_gauss, n = 500, plot = FALSE)
plot(res_gauss)

### Specifying competing count models
# Null (intercept-only) Poisson model: no predictors, just the overall average.
null <- glmmTMB(
  visits ~ 1, 
  data = data, 
  family = poisson(link = "log"))

# One-predictor Poisson model: does the number of chronic conditions predict physician visits?
one_pred <- glmmTMB(
  visits ~ chronic, 
  data = data, 
  family = poisson(link = "log"))

# Full Poisson model: all five predictors at once.
poisson <- glmmTMB(
  visits ~ chronic + health + gender + school + insurance,
  data = data,
  family = poisson(link = "log"))

# Looking at model summaries for model fit and coefficient estimates
summary(null)
summary(one_pred)
summary(poisson)

tests <- anova(null, one_pred, poisson) # Comparing models

tests <- tests |>
  select(Chisq, `Chi Df`, `Pr(>Chisq)`) |> # Selecting relevant stats from model comparison
  rename(Nested.df = `Chi Df`,
    Nested.Chisq.stat = Chisq,
    Nested.p.value = `Pr(>Chisq)`)

glance0 <- broom.mixed::glance(null) # More model-level statistics
glance1 <- broom.mixed::glance(one_pred)
glance2 <- broom.mixed::glance(poisson)

mods <- c("Null",
  "One_Pred: Chronic Conditions",
  "Full") # Naming models

glances <- rbind(glance0,
  glance1,
  glance2) # Combining individual glances into one object

comparison_table <- cbind(Model = mods,
  glances,
  tests) # Combining anova and glance results into one object

comparison_table |> #Formatting combined results into gt table
  select(Model,
    AIC,
    BIC,
    logLik,
    deviance,
    df.residual,
    Nested.df,
    Nested.Chisq.stat,
    Nested.p.value) |>
  gt() |>
  cols_label(Model = "Model",
    AIC = "AIC",
    BIC = "BIC",
    logLik = "Log-likelihood",
    deviance = "Deviance",
    df.residual = "Residual.df",
    Nested.df = "Nested df",
    Nested.Chisq.stat = "Nested χ²",
    Nested.p.value = "Nested p") |>
  fmt_number(columns = c(AIC,
      BIC,
      logLik,
      deviance,
      Nested.Chisq.stat),
    decimals = 2) |>
  fmt(columns = Nested.p.value,
    fns = function(x) {
      ifelse(
        is.na(x),
        "—",
        ifelse(x < .001, "< .001", sprintf("%.3f", x)))}) |>
  sub_missing(columns = c(Nested.df, Nested.Chisq.stat),
    missing_text = "—")

# Evaluating the Poisson Model
selected_row <- comparison_table |>
  filter(Nested.p.value < .05) |>
  slice_tail(n = 1)

# Creating an object with our selected model's name for dynamic reporting
selected_name <- selected_row$Model

# Case matching for dynamic reporting
selected_model <- switch(
  selected_name,
  "Null" = null,
  "Predictor: Chronic" = one_pred,
  "Full" = poisson)

# Setting a seed so the simulated residuals are reproducible
set.seed(9041)

# Simulating 500 new datasets from the fitted Poisson model.
res_pois <- simulateResiduals(poisson, n = 500, plot = FALSE)

# Two diagnostic panels: QQ plot (points should follow the diagonal)
# and residuals vs. predicted values (should be flat)
plot(res_pois)

# Formal dispersion test: is the observed variance bigger (or smaller) than the Poisson model expects?
# A significant p means the variance = mean assumption doesn't hold
disp_pois <- testDispersion(res_pois)

# Zero-inflation test: does the data have more zeros than the Poisson predicts?
zi_pois <- testZeroInflation(res_pois)

# Negative binomial (NB2) model
neg_binom <- glmmTMB(visits ~ chronic + health + gender + school + insurance,
  data   = data,
  family = nbinom2(link = "log"))

summary(neg_binom)

# LRT: does the NB fit better than the Poisson?
lrt_nb<-anova(poisson, neg_binom)

# Same seed as before so the residuals are reproducible and comparable to the Poisson check.
set.seed(9041)

# If the NB is adequate, these should now be uniform, unlike the Poisson's.
res_nb <- simulateResiduals(neg_binom, n = 500, plot = FALSE)

# QQ plot: points should now sit close to the diagonal.
# Residuals vs. predicted: should be roughly flat.
plot(res_nb)

# Dispersion: the ratio should now be close to 1 and non-significant.
disp_nb <- testDispersion(res_nb)

# Zero inflation: does the NB reproduce the observed number of zeros?
# A non-significant result means the NB's extra variance already accounts for the zeros, so there's no need for a zero-inflated or hurdle model.
zi_nb <- testZeroInflation(res_nb)

# Predicted mean article counts from the additive NB, on the count scale.
# "chronic [0:25]" plots mentor articles from 0 to 25
pred_nb <- predict_response(neg_binom, terms = "chronic [0:25]")

plot(pred_nb) +
  labs(title = "Predicted Number of Physician's Office Visits by Number of Chronic Conditions",
    x     = "Number of Chronic Conditions", y = "Visits")

# Zero-inflated NB with a constant zero-inflation probability
m_zinb <- glmmTMB(visits ~ chronic + health + gender + school + insurance,
  ziformula = ~ 1,
  data = data, family = nbinom2())

# Zero-inflated NB with predictors in the zero part
m_zinb_full <- glmmTMB(visits ~ chronic + health + gender + school + insurance,
  ziformula = ~ chronic + health + gender + school + insurance,
  data = data, family = nbinom2())

# Hurdle NB
m_hurd <- glmmTMB(visits ~ chronic + health + gender + school + insurance,
  ziformula = ~ chronic + health + gender + school + insurance,
  data = data, family = truncated_nbinom2())

# Compare models
lrt_nb_zi_full <- anova(neg_binom, m_zinb_full)

# The hurdle isn't nested in the NB, so compare with AIC/BIC (lower = better)
AIC(neg_binom, m_zinb, m_zinb_full, m_hurd)
BIC(neg_binom, m_zinb, m_zinb_full, m_hurd)

# Then re-check the chosen model with DHARMa
set.seed(9041)
res_hurd_nb <- simulateResiduals(m_hurd, n = 500, plot = FALSE)
plot(res_hurd_nb)

testZeroInflation(res_hurd_nb)

# Hurdle NB model results
hurdle_results <- summary(m_hurd)

# Giving conditional results their own data frame and adding 95% CIs
hurdle_cond_results <- as.data.frame(hurdle_results$coefficients$cond) |> 
  mutate(CI_low = Estimate - 1.96 * `Std. Error`,
         CI_high = Estimate + 1.96 * `Std. Error`)

# Giving zero results their own data frame and adding 95% CIs
hurdle_zi_results <- as.data.frame(hurdle_results$coefficients$zi) |>
  mutate(CI_low = Estimate - 1.96 * `Std. Error`,
         CI_high = Estimate + 1.96 * `Std. Error`)

# NB model with a chronic condition x insurance interaction.
# The other predictors are controls.
interact <- glmmTMB(visits ~ health + gender + school + chronic * insurance,
                    ziformula = ~ chronic + health + gender + school + insurance,
                    data = data, family = truncated_nbinom2())

interaction_results <- as.data.frame(summary(interact)$coefficients$cond)

# LRT: does adding the interaction improve fit over the additive NB?
lrt_int <- anova(m_hurd, interact)
lrt_int

# Predicted mean number of visits from the interaction model, on the count scale.
pred_int <- predict_response(interact, terms = c("chronic", "insurance"), type = "count")

plot_interaction <- ggplot(pred_int, aes(x = x, y = predicted,
                                        colour = group, fill = group)) +
  geom_line(linewidth = 1) +
  geom_ribbon(aes(ymin = conf.low, ymax = conf.high),
              alpha = .2, color = NA) +
  scale_colour_manual(values = c(no = "coral", yes = "green4")) +
  scale_fill_manual(values = c(no = "coral", yes = "green4")) +
  labs(title = "Predicted Physician's Office Visits by Chronic Conditions and Insurance",
    x = "Number of Chronic Conditions",
    y = "Predicted Physician's Office Visits",
    colour = "Insurance",
    fill = "Insurance") +
  theme_minimal()

# Simple slopes: the slope of chronic estimated separately for those with and without health insurance.
slopes <- emtrends(interact, 
  specs = ~ insurance, 
  var = "chronic")

pairs_slopes <- as.data.frame(pairs(slopes))

#test simple slopes against zero
slopes_sum <- summary(slopes, infer = TRUE)

# Simple effects: flip the probing around and look at the insurance difference
# at specific values of chronic condition number, instead of the chronic slope within each insurance response type

# Pick the probing values: low (-1 SD), average (mean) and high (+1 SD) chronic condition number
chronic_mean <- mean(data$chronic, na.rm = TRUE)
chronic_sd <- sd(data$chronic, na.rm = TRUE)
chronic_values <- chronic_mean + c(-1, 0, 1) * chronic_sd

# Estimated marginal means (EMMs) for each insurance response type at each chronic condition number value.
insurance_emmeans <- emmeans(interact,
  specs = ~ insurance | chronic,
  at = list(chronic = chronic_values))

# Estimated mean visits in each insurance x chronic cell.
summary(insurance_emmeans, type = "response")

# Insurance differences at each chronic value, on the log scale.
pairs(insurance_emmeans)

# Build a data frame of the simple effects, then format it as a table.
contrast_df <- summary(pairs(insurance_emmeans)) |>
  as.data.frame() |>
  mutate(chronic = rep(chronic_values, each = 1),
         chronic_level = c("-1 SD", "Mean", "+1 SD")[match(round(chronic, 6), round(chronic_values, 6))]) |>
  select(chronic_level, contrast, estimate, SE, z.ratio, p.value)

contrast_df |>
  gt() |>
  cols_label(chronic_level = "Number of Chronic Conditions", contrast = "Contrast",
    estimate = "Log Contrast", SE = "SE", z.ratio = "z", p.value = "p") |>
  fmt_number(columns = c(estimate, SE, z.ratio), decimals = 2) |>
  fmt(columns = p.value,
      fns = \(x) scales::pvalue(x, accuracy = 0.001, add_p = FALSE)) |>
  tab_header(title = "Simple Effects of Insurance Across Number of Chronic Conditions",
    subtitle = "Hurdle Negative-Binomial Regression") |>
  tab_source_note("Log contrast = No − Yes; z-tests and p-values are computed on the log scale.")