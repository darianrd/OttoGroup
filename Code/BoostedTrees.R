library(tidymodels)
library(vroom)
library(bonsai)
library(lightgbm)

# Read in data
train <- vroom("train.csv")
test <- vroom("test.csv")

# Create recipe
otto_recipe <- recipe(target ~ ., data = train) |> 
  step_rm(id) |> 
  step_mutate_at(all_nominal_predictors(), fn = "factor") |> 
  step_dummy(all_nominal_predictors())

# Create boosted model
boost_mod <- boost_tree(tree_depth = tune(),
                        trees = tune(),
                        learn_rate = tune()) |> 
  set_engine("lightgbm") |> 
  set_mode("classification")

# Create workflow
boost_wf <- workflow() |> 
  add_recipe(otto_recipe) |> 
  add_model(boost_mod)

# Grid of values to tune over
tuning <- grid_regular(tree_depth(),
                       trees(),
                       learn_rate(),
                       levels = 5)

# Split data for cross validation
folds <- vfold_cv(train, v = 5, repeats = 1)

# Run cross validation
cv_results <- boost_wf |> 
  tune_grid(resamples = folds,
            grid = tuning,
            metrics = metric_set(roc_auc, f_meas, accuracy))

# Find best tuning parameters
best_tuning <- cv_results |> 
  select_best(metric = "roc_auc")

# Finalize workflow
final_wf <- boost_wf |> 
  finalize_workflow(best_tuning) |> 
  fit(data = train)

# Make predictions
boost_preds <- final_wf |> 
  predict(new_data = test, type = "prob")

# Format for Kaggle submission
kaggle_sub <- boost_preds |> 
  bind_cols(test$id) |> 
  rename(id = ...10,
         Class_1 = .pred_Class_1,
         Class_2 = .pred_Class_2,
         Class_3 = .pred_Class_3,
         Class_4 = .pred_Class_4,
         Class_5 = .pred_Class_5,
         Class_6 = .pred_Class_6,
         Class_7 = .pred_Class_7,
         Class_8 = .pred_Class_8,
         Class_9 = .pred_Class_9) |> 
  select(id, everything())

# Write out file
vroom_write(x = kaggle_sub, file = "submission2.csv", delim = ",")