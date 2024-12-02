from global_aggregation import aggregate_and_save_global_model

# Paths to local models
local_model_paths = [
    "saved_models/model_1.h5",
    "saved_models/model_2.h5",
    "saved_models/model_3.h5"
]

# Data proportions (based on training sample sizes)
data_proportions = [
    2670 / 6675,  # Model 1
    2670 / 6675,  # Model 2
    1335 / 6675   # Model 3
]

# Aggregate local models and save the global model
global_model = aggregate_and_save_global_model(
    local_model_paths=local_model_paths,
    data_proportions=data_proportions,
    save_path="global_model.h5"
)
