# hp_use_case

This repo contains the code for the HP kaggle inventory management competition.  

[HP Supply Chain Optimization](https://www.kaggle.com/competitions/hp-supply-chain-optimization/overview)

- **1_data_exploration**: contains the exploratory/statistical analysis of the dataset
- **2_model_naive**: baseline of naive lagged and naive rolling means models
- **2_model_naive_bottom_up**: baseline of naive lagged and naive rolling means models with a bottom-up approach
- **2_model_lightGBM**: ligthGBM model directly forecasting the target time series
- **2_model_lightGBM_bottom_up**: ligthGBM model forecasting the target time series with a bottom-up approach
- **2_model_lightGBM_lags**: ligthGBM with recent lagged values