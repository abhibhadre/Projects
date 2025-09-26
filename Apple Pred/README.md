# RNN-LSTM Time Series Forecasting

This repository contains an implementation of a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) units for time series forecasting on Apple stock data.

## Repository Structure

- **DL_RNN_LSTM_Time_Series.ipynb**: Jupyter Notebook containing the complete implementation of the model, including preprocessing, training, and evaluation.
- **apple_training.csv**: Training dataset (Apple stock data).
- **apple_testing.csv**: Testing dataset (Apple stock data).

## Requirements

The dependencies are listed in `requirements.txt`. You can install them with:

```bash
pip install -r requirements.txt
```

## How to Run

1. Clone this repository:
   ```bash
   git clone <https://github.com/abhadre66/Apple_Prediction_Using_LSTM.git>
   cd <abhadre66>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open the Jupyter Notebook:
   ```bash
   jupyter notebook DL_RNN_LSTM_Time_Series.ipynb
   ```

4. Run all cells to train and evaluate the LSTM model.

## Dataset

- The project uses Apple stock datasets:
  - **apple_training.csv**: Used for model training.
  - **apple_testing.csv**: Used for model evaluation.

## Model Overview

- Uses **Recurrent Neural Network (RNN)** with **LSTM layers**.
- Forecasts stock price trends based on historical Apple stock data.
- Includes preprocessing steps such as normalization and sequence generation.

## Results

- The notebook demonstrates training and evaluation results, including plots comparing predicted vs. actual values.

---

**Author:** *Your Name Here*
