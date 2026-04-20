[![DOI](https://zenodo.org/badge/1216229123.svg)](https://doi.org/10.5281/zenodo.19669874)
# Intelligent daily rainfall prediction for early warning
### Using Deep Learning and Satellite Data (Ivory Coast)
## Overview
This repository provides the code and resources associated with the research article:
**“Intelligent daily rainfall prediction for early warning using deep learning and satellite data: Application to Bouaflé and Zuénoula stations, Ivory Coast.”**
The study develops a deep learning-based framework for predicting daily rainfall to support **flood early warning systems** in the Marahoué region (Ivory Coast).
## Authors
* **Satti J. R. Kamenan** (Corresponding author) – [sattijhon@gmail.com](mailto:sattijhon@gmail.com)
* Ta M. Youan (3-4)
* Miessan G. Adja (5)
* Sandona I. Soro (1)
* Amani M. Kouassi (2)
## Affiliations
1. Laboratory of Geographic Sciences, Civil Engineering and Geosciences, INP-HB, Yamoussoukro, Ivory Coast
2. School of Mines and Geology, INP-HB, Yamoussoukro, Ivory Coast
3. CURAT, Félix Houphouët-Boigny University, Abidjan, Ivory Coast
4. UFR-STRM, Félix Houphouët-Boigny University, Abidjan, Ivory Coast
5. ENS Abidjan, Department of Life and Earth Sciences, Ivory Coast
## Study Area
The study focuses on:
* **Bouaflé**
* **Zuénoula**
Located in the **Marahoué region**, a flood-prone area in central Ivory Coast.
## Objectives
* Develop a **rainfall prediction model** using deep learning
* Improve **early warning systems for floods**
* Compare deep learning with traditional machine learning models
## Methodology
### Model
* Long Short-Term Memory (**LSTM**) neural network
### Input Data
* Satellite-based precipitation products
* Atmospheric variables from reanalysis datasets
### Prediction Horizons
* t+1 day
* t+3 days
* t+7 days
### Benchmark Models
* Random Forest (RF)
* Extra Trees (ET)
* XGBoost (XGB)
## Evaluation Metrics
* Coefficient of determination (**R²**)
* Nash-Sutcliffe Efficiency (**NSE**)
* Pearson correlation coefficient (**R**)
* Normalized RMSE
* Mean Absolute Error (**MAE**)
## Key Results
* LSTM outperforms all benchmark models across all stations
* High performance at short and medium lead times:
  * R² > 0.90
  * NSE > 0.90
* Performance decreases at t+7 due to increased uncertainty
* LSTM remains more robust than tree-based models
## Repository Structure
├── data/              # Input datasets (or links to datasets)
├── models/            # Trained models
├── scripts/           # Data processing and training scripts
├── results/           # Outputs and evaluation results
├── requirements.txt   # Dependencies
└── README.md          # Project documentation
## Installation
bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
## Usage
Run the main script:
bash
python scripts/train_lstm.py
## Data Availability
Due to size or access restrictions, datasets may not be fully included.
Please refer to:
* Satellite data sources (e.g., GPM)
* Local hydrometeorological observations
## DOI
A DOI will be assigned via Zenodo upon release.
## icense
This project is licensed under the MIT License.
## Future Work
* Integration of additional atmospheric predictors
* Advanced hyperparameter optimization
* Improvement of long-term (t+7) predictions
## Acknowledgements
We acknowledge all institutions and collaborators involved in data provision and scientific support.
## Contact
**Satti J. R. Kamenan**
Email: [sattijhon@gmail.com](mailto:sattijhon@gmail.com)
