# Missing Spatial-Temporal Multimodal Traffic Data Imputation and Prediction

## 📌 Introduction
This repository focuses on the **imputation and prediction of missing multimodal traffic data** using Graph Neural Network (GNN) and Recurrent Neural Network (RNN). I leveraged the **Traffic4cast NeurIPS 2022 dataset** and implemented **GRIN (Graph Recurrent Imputation Network)** to reconstruct and predict traffic data efficiently. 

## 📊 Dataset: NeurIPS 2022 Traffic4cast
I used the **Traffic4cast dataset** provided by the **Institute of Advanced Research in Artificial Intelligence (IARAI)**. This dataset contains spatiotemporal traffic data from multiple cities, recorded at high resolution. 

- 📁 **Dataset Repository:** [NeurIPS2022-traffic4cast](https://github.com/iarai/NeurIPS2022-traffic4cast)
- 🏙 **Cities Covered:** London.
- ⏳ **Time Granularity:** 15-minute intervals.
- 🚘 **Traffic Data:** Flow, speed, and volume of vehicles.

## 🔍 Algorithm: Graph Recurrent Imputation Network (GRIN)
For missing data imputation, I used the **GRIN model** developed by the Graph Machine Learning Group. GRIN is designed to reconstruct missing traffic data using a **spatiotemporal graph-based approach**. Subsequently, I used the **Graph Attention Network (GAT)** to predict the **Congesion Class (CC)** and **Estimated time of Arrival(ETA)** based on the compete dataset.

- 📁 **Algorithm Repository:** [GRIN](https://github.com/Graph-Machine-Learning-Group/grin)
- 📌 **Key Features:**
  - Graph Neural Networks (GNNs) for spatial dependencies.
  - Recurrent Neural Networks (RNNs) for temporal patterns.
  - Imputation and prediction in an end-to-end pipeline.

## 🌍 Multimodal Geographic Data Integration
In addition to traffic data, I incorporated geographic information from OpenStreetMap (OSM) building data of London to enhance the model's ability to impute missing values:

- 📁 **Geographic Data Source:** [Geofabrik OpenStreetMap (OSM) Building Data](https://download.geofabrik.de/europe/united-kingdom/england.html)
- 🏢 **Features Extracted:**
  - **Building density**: Number of buildings per unit area.
  - **Building type diversity**: Residential, commercial, educational, etc.
  - **Proximity to infrastructure**: Influence of buildings on road traffic.

## 🚀 Installation
To set up the environment, install the required dependencies:

```bash
conda env create -f conda_env.yml
conda activate my_env  # Replace with your environment name
```

Or install via pip:

```bash
pip install -r requirements.txt
```

## 🛠 Setup and Running the Model
1. **Download the datasets:**
   - Traffic4cast data from [IARAI](https://github.com/iarai/NeurIPS2022-traffic4cast)
   - Building footprint data from [Geofabrik](https://download.geofabrik.de/europe/united-kingdom/england.html)

2. **Run the imputation model:**
   ```bash
   python run_imputation.py --config config/grin/traffic_block.yaml --dataset-name traffic_block --in-sample True 
   ```

4. **Submit jobs on an HPC cluster (SLURM):**
   ```bash
   sbatch slurmjob.sh
   ```

## 📜 Citation
If you use this repository, please consider citing:

```
@inproceedings{grin_t4c22,
  title={Missing Spatial-Temporal Multimodal Traffic Data Imputation and Prediction},
  author={Ankita Sarkar},
  booktitle={TU Dortmund},
  year={2025}
}
```
