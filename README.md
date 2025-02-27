# Missing Spatial-Temporal Multimodal Traffic Data Imputation and Prediction

## 📌 Introduction
This repository focuses on the **imputation and prediction of missing multimodal traffic data** using Graph Neural Networks (GNNs). We leverage the **Traffic4cast NeurIPS 2022 dataset** and implement **GRIN (Graph Recurrent Imputation Network)** to efficiently reconstruct and predict traffic data. 

## 📊 Dataset: NeurIPS 2022 Traffic4cast
We use the **Traffic4cast dataset** provided by the **Institute of Advanced Research in Artificial Intelligence (IARAI)**. This dataset contains spatiotemporal traffic data from multiple cities, recorded at high resolution. 

- 📁 **Dataset Repository:** [NeurIPS2022-traffic4cast](https://github.com/iarai/NeurIPS2022-traffic4cast)
- 🏙 **Cities Covered:** London, Madrid, Melbourne, etc.
- ⏳ **Time Granularity:** 5-minute intervals
- 🚘 **Traffic Data:** Flow, speed, and volume of vehicles

## 🔍 Algorithm: Graph Recurrent Imputation Network (GRIN)
For missing data imputation and traffic prediction, we use the **GRIN model** developed by the Graph Machine Learning Group. GRIN is designed to reconstruct missing traffic data using a **spatiotemporal graph-based approach**.

- 📁 **Algorithm Repository:** [GRIN](https://github.com/Graph-Machine-Learning-Group/grin)
- 📌 **Key Features:**
  - Graph Neural Networks (GNNs) for spatial dependencies
  - Recurrent Neural Networks (RNNs) for temporal patterns
  - Imputation and prediction in an end-to-end pipeline

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

## 🔧 Usage
To run the imputation model:

```bash
python run_imputation.py --config config/grin/traffic_block.yaml --dataset-name traffic_block --in-sample True 
```

To submit jobs on an HPC cluster (SLURM):

```bash
sbatch slurmjob.sh
```

## 📜 Citation
If you use this repository, please consider citing:

```
@inproceedings{your_citation,
  title={Missing Spatial-Temporal Multimodal Traffic Data Imputation and Prediction},
  author={Ankita Sarkar},
  booktitle={TU Dortmund},
  year={2025}
}

