import os
import sys

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(os.path.join(base_dir, 'D:/TU Dortmund/Semesters/Summer Semester 2024/Thesis/Experiment run/grin'))

config = {
    'logs': 'logs/'
}
datasets_path = {
    'air': 'datasets/air_quality',
    'la': 'datasets/metr_la',
    'bay': 'datasets/pems_bay',
    'traffic': 'datasets/t4c22_dataset_geometric',
    'synthetic': 'datasets/synthetic'
}
epsilon = 1e-8

for k, v in config.items():
    config[k] = os.path.join(base_dir, v)
for k, v in datasets_path.items():
    datasets_path[k] = os.path.join(base_dir, v)
