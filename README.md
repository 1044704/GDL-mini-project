This repository contains the code used to conduct experiments for the Geometric Deep Learning mini-project (Easter vacation 2023).

The repo is a fork of https://github.com/tk-rusch/GraphCON.

Key modifications include:
- The introduction of new models in the **models.py** file (not all reported in the project report):
   - MLP
   - GCN
   - GAT
   - H2GCN-lite
   - GRAFF
   - GraphCON-GRAFF
   - GraphCON-H2GCN
   - Modifications to GraphCON-GCN and GraphCON-GAT to report Dirichlet energy

- The notebook **hyperparameter_sweep.ipynb**, used to conduct a random hyperparameter searches for different models.
- The notebook **figure_gen.ipynb**, used to generate figures for the project.
- Modifications to **run_GNN.py** to accommodate addition models and record their Dirichlet energy for different layers. 
- The file **run_GNN_best_results** to run the best versions of each model.
- The file **best_params.py** was extended to include the best hyperparameters found for the new models evaluated.

