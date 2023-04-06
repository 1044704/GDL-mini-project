# GDL Mini-Project - "Does a heterophily-friendly aggregator improve GraphCON performance on heterophilic graphs?"

This repository contains the code used to conduct experiments for the Geometric Deep Learning mini-project (Easter vacation 2023) submission for candidate 1044704.

The repo is a fork of https://github.com/tk-rusch/GraphCON.

Key modifications include:
- The introduction of new models in the _**models.py**_ file (not all reported in the project report):
   - MLP
   - GCN
   - GAT
   - H2GCN-lite (building on an implementation of H2GCN given by https://github.com/GitEventhandler/H2GCN-PyTorch)
   - GRAFF
   - GraphCON-GRAFF 
   - GraphCON-H2GCN
-  Modifications to GraphCON-GCN and GraphCON-GAT in _**models.py**_ to report Dirichlet energy

- The notebook _**hyperparameter_sweep.ipynb**_, used to conduct a random hyperparameter searches for different models.
- The notebook _**figure_gen.ipynb**_, used to generate figures for the project.
- Modifications to _**run_GNN.py**_ to accommodate addition models and record their Dirichlet energy for different layers. 
- The file _**run_GNN_best_results.py**_ to run the best versions of each model.
- The file _**best_params.py**_ was extended to include the best hyperparameters found for the new models evaluated.

