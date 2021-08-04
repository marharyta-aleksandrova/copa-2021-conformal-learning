This repository contains all resources for the paper **Impact of 
Model-Agnostic Nonconformity Functions on Efficiency of Conformal 
Classifiers: an Extensive Study** by *Marharyta Aleksandrova*
and *Oleg Chertov* prepared for [COPA-2021: 10th Symposium on 
Conformal and Probabilistic Prediction with 
Applications](https://cml.rhul.ac.uk/copa2021/).

The repository includes:
1. `paper` folder - LaTeX source of the paper text.
2. `code`  folder - all code in Python used to perform the
    experiments, analyze the results and prepare plots/tables for 
    the paper.
3. `Detailed_results_for_real_datasets.pdf` - plots and tables for
    all 9 real datasets

The folder `code` includes the following:
1. Folder `datasets` - all datasets used to perform experiments;
   note, `iris` dataset was taken from `sklearn.datasets`.
2. Folder `for-paper` - contains helper text files to generate
    tables in LaTeX.
3. Folder `analysis-results` - results of experiments for all
    datasets.
4. `batch-analysis.py` - main file to launch experiments with
    different datasets and different classifiers.
5. `results-analysis.ipynb` - a notebook with visualization of 
    experimental results for a specified dataset.
6. `results-analysis-calibration.ipynb` - analysis of *calibration*
    or *validity* of conformal predictors.
7. `scripts.py` and `scrips2.py` - helper files with different
    functions.


