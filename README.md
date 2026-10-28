# DSA4262_Genomix

DSA4262 Team Project — Prediction of m6A RNA modifications from direct RNA sequencing data.

N6-methyladenosine (m6A) is a common mRNA modification involved in processes such as RNA degradation and translation. Nanopore direct RNA sequencing captures signals that can help identify these modifications.

This repository contains the Task 1 implementation of a reproducible machine learning workflow for site-level m6A prediction using m6Anet-processed sequencing signals and sequence features. It aims to:

- Distinguish modified from unmodified transcript sites and generate continuous m6A prediction scores for unseen data.
- Evaluate model performance and generalisation using gene-disjoint training and test partitions.
- Compare the final XGBoost model against a simple logistic regression baseline and pretrained m6Anet.
- Provide a documented prediction workflow with a pretrained model and a runnable example.

## Project Documentation

### Task 1: m6A Prediction

Training, prediction and model evaluation using XGBoost.

See the [Task 1 documentation](Task1/README.md) for installation instructions, a step-by-step prediction example, input/output formats and model evaluation.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE.md) for details.
