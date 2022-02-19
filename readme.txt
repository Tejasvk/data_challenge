This repository contains code to solve an old Multiple Myeloma DREAM Challenge (https://www.synapse.org/#!Synapse:syn6187098/wiki/401884).

Read the challenge statement in the document "Data challenge.pdf".

1) challenge_code.ipynb: Includes the code for transformations and model training.  Run all cells from this notebook first to generate the model pickle files and plots. 


2) external_validation.py: Used for evaluation on external test data sets. 

usage: "python external_evaluation.py --clinical_annotations_filename ___  --gene_expression_filename ___"

3) requirement.txt: Contains the library versions used.

4) report.pdf: Documenting the challenge outcomes.
