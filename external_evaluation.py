import pandas as pd 
import numpy as np
import pickle
import argparse
from sklearn.decomposition import PCA
from sklearn.metrics import recall_score,  precision_score
from sklearn import metrics

default_seed = 1234

def assign_label(HR_FLAG, D_OS, D_PFS):
    if HR_FLAG == "TRUE":
        return 1
    elif HR_FLAG == "FALSE":
        return 0
    elif HR_FLAG == "CENSORED":
        return (1 if ((D_OS < 18.0) or (D_PFS < 18.0)) else 0) ## Using the hint.

def evaluate_test_dataset():
    '''
    This function contains nearly the same transformations as they are in the notebook. 
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--clinical_annotations_filename', help='clinical_annotations_filename', type=str,default="sc3_Training_ClinAnnotations.csv")
    parser.add_argument('--gene_expression_filename', help='gene expression filename', type=str,default="MMRF_CoMMpass_IA9_E74GTF_Salmon_entrezID_TPM_hg19.csv")

    args = parser.parse_args()
    
    clinical_annotations_filename = args.clinical_annotations_filename
    gene_expression_filename = args.gene_expression_filename

    '''
    Should be same as the value used in the notebook.
    '''
    fraction_of_variance = 0.9 

    try:
        df_clinical_annotations = pd.read_csv(clinical_annotations_filename)    
    except IOError as e:
          print(e.errno)        
          raise e        

    try:
        df_gene_expressions = pd.read_csv(gene_expression_filename) 
    except IOError as e:
          print(e.errno)
          raise e        

    '''
    Pre-processing: Convert Entrez ids to string, to retain  consisteny with other columns while merging. 
    '''

    df_gene_expressions.rename(columns={"Unnamed: 0":"Entrez id"},inplace=True)
    df_gene_expressions["Entrez id"] = df_gene_expressions["Entrez id"].astype("str")
    df_gene_expressions.set_index("Entrez id",inplace=True)

    '''
    Pre-processing: Remove gene records for which there is no information (zero rows) for all patients. 
    '''
    df_gene_expressions = df_gene_expressions.loc[~(df_gene_expressions==0.0).all(axis=1)]

    '''
    Pre-processing: The scales for gene expressions vary a lot. Hence, apply the min-max scaling. 
    '''
    
    df_gene_expressions = (df_gene_expressions.T-df_gene_expressions.T.min(axis=0))/(df_gene_expressions.T.max(axis=0)-df_gene_expressions.T.min(axis=0))

    '''
    Deleting the columns containing no information.
    '''
    columns_to_be_dropped = []    
    for column in df_clinical_annotations.columns.tolist():
        if len(df_clinical_annotations[column].unique().tolist()) == 1:
            columns_to_be_dropped.append(column)
    

    columns_to_be_dropped.extend(["WES_mutationFileMutect","WES_mutationFileStrelkaIndel","WES_mutationFileStrelkaSNV","RNASeq_geneLevelExpFileSamplId"])

    df_clinical_annotations.drop(columns = columns_to_be_dropped,inplace=True)



    
    '''
    Pre-processing: Map gender to 1 or 0.
    '''
    df_clinical_annotations["D_Gender"]=df_clinical_annotations['D_Gender'].map({'Male': 1, 'Female': 0}) 

    '''
    Pre-processing: Convert days to months.
    '''
    df_clinical_annotations["D_OS"] /= 30.5
    df_clinical_annotations["D_PFS"] /= 30.5

    '''
    Pre-processing: Convert TRUE, FALSE, or CENSORED to 1 or 0. 
    '''
    df_clinical_annotations['HR_FLAG'] = df_clinical_annotations.apply(lambda row : assign_label(row['HR_FLAG'],
                        row['D_OS'], row['D_PFS']), axis = 1)

    '''
    Exploration and pre-processing: We observe high correlation between feature pairs (CYTO_predicted_feature_05, CYTO_predicted_feature_17) and
    (CYTO_predicted_feature_02, CYTO_predicted_feature_12). We remove one feature from these pairs to reduce dimensionality. 

    The features D_OS and D_PFS determine the HR_FLAG for CENSORED patients. We remove these features to avoid model leakage. 
    '''


    df_clinical_annotations.drop(columns=["CYTO_predicted_feature_10","CYTO_predicted_feature_05","CYTO_predicted_feature_02","Patient","D_PFS","D_OS"],inplace=True)

    df_clinical_annotation_columns = df_clinical_annotations.columns.tolist()
    df_clinical_annotation_columns.remove("RNASeq_transLevelExpFileSamplId")
    df_clinical_annotation_columns.remove("HR_FLAG")



    target = "HR_FLAG"

    '''
    Pre-processing: Join the df_clinical_annotations and df_gene_expressions. 
    For each patient record, we now have gene expression columns. 
    '''

    df_merged = pd.merge(df_clinical_annotations, df_gene_expressions, left_on='RNASeq_transLevelExpFileSamplId',right_index=True)

    '''
    Pre-processing: Split  the merged dataframe into X and Y.
    '''
    Y_test = df_merged[target].to_numpy()
    X_test = df_merged.drop(columns=["RNASeq_transLevelExpFileSamplId", "HR_FLAG"])

    '''
    Pre-processing: The D_ISS column contains a small number of nan values. Replacing these with 0.
    '''
    
    X_test.fillna(0,inplace=True)

    '''
    Pre-processing: Bring the remaining columns in df_clinical_annotation_columns to the same scale.
    '''

    X_test[df_clinical_annotation_columns] = (X_test[df_clinical_annotation_columns] - X_test[df_clinical_annotation_columns].min())/(X_test[df_clinical_annotation_columns].max()-X_test[df_clinical_annotation_columns].min())

    X_test = PCA(n_components=fraction_of_variance, random_state=default_seed,svd_solver = "full").fit_transform(X_test)

    ''' 
    Processing:
    Models
    1) Logistic regression.
    2) Ensamble decision tree with bagging.
    3) Random forest
    4) Multi-layer perceptron
    5) Support vector machine
    6) 5-nearest neighbor classification
    '''
    classifier_list = ["Multi-layer Perceptron","Support vector machine","Logistic regression","Bagging Decision tree", "Random forest",
    "Nearest neighbors"]

    
    for classifier in classifier_list:
        try:
            with open(classifier+'.pkl', 'rb') as file:
                clf = pickle.load(file)
        except IOError as e:
          print(e.errno)
          continue

        print("+=================")
        print("model=", classifier)
        Y_pred = clf.predict(X_test)
        print("Accuracy=",clf.score(X_test, Y_test)) 
        print("Recall= ",recall_score(Y_test, Y_pred))
        print("Precision= ",precision_score(Y_test, Y_pred))
        fpr, tpr, thresholds = metrics.roc_curve(Y_test, clf.predict_proba(X_test)[:,1])
        print("AUC=",metrics.auc(fpr, tpr))
        
        

if __name__ == "__main__":
    evaluate_test_dataset()