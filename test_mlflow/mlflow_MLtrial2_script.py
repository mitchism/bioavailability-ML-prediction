import os
import warnings
import sys

import pandas as pd
import numpy as np
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# ML Trial 2 
import findspark
findspark.init('/home/mitch/spark-3.3.0-bin-hadoop2')

import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vector
from pyspark.ml.feature import (VectorAssembler,VectorIndexer)
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.ml.feature import (StringIndexer,OneHotEncoder)
from pyspark.ml.feature import StandardScaler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    featuresDF = pd.read_parquet('featuresCatalogDF.parquet')
    featuresDF.head(10)

    # MSM VM config prep
    spark = SparkSession.builder.appName('BApredsV1').getOrCreate()

    # load the data
    data = spark.read.csv("bioavailability_data_final.csv",inferSchema=True,sep=',',header=True)

    # --- suppress future spark warnings/error/etc output ---
    spark.sparkContext.setLogLevel("OFF")


    ''' # FEATURE SELECTION:
    '''
    featuresDF = pd.read_parquet('featuresCatalogDF.parquet')
    feature_set1a = featuresDF.loc[0,'features']
    feature_set1b = featuresDF.loc[1,'features']
    feature_set2a = featuresDF.loc[2,'features']
    feature_set2b = featuresDF.loc[3,'features']
    feature_set3  = featuresDF.loc[4,'features']
    feature_set4a = featuresDF.loc[5,'features']
    feature_set4b = featuresDF.loc[6,'features']
    F1bANOVA = featuresDF.loc[7,'features']
    F2bANOVA = featuresDF.loc[8,'features']

    ''' # VECTORIZE FEATURES
    '''
    # VECTOR ASSEMBLY - feature sets 1a,1b,2a,2b,3,4a,4b
    vec_assembler1a = VectorAssembler(inputCols = feature_set1a, outputCol='features1a')
    vec_assembler1b = VectorAssembler(inputCols = feature_set1b, outputCol='features1b')
    vec_assembler2a = VectorAssembler(inputCols = feature_set2a, outputCol='features2a')
    vec_assembler2b = VectorAssembler(inputCols = feature_set2b, outputCol='features2b')
    vec_assembler3 = VectorAssembler(inputCols = feature_set3, outputCol='features3')
    vec_assembler4a = VectorAssembler(inputCols = feature_set4a, outputCol='features4a')
    vec_assembler4b = VectorAssembler(inputCols = feature_set4b, outputCol='features4b')
    vec_assembler1bANOVA = VectorAssembler(inputCols = F1bANOVA, outputCol='F1bANOVA')
    vec_assembler2bANOVA = VectorAssembler(inputCols = F2bANOVA, outputCol='F2bANOVA')

    feature_pipeline = Pipeline(stages=[vec_assembler1a,
                                        vec_assembler1b,
                                        vec_assembler2a,
                                        vec_assembler2b,
                                        vec_assembler3,
                                        vec_assembler4a,
                                        vec_assembler4b,
                                        vec_assembler1bANOVA,
                                       vec_assembler2bANOVA])
    data_features = feature_pipeline.fit(data).transform(data)


    ''' # DEPENDENT VARIABLE LABELS 
    '''
    qd5 = QuantileDiscretizer(numBuckets=5,inputCol='BA_pct',outputCol='label_QD5')

    data_features = qd5.fit(data_features).transform(data_features)

    # -- INDEX / ENCODE LABELS
    label_quant0 = 'BA_pct'
    label_cat0_vector = OneHotEncoder(inputCol='label_QD5',outputCol='label_cat0_vector')

    label_cat1_index = StringIndexer(inputCol='label1',outputCol='label_cat1_index')
    label_cat1_vector = OneHotEncoder(inputCol='label_cat1_index',outputCol='label_cat1_vector')

    label_cat2_index = StringIndexer(inputCol='label2',outputCol='label_cat2_index')
    label_cat2_vector = OneHotEncoder(inputCol='label_cat2_index',outputCol='label_cat2_vector')

    label_cat3_index = StringIndexer(inputCol='label3a',outputCol='label_cat3_index')
    label_cat3_vector = OneHotEncoder(inputCol='label_cat3_index',outputCol='label_cat3_vector')

    label_cat4_index = StringIndexer(inputCol='label3b',outputCol='label_cat4_index')
    label_cat4_vector = OneHotEncoder(inputCol='label_cat4_index',outputCol='label_cat4_vector')

    label_pipeline = Pipeline(stages=[label_cat0_vector,
                                     label_cat1_index,label_cat1_vector,
                                     label_cat2_index,label_cat2_vector,
                                     label_cat3_index,label_cat3_vector,
                                     label_cat4_index,label_cat4_vector])

    data_features = data_features.select(['Name','BA_pct',
                                          'label_QD5','label1','label2','label3a','label3b',
                                          'features1a','features1b','features2a','features2b',
                                          'features3','features4a','features4b','F1bANOVA','F2bANOVA'])

    data_prefinal = label_pipeline.fit(data_features).transform(data_features)

    data_prefinal2 = data_prefinal.withColumnRenamed('BA_pct','label_q0')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label_QD5','label_cat0')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label_cat1_index','label_cat1')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label_cat2_index','label_cat2')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label3a','label3')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label3b','label4')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label_cat3_index','label_cat3')
    data_prefinal2 = data_prefinal2.withColumnRenamed('label_cat4_index','label_cat4')

    data_final = data_prefinal2.select(['Name',
                                        'label_q0',
                                        'label_cat0','label_cat1',
                                        'label_cat2','label_cat3','label_cat4',
                                        'features1a','features1b','features2a','features2b',
                                        'features3','features4a','features4b',
                                        'F1bANOVA','F2bANOVA'])


    ''' # TEST FEATURE SCALING
    '''
    # Scale values
    scaler1 = StandardScaler(inputCol="features2b", outputCol="features2bSs", withStd=True, withMean=False)
    scaler2 = StandardScaler(inputCol="features2b", outputCol="features2bSm", withStd=False, withMean=True)

    data_final = scaler1.fit(data_final).transform(data_final)
    data_final = scaler2.fit(data_final).transform(data_final)

    
    ''' # DEFINE FEATURE CHOICE PARAM FOR MLFLOW '''
    choices = {'features1a','features1b','features2a','features2b','features3','features4a','features4b','F1bANOVA','F2bANOVA'}
    feature_choice = sys.argv[1] if sys.argv[1] in choices else 'features1b'

    with mlflow.start_run():
        labelName = 'label_q0'  
        
        #lr_df = pd.DataFrame()
        #df = lr_df

        #allFeatures = ['features1a','features1b',
        #               'features2a','features2b',
        #               'features3','features4a','features4b',
        #               'F1bANOVA','F2bANOVA',"features2bSs","features2bSm"]

        #subset = data_final.select(['Name',labelName,
        #                            'features1a','features1b',
        #                            'features2a','features2b',
        #                            'features3','features4a','features4b',
        #                            'F1bANOVA','F2bANOVA',"features2bSs","features2bSm"]) 

        #train,test = subset.randomSplit([0.7,0.3])
        train,test = data_final.randomSplit([0.7,0.3])

        #for index,features in enumerate(allFeatures):
        #featnum = ['F1a','F1b','F2a','F2b','F3','F4a','F4b','F1bANOVA','F2bANOVA','F2bSs','F2bSm']
        featuresName = feature_choice
        
        lr = LinearRegression(featuresCol=featuresName,labelCol=labelName,predictionCol='prediction')
        
        ''' # SPECIFY MODEL 
        '''
        modeltype = lr 
        #modeltypeVariantNo = '1' 
        modelname = f"lr_{featuresName}"  
        
        # FIT/TRAIN MODEL & TRANSFORM DATA
        mymodel = modeltype.fit(train)
        myresults = mymodel.transform(test)
        
        
        # CALCULATE KEY EVALS
        regEvaluator = RegressionEvaluator(labelCol=labelName,predictionCol='prediction')
        evalMetrics = {regEvaluator:['rmse','mse','mae','r2','var']}
        
        eval_results = {}
        eval_results[feature_choice] = {}

        evaluator = regEvaluator
        for each_metric in evalMetrics[evaluator]:        
            metric = each_metric
            result = evaluator.evaluate(myresults, {evaluator.metricName: metric})
            
            eval_results[feature_choice][each_metric] = result

        df_out = pd.DataFrame.from_dict(eval_results)

        ''' # LOG PARAMS & METRICS IN MLFLOW '''
        mlflow.log_param("features", feature_choice)
        mlflow.log_metric("rmse", eval_results[feature_choice]['rmse'])
        mlflow.log_metric("mae", eval_results[feature_choice]['mae'])
        mlflow.log_metric("r2", eval_results[feature_choice]['r2'])
        mlflow.spark.log_model(mymodel,"trial2_LinReg")


        #tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        #mlflow.spark.log_model(RFmodel,"spark-model")

    print(eval_results)
    df_out.head()