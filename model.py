# Import necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pickle
import pandas as pd
import numpy as np

# Initialize Spark session
spark = SparkSession.builder.appName("Personalized Medical Recommendation System").getOrCreate()

# Load the dataset
dataset = pd.read_csv('Training.csv')
df = spark.createDataFrame(dataset)

# Convert 'prognosis' column to indexed labels
indexer = StringIndexer(inputCol='prognosis', outputCol='label')
df = indexer.fit(df).transform(df)

# Prepare feature columns (excluding 'prognosis' and label)
feature_columns = df.columns[:-1]  # All columns except 'label'
assembler = VectorAssembler(inputCols=feature_columns.tolist(), outputCol='features')
df = assembler.transform(df)

# Split the dataset into training and testing sets
train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)

# Create and train the Random Forest model
rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=100)
pipeline = Pipeline(stages=[assembler, rf])
model = pipeline.fit(train_df)

# Save the model as a .pkl file
with open('svc.pkl', 'wb') as f:
    pickle.dump(model, f)



spark.stop()
