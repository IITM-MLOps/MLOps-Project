from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType
import numpy as np
import struct
import pickle
from utils import *
from dense_neural_class import *
import json
import mlflow


# Use a relative path that resolves to the current directory
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("MNIST_Digit_Classification")



# Initialize Spark Session
spark = SparkSession.builder.appName("MNIST Data Ingestion").getOrCreate()

def load_mnist_images_spark(filename):
    with open(filename, 'rb') as f:
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows*cols)

    # Create RDD
    rdd = spark.sparkContext.parallelize(images.tolist())
    schema = StructType([StructField(f'pixel_{i}', IntegerType(), False) for i in range(rows*cols)])
    df = spark.createDataFrame(rdd, schema=schema)
    return df

def load_mnist_labels_spark(filename):
    with open(filename, 'rb') as f:
        magic, num_labels = struct.unpack('>II', f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    rdd = spark.sparkContext.parallelize(labels.tolist())
    schema = StructType([StructField('label', IntegerType(), False)])
    df = spark.createDataFrame(rdd.map(lambda x: (x,)), schema=schema)
    return df

# Load using Spark
train_images_df = load_mnist_images_spark('./mnist/train-images.idx3-ubyte')
train_labels_df = load_mnist_labels_spark('./mnist/train-labels.idx1-ubyte')
test_images_df = load_mnist_images_spark('./mnist/t10k-images.idx3-ubyte')
test_labels_df = load_mnist_labels_spark('./mnist/t10k-labels.idx1-ubyte')

# To use in your model (convert to numpy arrays)
X = np.array(train_images_df.collect())
X = np.array([list(row) for row in X])  # Convert Row to list
Y = np.array(train_labels_df.collect()).reshape(-1, 1)

X_test = np.array(test_images_df.collect())
X_test = np.array([list(row) for row in X_test])
Y_test = np.array(test_labels_df.collect()).reshape(-1, 1)

print(f"Training images shape: {X.shape}")
print(f"Training labels shape: {Y.shape}")
print(f"Testing images shape: {X_test.shape}")
print(f"Testing labels shape: {Y_test.shape}")

# Define hyperparameters for training
learning_rate = 0.005
epochs_initial = 11
epochs_improve = 61
batch_size_initial = 60000
batch_size_improve = 40
hidden_layer1_size = 50
hidden_layer2_size = 20
output_size = 10

# Start an MLflow run to track the experiment
with mlflow.start_run(run_name="MNIST_Training_Run") as run:
    # Log parameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("epochs_initial", epochs_initial)
    mlflow.log_param("epochs_improve", epochs_improve)
    mlflow.log_param("batch_size_initial", batch_size_initial)
    mlflow.log_param("batch_size_improve", batch_size_improve)
    mlflow.log_param("hidden_layer1_size", hidden_layer1_size)
    mlflow.log_param("hidden_layer2_size", hidden_layer2_size)
    mlflow.log_param("output_size", output_size)

# Putting the data in a best-known format.

# Instantiation of the neural network as model2.
model2 = Dense_Neural_Diy(input_size=784, hidden_layer1_size=50, hidden_layer2_size=20 , output_size=10)

model2.fit(X,Y,learning_rate=0.005, epochs=11, batch_size=60000 )


model2.improve_train(X,Y, learning_rate=0.005, epochs=61, batch_size=40)

# Prediction using test set
Y_pred_test = model2.predict(X_test).reshape(-1,1)
# Prediction using train set
Y_hat = model2.predict(X).reshape(-1,1).reshape(-1,1)

# Calculate metrics
train_accuracy = float(np.mean(Y == Y_hat))
test_accuracy = float(np.mean(Y_test == Y_pred_test))

print(f'Accuracy on Test: {test_accuracy}')
print(f'Accuracy on Train: {train_accuracy}')

mlflow.log_metric("train_accuracy", train_accuracy)
mlflow.log_metric("test_accuracy", test_accuracy)

# Log metrics to metrics.json
metrics = {
    "train_accuracy": train_accuracy,
    "test_accuracy": test_accuracy
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
    print("Metrics saved to metrics.json")

# mlflow.log_artifact("metrics.json")
# mlflow.log_artifact("model_save_test.pkl") 

# Log additional information (custom tags)
mlflow.set_tag("model_type", "Custom Neural Network")
mlflow.set_tag("dataset", "MNIST")
print("########Experiment tracked with MLflow##########")
# Saving the Model
save_model('model_save_test', model2)

if __name__ == "__main__":
    # Load the model
    model2 = load_model('model_save_test')
    print(model2)
    # Prediction using test set
    Y_pred_test = model2.predict(X_test).reshape(-1,1)
    # Prediction using train set
    Y_hat = model2.predict(X).reshape(-1,1).reshape(-1,1)

    print(f'Accuracy on Test: {np.mean(Y_test == Y_pred_test)}')
    print(f'Accuracy on Train: {np.mean(Y == Y_hat)}')