from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import numpy as np
import struct

def load_data():
    # Add your MNIST loading logic here
    train_images = load_mnist_images('./mnist/train-images.idx3-ubyte')
    # Return processed data
    return train_images.reshape(-1, 28*28).astype(np.float32) / 255.0

def save_data(ti):
    processed_data = ti.xcom_pull(task_ids='load_task')
    np.save('./data/processed/train_data.npy', processed_data)

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2024, 1, 1)
}

with DAG('mnist_pipeline', 
         default_args=default_args,
         schedule_interval='@daily') as dag:
    
    load_task = PythonOperator(
        task_id='load_data',
        python_callable=load_data
    )
    
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=lambda: print("Processing completed")
    )
    
    save_task = PythonOperator(
        task_id='save_data',
        python_callable=save_data
    )

    load_task >> process_task >> save_task
