from airflow import DAG
from airflow.operators.python import PythonOperator  # Updated import path
from datetime import datetime
import numpy as np
import time

def load_data():
    # Your MNIST loading logic
    train_images = np.random.rand(60000, 28, 28)  # Example placeholder
    return train_images.reshape(-1, 28*28).astype(np.float32) / 255.0

def save_data(ti):
    processed_data = ti.xcom_pull(task_ids='load_task')
    np.save('./data/processed/train_data.npy', processed_data)
    print(f"Saved data of shape {processed_data.shape}")

default_args = {
    'owner': 'mlops',
    'start_date': datetime(2024, 1, 1),
    'retries': 1
}

with DAG(
    'mnist_pipeline',
    default_args=default_args,
    schedule='@daily',  # Changed parameter name
    catchup=False,
    tags=['mnist']
) as dag:
    
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
