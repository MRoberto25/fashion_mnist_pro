import os
import numpy as np
import tensorflow as tf

def run_data_pipeline():
    print("--- [1/3] INITIALIZARE DATA PIPELINE ---")
    
    # Creăm folderul pentru date
    os.makedirs('data', exist_ok=True)
    
    print("1. Se descarcă setul de date brut...")
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    print("2. Se preprocesează și normalizează pixelii...")
    X_train = np.expand_dims(X_train.astype('float32') / 255.0, -1)
    X_test = np.expand_dims(X_test.astype('float32') / 255.0, -1)
    
    print("3. Se salvează datele procesate (Data Serialization)...")
    np.save('data/X_train.npy', X_train)
    np.save('data/y_train.npy', y_train)
    np.save('data/X_test.npy', X_test)
    np.save('data/y_test.npy', y_test)
    
    print("-> DATA PIPELINE FINALIZAT! Datele au fost salvate în folderul '/data'.\n")

if __name__ == "__main__":
    run_data_pipeline()
