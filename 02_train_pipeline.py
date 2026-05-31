import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.stride = stride
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        
        # Only create these layers if we actually need to change the shape
        if self.stride != 1:
            self.shortcut_conv = layers.Conv2D(filters, 1, strides=stride, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = layers.ReLU()(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        # If shape changes, apply shortcut layers. Otherwise, just pass inputs directly.
        if self.stride != 1:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut)
        else:
            shortcut = inputs 
            
        x = layers.add([x, shortcut])
        return layers.ReLU()(x)

def build_advanced_resnet(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    x = ResidualBlock(32)(x)
    x = ResidualBlock(64, stride=2)(x)
    
    x = layers.GlobalAveragePooling2D(name="final_avg_pool")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return models.Model(inputs, outputs, name="Custom_ResNet")

def run_training_pipeline():
    print("--- [2/3] INITIALIZARE TRAINING PIPELINE ---")
    os.makedirs('models', exist_ok=True)
    
    print("1. Se încarcă datele procesate din Data Pipeline...")
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    
    print("2. Se construiește arhitectura Custom ResNet...")
    model = build_advanced_resnet()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
    ]
    
    print("3. Începe antrenarea modelului...")
    model.fit(X_train, y_train, batch_size=128, epochs=3, validation_split=0.2, callbacks=callbacks)
    
    print("4. Se salvează modelul în producție...")
    model.save('models/resnet_model.keras')
    print("-> TRAINING PIPELINE FINALIZAT! Modelul a fost salvat în folderul '/models'.\n")

if __name__ == "__main__":
    run_training_pipeline()
