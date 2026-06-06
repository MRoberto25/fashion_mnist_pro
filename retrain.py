"""
Targeted retrain — same lightweight ResidualBlock architecture as the original
(so the saved model loads without issue), but with:
  - Data augmentation during training
  - 15 epochs (vs original 3) with early stopping
  - Label smoothing + ReduceLROnPlateau
  - This should achieve ~91-93% test accuracy vs ~85% for the 3-epoch model
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

tf.random.set_seed(42)
np.random.seed(42)


class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.shortcut_conv = layers.Conv2D(filters, 1, strides=stride, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(inputs), training=training))
        x_proc = self.bn2(self.conv2(x), training=training)
        if self.stride != 1:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs), training=training)
        else:
            shortcut = inputs
        return tf.nn.relu(x_proc + shortcut)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'stride': self.stride})
        return cfg


def build_model(input_shape=(28, 28, 1), num_classes=10):
    inputs = layers.Input(shape=input_shape)

    # Augmentation — flips + small rotations help the model generalise
    aug = layers.RandomFlip('horizontal')(inputs)
    aug = layers.RandomRotation(0.06)(aug)
    aug = layers.RandomZoom(0.08)(aug)

    x = layers.Conv2D(32, 3, padding='same', use_bias=False)(aug)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = ResidualBlock(32)(x)
    x = ResidualBlock(64, stride=2)(x)
    x = ResidualBlock(64)(x)
    x = ResidualBlock(128, stride=2)(x)

    x = layers.GlobalAveragePooling2D(name='final_avg_pool')(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    return models.Model(inputs, outputs, name='Fashion_ResNet_v2')


def run():
    print('=' * 58)
    print(' FASHION RESNET v2 — Improved Accuracy Training')
    print('=' * 58)
    os.makedirs('models', exist_ok=True)

    print('\n[1/4] Loading data...')
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test  = np.load('data/X_test.npy')
    y_test  = np.load('data/y_test.npy')
    print(f'      Train: {X_train.shape}, Test: {X_test.shape}')

    print('\n[2/4] Building model...')
    model = build_model()
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'models/best_checkpoint.keras',
            monitor='val_accuracy', save_best_only=True, verbose=0
        ),
    ]

    print('\n[3/4] Training (up to 15 epochs)...')
    model.fit(
        X_train, y_train,
        batch_size=256,
        epochs=15,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )

    print('\n[4/4] Evaluating...')
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f'\n  Test Accuracy : {acc * 100:.2f}%')
    print(f'  Test Loss     : {loss:.4f}')

    model.save('models/resnet_model.keras')
    print('\n  Saved to models/resnet_model.keras')
    print('=' * 58)
    print(f' DONE — Accuracy: {acc * 100:.2f}%')
    print('=' * 58)


if __name__ == '__main__':
    run()
