import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Trebuie să redefinim ResidualBlock pentru ca TensorFlow să știe cum să încarce modelul custom
class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        self.shortcut = models.Sequential()
        if stride != 1:
            self.shortcut.add(layers.Conv2D(filters, 1, strides=stride, use_bias=False))
            self.shortcut.add(layers.BatchNormalization())
    def call(self, inputs):
        x = tf.nn.relu(self.bn1(self.conv1(inputs)))
        return tf.nn.relu(layers.add([self.bn2(self.conv2(x)), self.shortcut(inputs)]))

def generate_grad_cam(img_array, model, last_conv_layer_name):
    grad_model = models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, tf.argmax(preds[0])]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.maximum(tf.squeeze(heatmap), 0) / tf.math.reduce_max(tf.squeeze(heatmap))
    return heatmap.numpy()

def run_evaluation_pipeline():
    print("--- [3/3] INITIALIZARE EVALUATION PIPELINE ---")
    os.makedirs('outputs', exist_ok=True)
    
    class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    print("1. Se încarcă modelul de producție și datele de test...")
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    
    # Încărcăm modelul recunoscând stratul nostru Custom
    model = tf.keras.models.load_model('models/resnet_model.keras', custom_objects={'ResidualBlock': ResidualBlock})
    
    print("2. Se generează metricele de performanță...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n" + classification_report(y_test, y_pred, target_names=class_names))
    
    # Salvăm matricea de confuzie
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names, cmap='Blues')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('outputs/confusion_matrix.png')
    
    print("3. Se generează analiza Explainable AI (Grad-CAM)...")
    sample_img = X_test[0:1] # Luăm o gheată (boot)
    last_conv_layer = [layer.name for layer in model.layers if isinstance(layer, ResidualBlock)][-1]
    
    heatmap = generate_grad_cam(sample_img, model, last_conv_layer)
    
    # Creăm vizualizarea peste imaginea originală
    img = np.uint8(255 * sample_img[0])
    heatmap_resized = cv2.applyColorMap(np.uint8(255 * cv2.resize(heatmap, (28, 28))), cv2.COLORMAP_JET)
    superimposed_img = heatmap_resized * 0.4 + cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    plt.figure()
    plt.imshow(superimposed_img.astype("uint8"))
    plt.title("Explainable AI Heatmap")
    plt.axis('off')
    plt.savefig('outputs/gradcam_heatmap.png')
    
    print("-> EVALUATION PIPELINE FINALIZAT! Graficele au fost salvate în folderul '/outputs'.\n")

if __name__ == "__main__":
    run_evaluation_pipeline()
