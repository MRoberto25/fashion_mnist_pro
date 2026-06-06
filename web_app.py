import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import tensorflow as tf
from tensorflow.keras import layers

app = Flask(__name__)

class ResidualBlock(layers.Layer):
    def __init__(self, filters, stride=1, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.stride = stride
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()
        if self.stride != 1:
            self.shortcut_conv = layers.Conv2D(filters, 1, strides=stride, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()

    def call(self, inputs):
        x = layers.ReLU()(self.bn1(self.conv1(inputs)))
        x_processed = self.bn2(self.conv2(x))
        if self.stride != 1:
            shortcut = self.shortcut_conv(inputs)
            shortcut = self.shortcut_bn(shortcut)
        else:
            shortcut = inputs
        x = layers.add([x_processed, shortcut])
        return layers.ReLU()(x)

CLASS_NAMES = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

CLASS_EMOJIS = ['👕', '👖', '🧥', '👗', '🧥', '👡', '👔', '👟', '👜', '👢']

model = None

def load_model():
    global model
    model_path = 'models/resnet_model.keras'
    if os.path.exists(model_path):
        print("Loading model...")
        model = tf.keras.models.load_model(model_path, custom_objects={'ResidualBlock': ResidualBlock})
        print("Model loaded successfully!")
    else:
        print("No model found at models/resnet_model.keras")

@app.route('/')
def index():
    model_loaded = model is not None
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        img = Image.open(file.stream)

        img_display = img.copy()
        img_display.thumbnail((300, 300))
        buf = io.BytesIO()
        img_display.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        img_gray = img.convert('L')
        img_resized = img_gray.resize((28, 28))
        img_inverted = ImageOps.invert(img_resized)

        img_array = np.array(img_inverted).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        all_probs = [
            {
                'class': CLASS_NAMES[i],
                'emoji': CLASS_EMOJIS[i],
                'probability': float(prediction[0][i]) * 100
            }
            for i in range(len(CLASS_NAMES))
        ]
        all_probs.sort(key=lambda x: x['probability'], reverse=True)

        return jsonify({
            'predicted_class': CLASS_NAMES[predicted_idx],
            'emoji': CLASS_EMOJIS[predicted_idx],
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs,
            'image_b64': img_b64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
