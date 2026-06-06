import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2

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
CLASS_COLORS = [
    '#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#e67e22',
    '#1abc9c', '#f39c12', '#e91e63', '#00bcd4', '#795548'
]

model = None
grad_model = None


def load_model():
    global model, grad_model
    model_path = 'models/resnet_model.keras'
    if os.path.exists(model_path):
        print("Loading model...")
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'ResidualBlock': ResidualBlock}
        )
        print("Model loaded successfully!")
        _build_grad_model()
    else:
        print("No model found at models/resnet_model.keras")


def _build_grad_model():
    global grad_model
    last_conv_layer = None
    for layer in model.layers:
        if isinstance(layer, (layers.Conv2D, ResidualBlock)):
            last_conv_layer = layer
    if last_conv_layer is not None:
        grad_model = models.Model(
            inputs=model.inputs,
            outputs=[last_conv_layer.output, model.output]
        )
        print(f"Grad-CAM model built using layer: {last_conv_layer.name}")
    else:
        print("Could not find conv layer for Grad-CAM")


def smart_preprocess(img: Image.Image) -> np.ndarray:
    """
    Improved preprocessing that:
    1. Detects whether background is light or dark using corner sampling
    2. Applies CLAHE-style contrast enhancement
    3. Centers the garment via bounding box crop
    4. Resizes to 28x28 with padding to preserve aspect ratio
    """
    img_gray = img.convert('L')
    arr = np.array(img_gray, dtype=np.uint8)

    h, w = arr.shape
    margin = max(1, min(h, w) // 10)
    corners = [
        arr[:margin, :margin],
        arr[:margin, w - margin:],
        arr[h - margin:, :margin],
        arr[h - margin:, w - margin:]
    ]
    bg_mean = np.mean([c.mean() for c in corners])

    if bg_mean > 127:
        arr = 255 - arr

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    arr = clahe.apply(arr)

    _, binary = cv2.threshold(arr, 30, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        pad = max(bw, bh) // 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad)
        y2 = min(h, y + bh + pad)
        arr = arr[y1:y2, x1:x2]

    side = max(arr.shape[0], arr.shape[1])
    canvas = np.zeros((side, side), dtype=np.uint8)
    offset_y = (side - arr.shape[0]) // 2
    offset_x = (side - arr.shape[1]) // 2
    canvas[offset_y:offset_y + arr.shape[0], offset_x:offset_x + arr.shape[1]] = arr

    final = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)

    final = final.astype('float32') / 255.0
    final = np.expand_dims(final, axis=-1)
    final = np.expand_dims(final, axis=0)
    return final


def generate_gradcam(img_array: np.ndarray, pred_idx: int) -> np.ndarray | None:
    if grad_model is None:
        return None
    try:
        img_tensor = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            conv_output, predictions = grad_model(img_tensor)
            class_score = predictions[:, pred_idx]

        grads = tape.gradient(class_score, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output_np = conv_output[0].numpy()
        pooled_grads_np = pooled_grads.numpy()

        for i in range(pooled_grads_np.shape[-1]):
            conv_output_np[:, :, i] *= pooled_grads_np[i]

        heatmap = np.mean(conv_output_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None


def heatmap_to_b64(heatmap: np.ndarray, original_arr_28: np.ndarray) -> str:
    heatmap_resized = cv2.resize(heatmap, (28, 28))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    original_gray = np.uint8(original_arr_28 * 255)
    original_rgb = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)

    overlay = cv2.addWeighted(heatmap_colored, 0.5, original_rgb, 0.5, 0)

    scale = 8
    overlay_large = cv2.resize(overlay, (28 * scale, 28 * scale), interpolation=cv2.INTER_NEAREST)
    original_large = cv2.resize(original_rgb, (28 * scale, 28 * scale), interpolation=cv2.INTER_NEAREST)

    combined = np.hstack([original_large, overlay_large])

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), facecolor='#1a1a2e')
    fig.suptitle('Grad-CAM Analysis', color='white', fontsize=11, fontweight='bold', y=1.02)

    axes[0].imshow(cv2.cvtColor(original_large, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Preprocessed Input', color='#aaa', fontsize=8)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(overlay_large, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Grad-CAM Heatmap', color='#aaa', fontsize=8)
    axes[1].axis('off')

    plt.tight_layout(pad=0.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=100, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


def confusion_matrix_b64(probabilities: list) -> str:
    probs = np.array(probabilities)
    short_names = [
        'T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
    ]

    matrix_data = probs.reshape(1, 10)

    fig, ax = plt.subplots(figsize=(8, 2.2), facecolor='#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    im = ax.imshow(matrix_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(range(10))
    ax.set_xticklabels(short_names, rotation=35, ha='right', fontsize=8, color='white')
    ax.set_yticks([0])
    ax.set_yticklabels(['Score'], color='white', fontsize=8)

    for j in range(10):
        val = matrix_data[0, j]
        text_color = 'black' if val > 0.5 else 'white'
        ax.text(j, 0, f'{val:.2f}', ha='center', va='center',
                fontsize=8, fontweight='bold', color=text_color)

    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=7)

    ax.set_title('Prediction Confidence Matrix (per class)', color='white',
                 fontsize=9, fontweight='bold', pad=8)

    for spine in ax.spines.values():
        spine.set_edgecolor('#444')

    plt.tight_layout(pad=0.6)
    buf = io.BytesIO()
    plt.savefig(buf, format='PNG', dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')


@app.route('/')
def index():
    return render_template('index.html', model_loaded=model is not None)


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
        img = Image.open(file.stream).convert('RGB')

        img_display = img.copy()
        img_display.thumbnail((300, 300))
        buf = io.BytesIO()
        img_display.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        img_array = smart_preprocess(img)

        prediction = model.predict(img_array, verbose=0)
        predicted_idx = int(np.argmax(prediction))
        confidence = float(np.max(prediction)) * 100

        raw_probs = prediction[0].tolist()

        all_probs = [
            {
                'class': CLASS_NAMES[i],
                'emoji': CLASS_EMOJIS[i],
                'color': CLASS_COLORS[i],
                'probability': float(prediction[0][i]) * 100
            }
            for i in range(len(CLASS_NAMES))
        ]
        all_probs_sorted = sorted(all_probs, key=lambda x: x['probability'], reverse=True)

        heatmap = generate_gradcam(img_array, predicted_idx)
        gradcam_b64 = None
        if heatmap is not None:
            arr_28 = img_array[0, :, :, 0]
            gradcam_b64 = heatmap_to_b64(heatmap, arr_28)

        conf_matrix_b64 = confusion_matrix_b64(raw_probs)

        return jsonify({
            'predicted_class': CLASS_NAMES[predicted_idx],
            'emoji': CLASS_EMOJIS[predicted_idx],
            'confidence': round(confidence, 2),
            'all_probabilities': all_probs_sorted,
            'image_b64': img_b64,
            'gradcam_b64': gradcam_b64,
            'confusion_matrix_b64': conf_matrix_b64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
