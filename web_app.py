import os
import io
import base64
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
import tensorflow as tf
import keras
from tensorflow.keras import layers, models
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import cv2

app = Flask(__name__)

# ── Custom layers ─────────────────────────────────────────────────────────────
# "Simple" ResidualBlock matches the ORIGINAL saved model (no SE attention).
@keras.saving.register_keras_serializable()
class ResidualBlock(layers.Layer):
    """Original architecture — used to load the old resnet_model.keras."""
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

    def call(self, inputs):
        x = layers.ReLU()(self.bn1(self.conv1(inputs)))
        x_proc = self.bn2(self.conv2(x))
        if self.stride != 1:
            shortcut = self.shortcut_bn(self.shortcut_conv(inputs))
        else:
            shortcut = inputs
        return layers.ReLU()(layers.add([x_proc, shortcut]))

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'stride': self.stride})
        return cfg


@keras.saving.register_keras_serializable(package='Custom')
class SEBlock(layers.Layer):
    """SE attention block — only present in the retrained model."""
    def __init__(self, filters, ratio=8, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.ratio = ratio
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(max(1, filters // ratio), activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, filters))

    def call(self, x):
        se = self.gap(x)
        se = self.dense1(se)
        se = self.dense2(se)
        se = self.reshape(se)
        return x * se

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'ratio': self.ratio})
        return cfg


@keras.saving.register_keras_serializable(package='Custom')
class ResidualBlockSE(layers.Layer):
    """Upgraded block with SE attention — used in retrained model."""
    def __init__(self, filters, stride=1, use_se=True, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.stride = stride
        self.use_se = use_se
        self.conv1 = layers.Conv2D(filters, 3, strides=stride, padding='same', use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, 3, padding='same', use_bias=False)
        self.bn2 = layers.BatchNormalization()
        if use_se:
            self.se = SEBlock(filters)
        if stride != 1:
            self.proj_conv = layers.Conv2D(filters, 1, strides=stride, use_bias=False)
            self.proj_bn = layers.BatchNormalization()

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)
        if self.use_se:
            out = self.se(out)
        if self.stride != 1:
            x = self.proj_bn(self.proj_conv(x), training=training)
        return tf.nn.relu(out + x)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({'filters': self.filters, 'stride': self.stride, 'use_se': self.use_se})
        return cfg


CLASS_NAMES  = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
                 'Sandal','Shirt','Sneaker','Bag','Ankle boot']
CLASS_EMOJIS = ['👕','👖','🧥','👗','🧥','👡','👔','👟','👜','👢']
CLASS_COLORS = ['#e74c3c','#3498db','#2ecc71','#9b59b6','#e67e22',
                '#1abc9c','#f39c12','#e91e63','#00bcd4','#795548']

model      = None
grad_model = None


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    global model, grad_model
    path = 'models/resnet_model.keras'
    if not os.path.exists(path):
        print('No model file found.')
        return
    print('Loading model…')
    # Inject custom classes directly into Keras global registry so they are
    # found regardless of how the model was saved (with or without decorator).
    custom_reg = keras.saving.get_custom_objects()
    custom_reg['ResidualBlock']   = ResidualBlock
    custom_reg['SEBlock']         = SEBlock
    custom_reg['ResidualBlockSE'] = ResidualBlockSE
    try:
        model = tf.keras.models.load_model(path)
        print('Model loaded OK.')
    except Exception as e:
        print(f'ERROR loading model: {e}')
        model = None
        return
    _build_grad_model()


def _build_grad_model():
    global grad_model
    # Find the last layer that produces a 4-D feature map (conv / residual)
    last_4d = None
    for layer in model.layers:
        try:
            if len(layer.output_shape) == 4:
                last_4d = layer
        except Exception:
            pass
    if last_4d:
        grad_model = models.Model(
            inputs=model.inputs,
            outputs=[last_4d.output, model.output]
        )
        print(f'Grad-CAM attached to: {last_4d.name}')


# ── Preprocessing ─────────────────────────────────────────────────────────────
def smart_preprocess(img: Image.Image):
    """
    Returns (img_array_1x28x28x1, pipeline_steps_dict).
    pipeline_steps_dict holds intermediate uint8 28x28 images for visualisation.
    """
    steps = {}

    # 1. Original (thumbnail)
    thumb = img.copy()
    thumb.thumbnail((112, 112))
    steps['original'] = np.array(thumb.convert('L'))

    # 2. Grayscale
    gray_arr = np.array(img.convert('L'), dtype=np.uint8)
    steps['grayscale'] = cv2.resize(gray_arr, (112, 112))

    # 3. Smart invert — sample corners
    h, w = gray_arr.shape
    m = max(1, min(h, w) // 10)
    bg = np.mean([gray_arr[:m,:m], gray_arr[:m,w-m:],
                  gray_arr[h-m:,:m], gray_arr[h-m:,w-m:]])
    inverted = (255 - gray_arr) if bg > 127 else gray_arr.copy()
    steps['invert_bg'] = cv2.resize(inverted, (112, 112))

    # 4. CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4,4))
    enhanced = clahe.apply(inverted)
    steps['clahe'] = cv2.resize(enhanced, (112, 112))

    # 5. Crop to content
    _, binary = cv2.threshold(enhanced, 25, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(binary)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        pad = max(bw, bh) // 8
        x1,y1 = max(0,x-pad), max(0,y-pad)
        x2,y2 = min(w,x+bw+pad), min(h,y+bh+pad)
        cropped = enhanced[y1:y2, x1:x2]
    else:
        cropped = enhanced

    # 6. Square-pad + resize to 28x28
    side = max(cropped.shape)
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy = (side - cropped.shape[0]) // 2
    ox = (side - cropped.shape[1]) // 2
    canvas[oy:oy+cropped.shape[0], ox:ox+cropped.shape[1]] = cropped
    final28 = cv2.resize(canvas, (28, 28), interpolation=cv2.INTER_AREA)
    steps['final28'] = cv2.resize(final28, (112, 112))

    arr = final28.astype('float32') / 255.0
    arr = arr[np.newaxis, :, :, np.newaxis]
    return arr, steps, final28


# ── TTA (Test-Time Augmentation) ─────────────────────────────────────────────
def predict_with_tta(img_array: np.ndarray, n_aug: int = 8) -> np.ndarray:
    """Average predictions over original + horizontal flip + small perturbations."""
    preds = []
    base = img_array[0, :, :, 0]

    for i in range(n_aug):
        aug = base.copy()
        if i == 1:
            aug = aug[:, ::-1]          # horizontal flip
        elif i == 2:
            aug = np.clip(aug * 1.1, 0, 1)   # brighter
        elif i == 3:
            aug = np.clip(aug * 0.9, 0, 1)   # darker
        elif i == 4:
            aug = np.roll(aug, 1, axis=0)    # shift down 1px
        elif i == 5:
            aug = np.roll(aug, -1, axis=0)   # shift up 1px
        elif i == 6:
            aug = np.roll(aug, 1, axis=1)    # shift right 1px
        elif i == 7:
            aug = np.roll(aug, -1, axis=1)   # shift left 1px

        batch = aug[np.newaxis, :, :, np.newaxis]
        preds.append(model.predict(batch, verbose=0)[0])

    return np.mean(preds, axis=0)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def generate_gradcam(img_array, pred_idx):
    if grad_model is None:
        return None
    try:
        t = tf.cast(img_array, tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(t)
            conv_out, preds = grad_model(t)
            score = preds[:, pred_idx]
        grads = tape.gradient(score, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2)).numpy()
        cam = conv_out[0].numpy()
        for i in range(pooled.shape[-1]):
            cam[:,:,i] *= pooled[i]
        heatmap = np.maximum(np.mean(cam, axis=-1), 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap
    except Exception as e:
        print(f'Grad-CAM error: {e}')
        return None


# ── Saliency map ──────────────────────────────────────────────────────────────
def generate_saliency(img_array, pred_idx):
    try:
        t = tf.Variable(tf.cast(img_array, tf.float32))
        with tf.GradientTape() as tape:
            preds = model(t, training=False)
            score = preds[:, pred_idx]
        grads = tape.gradient(score, t).numpy()[0, :, :, 0]
        sal = np.abs(grads)
        if sal.max() > 0:
            sal = sal / sal.max()
        return sal
    except Exception as e:
        print(f'Saliency error: {e}')
        return None


# ── Figure renderers ──────────────────────────────────────────────────────────
BG = '#0d0d1a'
CARD = '#13132b'
ACCENT = '#7c6fff'

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='PNG', dpi=110, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode()


def render_pipeline(steps: dict) -> str:
    order = ['original','grayscale','invert_bg','clahe','final28']
    labels = ['Original','Grayscale','BG Fix','CLAHE','28×28 Input']
    fig, axes = plt.subplots(1, 5, figsize=(10, 2.4), facecolor=BG)
    for ax, key, label in zip(axes, order, labels):
        img = steps.get(key)
        if img is not None:
            ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(label, color='#aaa', fontsize=7.5, pad=4)
        ax.axis('off')
        for sp in ax.spines.values():
            sp.set_edgecolor('#2a2a55')
            sp.set_linewidth(1.2)
            sp.set_visible(True)
    fig.suptitle('Preprocessing Pipeline', color='white', fontsize=9,
                 fontweight='bold', y=1.04)
    plt.tight_layout(pad=0.4)
    return fig_to_b64(fig)


def render_gradcam(heatmap, final28) -> str:
    scale = 10
    hm = cv2.resize(heatmap, (28*scale, 28*scale))
    hm_c = cv2.applyColorMap(np.uint8(255*hm), cv2.COLORMAP_INFERNO)
    orig = cv2.resize(np.uint8(final28*255), (28*scale, 28*scale),
                      interpolation=cv2.INTER_NEAREST)
    orig_rgb = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    overlay = (hm_c * 0.55 + orig_rgb * 0.45).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), facecolor=BG)
    titles = ['Input to Model', 'Activation Map', 'Overlay (Focus Regions)']
    imgs   = [orig_rgb, cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB),
              cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)]
    cmaps  = ['gray', None, None]
    for ax, im, title in zip(axes, imgs, titles):
        ax.imshow(im)
        ax.set_title(title, color='#bbb', fontsize=8, pad=5)
        ax.axis('off')
    fig.suptitle('Grad-CAM — Where the AI Looks', color='white',
                 fontsize=10, fontweight='bold', y=1.04)
    plt.tight_layout(pad=0.5)
    return fig_to_b64(fig)


def render_saliency(saliency, final28) -> str:
    scale = 10
    sal_big = cv2.resize(saliency, (28*scale, 28*scale))
    orig_big = cv2.resize(final28, (28*scale, 28*scale),
                          interpolation=cv2.INTER_NEAREST)

    cmap_sal = LinearSegmentedColormap.from_list(
        'sal', ['#0d0d1a','#7c6fff','#e91e63','#ffeb3b'])

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), facecolor=BG)
    axes[0].imshow(orig_big, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Input', color='#bbb', fontsize=8)
    axes[0].axis('off')
    im = axes[1].imshow(sal_big, cmap=cmap_sal, vmin=0, vmax=1)
    axes[1].set_title('Pixel Saliency', color='#bbb', fontsize=8)
    axes[1].axis('off')
    cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white', fontsize=6)
    cbar.outline.set_edgecolor('#444')
    fig.suptitle('Saliency Map — Pixel-level Importance', color='white',
                 fontsize=10, fontweight='bold', y=1.04)
    plt.tight_layout(pad=0.5)
    return fig_to_b64(fig)


def render_confidence_matrix(probs: list) -> str:
    p = np.array(probs)
    short = ['T-shirt','Trouser','Pullover','Dress','Coat',
             'Sandal','Shirt','Sneaker','Bag','Boot']

    fig = plt.figure(figsize=(10, 3.8), facecolor=BG)
    gs = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1], figure=fig)

    # ── Heatmap row ───────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(BG)
    mat = p.reshape(1, 10)
    cmap = LinearSegmentedColormap.from_list('conf',['#1a0030','#7c00ff','#00e5ff','#ffff00'])
    im = ax1.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(10))
    ax1.set_xticklabels(short, rotation=38, ha='right', fontsize=8, color='#ccc')
    ax1.set_yticks([])
    for j in range(10):
        val = mat[0, j]
        color = 'black' if val > 0.6 else 'white'
        ax1.text(j, 0, f'{val:.3f}', ha='center', va='center',
                 fontsize=7.5, fontweight='bold', color=color)
    cbar = plt.colorbar(im, ax=ax1, orientation='vertical', shrink=0.85)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa', fontsize=6)
    cbar.outline.set_edgecolor('#444')
    ax1.set_title('Confidence Score per Class', color='white', fontsize=9,
                  fontweight='bold', pad=10)
    for sp in ax1.spines.values():
        sp.set_edgecolor('#333')

    # ── Radar / polar chart ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], polar=True)
    ax2.set_facecolor(BG)
    N = 10
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values = p.tolist()
    angles += angles[:1]
    values += values[:1]
    ax2.plot(angles, values, color=ACCENT, linewidth=1.8)
    ax2.fill(angles, values, color=ACCENT, alpha=0.25)
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(
        [s[:4] for s in short], fontsize=6.5, color='#bbb'
    )
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_tick_params(labelcolor='#555', labelsize=5)
    ax2.set_facecolor(BG)
    ax2.spines['polar'].set_color('#2a2a55')
    ax2.grid(color='#2a2a45', linewidth=0.6)
    ax2.set_title('Radar', color='white', fontsize=8, pad=14)

    fig.suptitle('Class Probability Analysis', color='white',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.6)
    return fig_to_b64(fig)


def render_tta_breakdown(tta_probs: list, final_probs: np.ndarray, pred_idx: int) -> str:
    """Show how each TTA variant voted."""
    aug_labels = ['Original','H-Flip','Brighter','Darker',
                  'Shift↓','Shift↑','Shift→','Shift←']
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor=BG)

    # ── Heatmap of TTA votes ─────────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG)
    mat = np.array(tta_probs)
    cmap = LinearSegmentedColormap.from_list('tta',['#0d0d1a','#1a0050','#7c00ff','#ff6b6b'])
    im = ax.imshow(mat, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    short = ['T-sh','Trou','Pull','Dres','Coat','Sand','Shrt','Snkr','Bag','Boot']
    ax.set_xticks(range(10))
    ax.set_xticklabels(short, rotation=38, ha='right', fontsize=7, color='#ccc')
    ax.set_yticks(range(8))
    ax.set_yticklabels(aug_labels, fontsize=7, color='#aaa')
    ax.set_title('TTA Augmentation Votes (8 variants)', color='white', fontsize=8.5,
                 fontweight='bold', pad=6)
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#aaa', fontsize=5.5)
    cbar.outline.set_edgecolor('#444')
    for sp in ax.spines.values(): sp.set_edgecolor('#333')

    # ── Averaged final bar chart ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG)
    colors = [CLASS_COLORS[i] for i in range(10)]
    bars = ax2.barh(range(10), final_probs * 100, color=colors, height=0.7, alpha=0.85)
    ax2.set_yticks(range(10))
    ax2.set_yticklabels(CLASS_NAMES, fontsize=7.5, color='#bbb')
    ax2.set_xlabel('Confidence %', color='#aaa', fontsize=7)
    ax2.tick_params(axis='x', colors='#555', labelsize=7)
    ax2.set_xlim(0, 105)
    ax2.set_facecolor(BG)
    for sp in ax2.spines.values(): sp.set_edgecolor('#2a2a45')
    # Highlight winning bar
    bars[pred_idx].set_edgecolor('white')
    bars[pred_idx].set_linewidth(1.8)
    ax2.text(final_probs[pred_idx]*100 + 1.5, pred_idx,
             f'{final_probs[pred_idx]*100:.1f}%', va='center',
             color='white', fontsize=7.5, fontweight='bold')
    ax2.set_title('Averaged TTA Prediction', color='white', fontsize=8.5,
                  fontweight='bold', pad=6)
    ax2.invert_yaxis()

    fig.suptitle('Test-Time Augmentation Analysis', color='white',
                 fontsize=10, fontweight='bold')
    plt.tight_layout(pad=0.6)
    return fig_to_b64(fig)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    training = os.path.exists('models/training_log.txt') and not _training_done()
    return render_template('index.html', model_loaded=model is not None,
                           training_in_progress=training)


def _training_done():
    try:
        with open('models/training_log.txt') as f:
            content = f.read()
        return 'DONE' in content
    except Exception:
        return False


@app.route('/training_status')
def training_status():
    try:
        with open('models/training_log.txt') as f:
            lines = f.readlines()
        last = [l.rstrip() for l in lines[-25:] if l.strip()]
        done = any('DONE' in l for l in lines)
        acc  = None
        for l in lines:
            if 'Test Accuracy' in l:
                try:
                    acc = float(l.split(':')[1].strip().replace('%',''))
                except Exception:
                    pass
        return jsonify({'lines': last, 'done': done, 'accuracy': acc})
    except Exception:
        return jsonify({'lines': [], 'done': False, 'accuracy': None})


@app.route('/reload_model', methods=['POST'])
def reload_model():
    load_model()
    return jsonify({'success': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided.'}), 400
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        img = Image.open(file.stream).convert('RGB')

        # Preview thumbnail
        disp = img.copy()
        disp.thumbnail((320, 320))
        buf = io.BytesIO()
        disp.save(buf, format='PNG')
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        img_array, steps, final28 = smart_preprocess(img)

        # TTA prediction
        tta_preds = []
        base = img_array[0, :, :, 0]
        augments = [
            base.copy(),
            base[:, ::-1].copy(),
            np.clip(base * 1.1, 0, 1),
            np.clip(base * 0.9, 0, 1),
            np.roll(base, 1, axis=0),
            np.roll(base, -1, axis=0),
            np.roll(base, 1, axis=1),
            np.roll(base, -1, axis=1),
        ]
        for aug in augments:
            batch = aug[np.newaxis, :, :, np.newaxis]
            tta_preds.append(model.predict(batch, verbose=0)[0])

        final_probs = np.mean(tta_preds, axis=0)
        pred_idx    = int(np.argmax(final_probs))
        confidence  = float(final_probs[pred_idx]) * 100

        # Entropy (model uncertainty)
        entropy = float(-np.sum(final_probs * np.log(final_probs + 1e-9)))
        max_entropy = float(np.log(10))
        certainty = max(0.0, 1.0 - entropy / max_entropy)

        all_probs = [
            {'class': CLASS_NAMES[i], 'emoji': CLASS_EMOJIS[i],
             'color': CLASS_COLORS[i], 'probability': float(final_probs[i]) * 100}
            for i in range(10)
        ]
        all_probs_sorted = sorted(all_probs, key=lambda x: x['probability'], reverse=True)

        # Visuals
        pipeline_b64    = render_pipeline(steps)
        conf_matrix_b64 = render_confidence_matrix(final_probs.tolist())
        tta_b64         = render_tta_breakdown(tta_preds, final_probs, pred_idx)

        heatmap = generate_gradcam(img_array, pred_idx)
        gradcam_b64 = render_gradcam(heatmap, final28.astype('float32')/255.0) \
                      if heatmap is not None else None

        saliency = generate_saliency(img_array, pred_idx)
        saliency_b64 = render_saliency(saliency, final28.astype('float32')/255.0) \
                       if saliency is not None else None

        return jsonify({
            'predicted_class': CLASS_NAMES[pred_idx],
            'emoji':           CLASS_EMOJIS[pred_idx],
            'confidence':      round(confidence, 2),
            'certainty':       round(certainty * 100, 1),
            'entropy':         round(entropy, 4),
            'all_probabilities': all_probs_sorted,
            'image_b64':         img_b64,
            'pipeline_b64':      pipeline_b64,
            'gradcam_b64':       gradcam_b64,
            'saliency_b64':      saliency_b64,
            'conf_matrix_b64':   conf_matrix_b64,
            'tta_b64':           tta_b64,
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
