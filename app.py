import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# --- 1. DEFINIREA ARHITECTURII (Necesară pentru a încărca modelul) ---
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

# --- 2. ÎNCĂRCAREA MODELULUI ---
print("Se încarcă modelul AI... te rog așteaptă.")
try:
    model = tf.keras.models.load_model('models/resnet_model.keras', custom_objects={'ResidualBlock': ResidualBlock})
    print("Model încărcat cu succes!")
except Exception as e:
    print(f"Eroare la încărcarea modelului: {e}")
    exit()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# --- 3. LOGICA INTERFEȚEI GRAFICE (UI) ---
def process_and_predict(file_path):
    try:
        # Deschidem imaginea folosind Pillow
        img = Image.open(file_path)
        
        # Afișăm imaginea originală pe interfață
        img_display = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(img_display)
        label_image.configure(image=img_tk)
        label_image.image = img_tk
        
        # PRE-PROCESAREA PENTRU AI
        # Fashion MNIST se așteaptă la o imagine 28x28, alb-negru (haină albă, fundal negru)
        img_gray = img.convert('L') # Convertim în nuanțe de gri
        img_resized = img_gray.resize((28, 28))
        
        # Inversăm culorile dacă fundalul este alb (majoritatea pozelor descărcate de pe net au fundal alb)
        # Modelul este antrenat pe fundal negru!
        img_inverted = ImageOps.invert(img_resized)
        
        # Convertim în formatul pe care îl înțelege matematica rețelei
        img_array = np.array(img_inverted)
        img_array = img_array.astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=-1)
        img_array = np.expand_dims(img_array, axis=0) # Forma finală: (1, 28, 28, 1)
        
        # PREDICȚIA
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        
        # Actualizăm textul de pe ecran
        label_result.configure(text=f"Detecție: {predicted_class}\nÎncredere: {confidence:.2f}%", fg="#00b300")
        
    except Exception as e:
        messagebox.showerror("Eroare", f"Nu s-a putut procesa imaginea:\n{e}")

def upload_image():
    # Deschide fereastra de Windows pentru a selecta o poză
    file_path = filedialog.askopenfilename(
        title="Alege o imagine vestimentară",
        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")]
    )
    if file_path:
        label_result.configure(text="Se analizează...", fg="blue")
        # Chemăm funcția de predicție
        root.after(100, process_and_predict, file_path)

# --- 4. CONSTRUIREA FERESTREI PRINCIPALE ---
root = tk.Tk()
root.title("Fashion AI Detector")
root.geometry("400x550")
root.configure(bg="#f0f0f0")

# Titlul aplicației
title_label = tk.Label(root, text="Inteligență Artificială\nRecunoaștere Vestimentară", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
title_label.pack(pady=20)

# Butonul de Upload
btn_upload = tk.Button(root, text="Încarcă Imagine", font=("Helvetica", 14), bg="#0066cc", fg="white", cursor="hand2", command=upload_image)
btn_upload.pack(pady=10)

# Locul unde va apărea imaginea (inițial gol)
label_image = tk.Label(root, bg="#f0f0f0")
label_image.pack(pady=20)

# Locul unde va apărea predicția (rezultatul)
label_result = tk.Label(root, text="Încarcă o imagine pentru a începe.", font=("Helvetica", 14), bg="#f0f0f0")
label_result.pack(pady=20)

# Pornește aplicația
root.mainloop()
