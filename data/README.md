# 📦 Date — Fashion MNIST

Datele brute **nu sunt stocate în acest repository** deoarece sunt
descărcate automat de TensorFlow/Keras la prima rulare.

## Sursă
- **Dataset:** Fashion MNIST
- **Origine:** Zalando Research
- **Link oficial:** https://github.com/zalandoresearch/fashion-mnist

## Descărcare automată
\`\`\`python
import tensorflow as tf
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
\`\`\`

## Detalii
| Proprietate | Valoare |
|---|---|
| Total imagini | 70,000 |
| Antrenare | 60,000 |
| Testare | 10,000 |
| Dimensiune imagine | 28×28 px, grayscale |
| Clase | 10 |

## Clase
| Index | Denumire |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |
