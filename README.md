# 🧬 Deep Learning for Cancer Cell Classification  
### A CNN Approach to Identifying Adenocarcinoma, Benign, and Squamous Cell Carcinoma

This project implements a deep convolutional neural network (CNN) to classify histopathological images of lung cancer cells into three major types: **Adenocarcinoma**, **Benign**, and **Squamous Cell Carcinoma**.

---

## 📌 Objective

To develop a deep learning model capable of accurately classifying lung cancer cell types from microscopic images, aiding in faster and more reliable diagnostics.

---

## 🧠 Technologies & Frameworks

- **Python 3.x**
- **TensorFlow 2.x / Keras**
- **NumPy, Matplotlib, Seaborn** – for data manipulation and visualization
- **OpenCV / PIL** – for image preprocessing (if applicable)
- **Jupyter Notebook** – for experimentation and prototyping
- **Google Colab** – for training with GPU acceleration (optional)

---

## 🖼️ Dataset

The dataset consists of labeled histopathological images of lung cells classified into:
- **Adenocarcinoma**
- **Benign**
- **Squamous Cell Carcinoma**

Each image was resized to `(128, 128, 3)` for compatibility with the CNN model.

---

## 🏗️ CNN Model Architecture

```python
model = keras.Sequential()
model.add(layers.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=1, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(3, activation='softmax'))
