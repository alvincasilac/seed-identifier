import tensorflow as tf
import os

img_size = 128
batch_size = 16

dataset_path = "dataset"

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    image_size=(img_size, img_size),
    batch_size=batch_size
)

class_names = train_ds.class_names

train_ds = train_ds.map(lambda x, y: (x/255.0, y))

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 3)),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_ds, epochs=10)

model.save("model/seed_model.h5")

print("Model saved!")
print("Classes:", class_names)