from keras.layers import Dense, GlobalAveragePooling2D
from keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import ResNet50
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import os

# Image dimensions
img_height, img_width = (96, 96)
batch_size = 32

# Data directories
train_data_dir = "Random3/train"
val_data_dir = "Random3/train"  # using training set with subset for validation

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% of training set used as validation
)

# Training generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

# Validation generator
val_generator = train_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Load base ResNet50 model without top layer
base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(img_height, img_width, 3))

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback to save best model based on validation accuracy
os.makedirs("model", exist_ok=True)
checkpoint = ModelCheckpoint(
    filepath="model/ASL_ResNet50_Best.h5",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# Train model for 20 epochs
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[checkpoint]
)

# Save final model after 20 epochs
model.save("model/ASL_ResNet50_Final.h5")
print("✅ Model training completed and saved.")

