
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import json

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

DATA_DIR_BASE = './data/'
MODEL_SAVE_PATH = './models/dog_breed_25_classifier_model.h5' 
OUTPUT_GRAPH_PATH = './outputs/dog_breed_25_training_history.png' 
CLASS_INDICES_PATH = './models/dog_breed_25_class_indices.json' 

os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_GRAPH_PATH), exist_ok=True)


dataset_path = os.path.join(DATA_DIR_BASE, 'dog-breeds')

print(">> กำลังอ่านข้อมูลจากโฟลเดอร์ที่เตรียมไว้...")
if not os.path.exists(dataset_path):
    print(f"!! [Error] ไม่พบโฟลเดอร์ข้อมูลที่: '{dataset_path}'")
    print("!! กรุณาตรวจสอบว่าคุณได้วางโฟลเดอร์ 'dog-breeds' ไว้ในโฟลเดอร์ 'data' ถูกต้องแล้ว")
    exit() 

DATA_DIR = dataset_path

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

print(">> กำลังเตรียมข้อมูลสำหรับ Training...")
train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)
print(">> กำลังเตรียมข้อมูลสำหรับ Validation...")
validation_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

if os.path.exists(MODEL_SAVE_PATH):
    print(f">> กำลังโหลดโมเดลที่มีอยู่จาก: '{MODEL_SAVE_PATH}' เพื่อทำการฝึกสอนต่อ...")
    model = load_model(MODEL_SAVE_PATH)
else:
    print(">> ไม่พบโมเดลเดิม, กำลังสร้างโมเดลใหม่ด้วย Transfer Learning...")
    NUM_CLASSES = len(train_generator.class_indices)
    print(f">> จำนวนคลาสที่พบ: {NUM_CLASSES}")

    inputs = Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    base_model = MobileNetV2(input_tensor=inputs, include_top=False, weights='imagenet')
    base_model.trainable = False 

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(NUM_CLASSES, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

# เฟสที่ 1: Feature Extraction
print("\n--- เฟสที่ 1: Feature Extraction ---")
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# กำหนด Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=7, verbose=1, restore_best_weights=True)

EPOCHS_INITIAL = 50
history = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping]
)

# เฟสที่ 2: Fine-Tuning
print("\n--- เฟสที่ 2: Fine-Tuning ---")
base_model.trainable = True
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

EPOCHS_FINE_TUNE = 20
initial_epoch_fine_tune = 0
if history.epoch:
    initial_epoch_fine_tune = history.epoch[-1] + 1

history_fine = model.fit(
    train_generator,
    epochs=initial_epoch_fine_tune + EPOCHS_FINE_TUNE, 
    initial_epoch=initial_epoch_fine_tune, 
    validation_data=validation_generator,
    callbacks=[reduce_lr, early_stopping]
)

model.save(MODEL_SAVE_PATH)
class_indices = train_generator.class_indices
with open(CLASS_INDICES_PATH, 'w') as f:
    json.dump(class_indices, f, indent=4)
print(f">> บันทึกโมเดลและ Class Indices เรียบร้อย")

acc = history.history['accuracy'] + history_fine.history.get('accuracy', [])
val_acc = history.history['val_accuracy'] + history_fine.history.get('val_accuracy', [])
loss = history.history['loss'] + history_fine.history.get('loss', [])
val_loss = history.history['val_loss'] + history_fine.history.get('val_loss', [])

epochs_range = range(len(acc)) 

plt.figure(figsize=(16, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
if history.epoch: 
    plt.axvline(history.epoch[-1], color='r', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
if history.epoch:
    plt.axvline(history.epoch[-1], color='r', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss') 

plt.savefig(OUTPUT_GRAPH_PATH)
print(f">> บันทึกกราฟผลการฝึกสอนที่: {OUTPUT_GRAPH_PATH}")
plt.show()

print(">> กระบวนการฝึกสอนเสร็จสมบูรณ์ <<")

