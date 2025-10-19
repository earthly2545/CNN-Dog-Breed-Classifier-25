
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

MODEL_PATH = './models/dog_breed_120_classifier_model.h5'
DATA_DIR_BASE = './data/'
dataset_path = os.path.join(DATA_DIR_BASE, 'dog-breeds')
DATA_DIR = dataset_path

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32 

if not os.path.exists(MODEL_PATH) or not os.path.exists(DATA_DIR):
    print(f"!! [Error] ไม่พบไฟล์โมเดลที่ '{MODEL_PATH}' หรือโฟลเดอร์ข้อมูลที่ '{DATA_DIR}'")
    print("!! กรุณาตรวจสอบว่าคุณได้เทรนโมเดลและเตรียมข้อมูลเรียบร้อยแล้ว")
    exit()

print(f">> กำลังโหลดโมเดลจาก: '{MODEL_PATH}'...")
model = tf.keras.models.load_model(MODEL_PATH)
print(">> โหลดโมเดลสำเร็จ!")


validation_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2 
)

print(">> กำลังเตรียม Validation data สำหรับการประเมินผล...")
validation_generator = validation_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation', 
    shuffle=False      
)

print(">> กำลังทำนายผลจาก Validation data ทั้งหมด (อาจใช้เวลาสักครู่)...")
steps = int (np.ceil(validation_generator.samples / validation_generator.batch_size))
predictions = model.predict(validation_generator, steps=steps)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

print("\n" + "="*50)
print("--- Classification Report ---")
print("="*50)
report = classification_report(true_classes, predicted_classes, target_names=class_labels)
print(report)

report_path = './outputs/classification_report.txt'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(report)
print(f"\n>> บันทึก Classification Report ที่: '{report_path}' เรียบร้อยแล้ว")

print("\n--- กำลังสร้าง Confusion Matrix ---")
cm = confusion_matrix(true_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)

fig, ax = plt.subplots(figsize=(25, 25)) 
disp.plot(ax=ax, xticks_rotation='vertical', cmap='Blues')

plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('./outputs/confusion_matrix.png')
print(">> บันทึก Confusion Matrix ที่ './outputs/confusion_matrix.png' เรียบร้อยแล้ว")
plt.show()

print("\n>> การประเมินผลเสร็จสมบูรณ์ <<")