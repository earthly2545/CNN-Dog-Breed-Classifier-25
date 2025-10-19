
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import tensorflow as tf
import json
MODEL_PATH = './models/dog_breed_25_classifier_model.h5'
CLASS_INDICES_PATH = './models/dog_breed_25_class_indices.json'
IMAGE_SIZE = (224, 224)
def load_model_and_classes():
    global model, class_names
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        class_names = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]
        
        num_breeds = len(class_names)
        app.title(f"โปรแกรมจำแนกสายพันธุ์สุนัข ({num_breeds} สายพันธุ์)")
        lbl_info.configure(text=f"รองรับการจำแนกทั้งหมด {num_breeds} สายพันธุ์ (คลิกเพื่อดูรายชื่อ)")
        
        print("โหลดโมเดลและคลาสสำเร็จ!")
    except Exception as e:
        model, class_names = None, None
        lbl_result.configure(text=f"Error: ไม่สามารถโหลดโมเดลได้", text_color="red")
        lbl_info.configure(text="ไม่สามารถโหลดข้อมูลสายพันธุ์ได้", text_color="red", cursor="arrow")
def show_breed_list(): 
    """สร้างและแสดงหน้าต่างใหม่ที่มีรายชื่อสายพันธุ์ทั้งหมด"""
    if not class_names:
        return     
    list_window = ctk.CTkToplevel(app)
    list_window.title("รายชื่อสายพันธุ์ที่รองรับ")
    list_window.geometry("400x500")
    list_window.transient(app)
    list_window.grab_set()
    scrollable_frame = ctk.CTkScrollableFrame(list_window, label_text=f"ทั้งหมด {len(class_names)} สายพันธุ์")
    scrollable_frame.pack(fill="both", expand=True, padx=10, pady=10)

    for breed in class_names:
        formatted_breed = breed.replace('_', ' ').title()
        breed_label = ctk.CTkLabel(scrollable_frame, text=f"• {formatted_breed}", anchor="w", justify="left")
        breed_label.pack(fill="x", padx=10, pady=2)
def predict_image(img_array):
    if model is None: return "Error", 0.0
    
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = img_batch / 255.0
    prediction = model.predict(img_preprocessed)
    confidence = np.max(prediction[0]) * 100
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name, confidence
def select_and_predict_image():
    file_path = filedialog.askopenfilename(
        title="เลือกไฟล์รูปภาพ",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
    )
    if not file_path:
        return
    if model is None:
        lbl_result.configure(text="กรุณารอสักครู่... กำลังโหลดโมเดล", text_color="orange")
        return 
    img_display = Image.open(file_path).convert("RGB")
    ctk_image = ctk.CTkImage(light_image=img_display, dark_image=img_display, size=(300, 300))
    lbl_image.configure(image=ctk_image, text="")
    img_for_model = img_display.resize(IMAGE_SIZE)
    img_array = np.array(img_for_model)
    lbl_result.configure(text="กำลังจำแนก...", text_color="orange")
    app.update_idletasks()
    class_name, confidence = predict_image(img_array)
    lbl_result.configure(text=f"ผลการจำแนก: {class_name}\nความมั่นใจ: {confidence:.2f}%", text_color=("blue", "cyan"))
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("โปรแกรมจำแนกสายพันธุ์สุนัข")
app.geometry("500x550")
app.resizable(False, False)
lbl_title = ctk.CTkLabel(app, text="เลือกรูปภาพเพื่อจำแนก", font=ctk.CTkFont(size=20, weight="bold"))
lbl_info = ctk.CTkLabel(app, text="กำลังโหลดข้อมูลสายพันธุ์...", font=ctk.CTkFont(size=12, underline=True), text_color="gray", cursor="hand2")
lbl_info.bind("<Button-1>", lambda event: show_breed_list())
btn_select = ctk.CTkButton(app, text="อัปโหลดรูปภาพ", command=select_and_predict_image)
lbl_image = ctk.CTkLabel(app, text="ยังไม่ได้เลือกรูปภาพ", height=300, width=300, corner_radius=10, fg_color=("gray90", "gray20"))
lbl_result = ctk.CTkLabel(app, text="ผลลัพธ์จะแสดงที่นี่", font=ctk.CTkFont(size=18), height=50)
lbl_title.pack(pady=(20, 5))
lbl_info.pack(pady=(0, 20))
btn_select.pack(pady=10)
lbl_image.pack(pady=10)
lbl_result.pack(pady=(10, 20))

model, class_names = None, None
app.after(10, load_model_and_classes)
app.mainloop()