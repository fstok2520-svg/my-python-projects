import cv2
import numpy as np
from ultralytics import YOLO

try:
    model = YOLO("yolov8n.pt")  
    print("تم تحميل نموذج YOLOv8 بنجاح! جاري تشغيل النظام.")
except Exception as e:
    print(f"خطأ في تحميل النموذج: {e}. يرجى التأكد من تثبيت مكتبة 'ultralytics' واتصال الإنترنت.")
    exit()

def get_object_coordinates(box):
    coords = box.xyxy[0].cpu().numpy() # تحويل الإحداثيات لـ numpy array
    return [int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])]

# --- 2. تشغيل الكاميرا ---
cap = cv2.VideoCapture(0) # 0 للكاميرا المدمجة

if not cap.isOpened():
    print("خطأ: تعذر فتح الكاميرا.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("لم يتم استقبال أي صورة من الكاميرا، جاري إعادة المحاولة.")
        continue

    results = model(image, stream=True)

    # تهيئة المتغيرات
    person_count = 0
    phone_detected = False
    suspicious_objects = []
  
    for result in results:
        for box in result.boxes:
            # استخراج الإحداثيات بالطريقة الآمنة
            x1, y1, x2, y2 = get_object_coordinates(box)
            
            # فئة الكائن ودقة الكشف
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]
            conf = float(box.conf[0])

          
            if class_name == 'person' and conf > 0.5:
                person_count += 1
                color = (255, 0, 0) # أزرق للأشخاص
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, f"{class_name} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            
            if class_name == 'cell phone' and conf > 0.6: 
                phone_detected = True
                color = (0, 0, 255) # أحمر للتحذير
                suspicious_objects.append([x1, y1, x2, y2, class_name, conf])
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 3) 
                cv2.putText(image, f"{class_name} {conf:.2f} (SUSPICIOUS)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(image, f"Examinees Detected: {person_count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    if phone_detected:
        status_text = "CELL PHONE DETECTED! Possible Cheating."
        status_color = (0, 0, 255) # أحمر
    else:
        status_text = "Monitor: Normal (No phone visible)."
        status_color = (0, 255, 0) # أخضر
        
    cv2.putText(image, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

    
    cv2.imshow('Exam Proctoring System - YOLOv8 FINAL', image)

   
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# تحرير الموارد وإغلاق النوافذ
cap.release()
cv2.destroyAllWindows()
