import cv2
import numpy as np
from skimage import feature
from skimage.color import rgb2gray

# مرحله 1: ضبط تصویر از دوربین
def capture_image():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite("face.jpg", frame)
        return frame
    else:
        raise Exception("Failed to capture image")

# مرحله 2: تشخیص چهره با استفاده از Haar Cascade
def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        return faces[0]  # فقط اولین چهره را برمی‌گردانیم
    else:
        raise Exception("No face detected")

# مرحله 3: تحلیل رنگ پوست
def analyze_skin_color(face_img):
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    mean_hue = np.mean(h)
    mean_saturation = np.mean(s)
    mean_value = np.mean(v)

    skin_info = {
        "mean_hue": mean_hue,
        "mean_saturation": mean_saturation,
        "mean_value": mean_value
    }

    return skin_info

# مرحله 4: تحلیل جوش‌ها
def analyze_acne(face_img):
    gray_face = rgb2gray(face_img)
    edges = feature.canny(gray_face, sigma=3)
    acne_count = np.sum(edges)

    return acne_count

# مرحله 5: تخمین خشک بودن پوست
def estimate_skin_dryness(face_img):
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()

    dryness_level = "Unknown"
    if laplacian_var < 50:
        dryness_level = "Dry"
    elif laplacian_var >= 50 and laplacian_var <= 100:
        dryness_level = "Normal"
    else:
        dryness_level = "Oily"

    return dryness_level

# تبدیل hue به رنگ پوست قابل فهم
def get_skin_color_name(mean_hue, mean_saturation, mean_value):
    if mean_value > 200:
        return "White"
    elif mean_hue < 15 or mean_hue >= 165:
        if mean_saturation < 50 and mean_value < 128:
            return "Black"
        else:
            return "Brown"
    elif 15 <= mean_hue < 45:
        return "Yellow"
    else:
        return "Brown"

# اجرای مراحل
try:
    image = capture_image()
    face_rect = detect_face(image)
    x, y, w, h = face_rect
    face_img = image[y:y+h, x:x+w]

    skin_color_info = analyze_skin_color(face_img)
    acne_count = analyze_acne(face_img)
    dryness_level = estimate_skin_dryness(face_img)

    mean_hue = skin_color_info['mean_hue']
    mean_saturation = skin_color_info['mean_saturation']
    mean_value = skin_color_info['mean_value']

    skin_color = get_skin_color_name(mean_hue, mean_saturation, mean_value)

    print("Skin Color:", skin_color)
    if acne_count > 1000:
        print("صورت پر جوشی دارید")
    else:
        print("صورت پر جوشی نیست")
    print("Dryness Level:", dryness_level)

except Exception as e:
    print(str(e))
