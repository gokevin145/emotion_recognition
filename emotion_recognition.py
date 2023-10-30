import cv2
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
EMOTION_LABELS = {
    'angry': '生氣',
    'disgust': '噁心',
    'fear': '害怕',
    'happy': '開心',
    'sad': '難過',
    'surprise': '驚訝',
    'neutral': '正常'
}
def load_font(size=50):
    font_path = r'C:\WINDOWS\Fonts\MSYHL.TTC'
    return ImageFont.truetype(font_path, size)
def display_emotion_text(image, x, y, text, font, color=(0, 0, 0)):
    img_pil = Image.fromarray(image)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x, y), EMOTION_LABELS[text], fill=color, font=font)
    return np.array(img_pil)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法啟動攝影機")
        exit()
    font = load_font()
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法取得影格")
            break
        img = cv2.resize(frame, (384, 240))
        try:
            analysis = DeepFace.analyze(img, actions=['emotion'])
            emotion = analysis[0]['dominant_emotion']
            print(f"Detected emotion: {emotion}")
            img = display_emotion_text(img, 0, 40, emotion, font)
        except Exception as e:
            print(f"Error during emotion analysis: {e}")
            pass
        cv2.imshow('oxxostudio', img)
        if cv2.waitKey(5) == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
