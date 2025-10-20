import os
import cv2
import numpy as np

# Paths and parameters
DATA_DIR = "data"
MODEL_FILE = "classifier.xml"
SAMPLES_PER_PERSON = 200
CONF_THRESHOLD = 70

# Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def capture_and_train():
    name = input("Enter name to register: ").strip()
    if not name:
        print("❌ Invalid name.")
        return

    person_dir = os.path.join(DATA_DIR, name)
    os.makedirs(person_dir, exist_ok=True)
    print(f"📷 Capturing {SAMPLES_PER_PERSON} images for '{name}'. Press 'q' to quit early.")

    cap = cv2.VideoCapture(0)
    count = 0
    while count < SAMPLES_PER_PERSON:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not detected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]
            cv2.imwrite(f"{person_dir}/{name}_{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)

        cv2.imshow("Register (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Collected {count} images. Training model...")

    # Prepare training data
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, labels, label_map = [], [], {}
    label_id = 0

    for person in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person)
        if not os.path.isdir(person_path):
            continue
        label_map[label_id] = person
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                continue
            faces.append(gray)
            labels.append(label_id)
        label_id += 1

    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_FILE)

    # Save label mapping
    with open("labels.npy", "wb") as f:
        np.save(f, label_map)
    print("✅ Model trained and saved.")

def recognize_faces():
    if not os.path.exists(MODEL_FILE) or not os.path.exists("labels.npy"):
        print("❌ Model or label map not found. Register someone first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)
    label_map = np.load("labels.npy", allow_pickle=True).item()

    cap = cv2.VideoCapture(0)
    print("📷 Starting recognition. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Camera not detected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label_id, conf = recognizer.predict(face_img)
            name = label_map[label_id] if conf < CONF_THRESHOLD else "Unknown"
            color = (0,255,0) if name != "Unknown" else (0,0,255)

            cv2.putText(frame, name, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        cv2.imshow("Recognize (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Session ended.")

def main():
    menu = """
    ===== Face Recognition System =====
    1. Register a new face
    2. Recognize faces
    3. Exit
    ===================================
    """
    while True:
        print(menu)
        choice = input("Select [1-3]: ").strip()
        if choice == "1":
            capture_and_train()
        elif choice == "2":
            recognize_faces()
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("❌ Invalid choice.")

if __name__ == "__main__":
    main()