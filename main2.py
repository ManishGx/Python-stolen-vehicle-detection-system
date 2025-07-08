import cv2
import pytesseract
from plyer import notification
import time
import re
import difflib

# List of stolen car numbers (add as many as you want)
stolen_car_numbers = [ "CH04E3760","KA64N0099","21BH2345AA","DL7CQ1939"]

# Optional: set Tesseract path if not in system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Haar cascade
cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
plate_cascade = cv2.CascadeClassifier(cascade_path)
if plate_cascade.empty():
    raise IOError(f"Failed to load cascade from {cascade_path}")

def alert_owner(detected_text, matched_plate):
    notification.notify(
        title='Stolen Car Detected!',
        message=f'Detected: {detected_text}\nMatched: {matched_plate}',
        timeout=10
    )

def clean_plate_text(text):
    # Remove non-alphanumeric and convert to uppercase
    return re.sub(r'[^A-Z0-9]', '', text.upper())

def extract_plate_text(plate_img):
    """Extracts text from a plate image using OCR with advanced preprocessing."""
    plate_gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.resize(plate_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    plate_gray = cv2.bilateralFilter(plate_gray, 11, 17, 17)

    # CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    plate_clahe = clahe.apply(plate_gray)

    # Adaptive thresholding
    plate_thresh = cv2.adaptiveThreshold(
        plate_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    plate_thresh = cv2.morphologyEx(plate_thresh, cv2.MORPH_CLOSE, kernel)

    # Try multiple images and PSM modes
    ocr_results = []
    for img in [plate_gray, plate_clahe, plate_thresh]:
        for psm in [6, 7, 8]:
            config = f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            text = pytesseract.image_to_string(img, config=config)
            cleaned = clean_plate_text(text)
            if len(cleaned) >= 6:
                ocr_results.append(cleaned)

    # Return the most frequent or longest result
    if ocr_results:
        from collections import Counter
        return Counter(ocr_results).most_common(1)[0][0]
    return ""

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Starting license plate monitoring... Press 'q' to exit.")

last_alert_times = {plate: 0 for plate in stolen_car_numbers}
alert_cooldown = 10  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    for (x, y, w, h) in plates:
        pad = 10
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, frame.shape[1])
        y2 = min(y + h + pad, frame.shape[0])
        plate_img = frame[y1:y2, x1:x2]

        detected_text = extract_plate_text(plate_img)
        print("Detected:", detected_text)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, detected_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        current_time = time.time()
        found_stolen = False
        similarity_threshold = 0.85  # Adjust as needed

        for stolen_plate in stolen_car_numbers:
            # Exact match or high similarity
            ratio = difflib.SequenceMatcher(None, detected_text, stolen_plate).ratio()
            if detected_text == stolen_plate or ratio >= similarity_threshold:
                found_stolen = True
                if current_time - last_alert_times[stolen_plate] > alert_cooldown:
                    alert_owner(detected_text, stolen_plate)
                    last_alert_times[stolen_plate] = current_time

        if not found_stolen and len(detected_text) >= 6:
            cv2.putText(frame, "No stolen vehicle found", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

    cv2.imshow("Live Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()