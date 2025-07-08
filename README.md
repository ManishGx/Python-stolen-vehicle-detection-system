# Automatic Number Plate Recognition (ANPR) with Stolen Vehicle Alert

This project uses OpenCV and Tesseract OCR to detect vehicle license plates from a webcam feed, extract the plate numbers, and alert the user if a detected plate matches any in a list of stolen vehicle numbers. If no match is found, it displays "No stolen vehicle found" on the video feed.

## Features

- Real-time license plate detection using Haar cascades.
- Robust text extraction from plates using Tesseract OCR with advanced preprocessing.
- Fuzzy matching to account for minor OCR errors.
- Desktop notification alert when a stolen vehicle is detected.
- Visual feedback on the video feed, including detected plate numbers and status messages.
- Cooldown mechanism to prevent notification spam.

## Requirements

- Python 3.7+
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (install and note the path)
- OpenCV (`opencv-python`)
- pytesseract
- plyer
- difflib (standard library)
- re (standard library)

## Installation

1. **Install Tesseract OCR**  
   Download and install from [here](https://github.com/tesseract-ocr/tesseract).  
   Note the installation path (e.g., `C:\Program Files\Tesseract-OCR\tesseract.exe`).

2. **Install Python dependencies**  
   ```
   pip install opencv-python pytesseract plyer
   ```

3. **Clone or download this repository.**

## Usage

1. **Edit the list of stolen car numbers**  
   Open `main2.py` and add your stolen vehicle numbers to the `stolen_car_numbers` list.

2. **Set the Tesseract path**  
   Make sure the `pytesseract.pytesseract.tesseract_cmd` variable points to your Tesseract executable.

3. **Run the script**  
   ```
   python main2.py
   ```

4. **Operation**  
   - The webcam feed will open.
   - Detected plate numbers will be shown on the video.
   - If a plate matches a stolen number (with fuzzy matching), a desktop notification will appear.
   - If no match is found, "No stolen vehicle found" will be displayed below the detected plate.

5. **Exit**  
   - Press `q` to quit the application.

## Customization

- **Adjust Fuzzy Matching:**  
  Change the `similarity_threshold` variable in `main2.py` for stricter or looser matching.

- **Change Cooldown:**  
  Adjust the `alert_cooldown` variable to set how often notifications can appear for the same plate.

## Notes

- Detection accuracy depends on camera quality, lighting, and plate clarity.
- The Haar cascade used is for Russian plates; for better results with other regions, consider training or using a more suitable cascade.

## Example

![Example Screenshot](screenshot.png)

## License

This project is for educational purposes.