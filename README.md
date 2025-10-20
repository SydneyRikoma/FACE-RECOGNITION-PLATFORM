 # FACE RECOGNITION PLATFORM

---

A lightweight, local face recognition system built with Python and OpenCV. Designed for simplicity and speed, this CLI-based tool allows you to:


1. Register a face by name  — captures 200 images and retrains the model from scratch  

2. Recognize faces live — identifies known faces and labels unknown ones



No database, no web interface — just a clean folder-based dataset and a reliable LBPH model.



---



## FEATURES

\- Folder-based dataset (`data/`) with one folder per person

\- LBPH model retrained after each registration

\- Simple CLI menu with just two options

\- Confidence threshold set to 70 for recognition

\- Works entirely offline






## SETUP



Install dependencies:

```bash

pip install opencv-python opencv-contrib-python numpy

```



---



## USAGE



Run the script:

```bash

python main.py

```



You'll see a menu:

```

1\. Register a new face

2\. Recognize faces

3\. Exit

```



---



## FOLDER STRUCTURE



```

FACE RECOGNITION PROJECT/

├── main.py

├── data/              # auto-created folders per person

├── classifier.xml     # trained model

├── labels.npy         # label-to-name mapping

```



---



## NOTES

\- Press `q` to quit during capture or recognition

\- Model and label map are regenerated after each registration

\- Unknown faces are labeled as "Unknown" and skipped



---



## COMPATIBILITY



\- Python: 3.7+

\- Operating System: Windows, Linux, macOS (requires webcam and OpenCV support)







## 👤 AUTHOR



Created by Sydney Panashe Rikoma (https://github.com/SydneyRikoma)  

Feel free to fork, modify, or contribute!



---



## 📄 LiCENSE



This project is open-source under the MIT License. See `LICENSE` file for details.

