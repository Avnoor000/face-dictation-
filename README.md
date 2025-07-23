# face-dictation-

A Python-based project that includes:

A face recognition model trainer using OpenCV and the LBPH algorithm.

A real-time webcam utility with edge detection using the Canny algorithm.

This suite demonstrates core computer vision capabilities using OpenCV in an easy-to-understand way.

📁 Project Structure
bash
Copy
Edit
ai-vision-suite/
├── data/                         # Face images dataset
│   ├── Dwayne Johnson/
│   ├── Jacqueline Fernandez/
│   ├── Janhvi Kapoor/
│   ├── Mitchell Starc/
│   └── Tom Holland/
├── haar_face.xml                # Haar cascade for face detection
├── face_trainer.py              # Face recognition model trainer
├── webcam_edge_detection.py     # Webcam + edge detection utility
├── face_t.yml                   # Trained model output (generated after training)
├── features.npy                 # Saved features from training
├── labels.npy                   # Saved labels
├── README.md                    # Project documentation
📄 File Descriptions
🧠 face_trainer.py
Trains a facial recognition model using OpenCV’s LBPH (Local Binary Patterns Histograms) algorithm.

Workflow:

Loads grayscale face images from labeled folders.

Detects faces using Haar Cascade.

Extracts face regions and labels them.

Trains and saves a model.

How to Run:

bash
Copy
Edit
python face_trainer.py
Output Files:

face_t.yml — Trained model.

features.npy — Numpy array of face data.

labels.npy — Numpy array of labels.

📷 webcam_edge_detection.py
Captures live webcam video, displays both the original and resized views, and performs edge detection when prompted.

Features:

Real-time video display with OpenCV.

Resized preview window.

Canny edge detection triggered with key press.

How to Run:

bash
Copy
Edit
python webcam_edge_detection.py
How to Use:

Press d while the video window is active to capture and apply edge detection.

Closes webcam stream on key press.

📦 Installation
Install required Python packages:

bash
Copy
Edit
pip install opencv-python opencv-contrib-python numpy matplotlib
📁 Dataset Structure
Each person should have their own folder inside the data/ directory:

python-repl
Copy
Edit
data/
├── Dwayne Johnson/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
├── Jacqueline Fernandez/
│   └── ...
...
Ensure each image clearly shows the person's face.

🔒 License
MIT License

👨‍💻 Author
Your Name
Computer Vision Enthusiast
Open for collaboration!
