# 🖍️ Handwritten Digit Recognition System  

This project allows users to **recognize handwritten digits** using a **deep learning model (CNN)** trained on the **MNIST dataset**.  
Users can either:  
✔️ **Upload an image** of a digit  
✔️ **Draw on a writing pad** and get instant predictions  

---

## ✨ Features  

✅ **Image Upload** – Choose an image of a digit and get a prediction  
✅ **Interactive Writing Pad** – Draw a digit using a mouse  
✅ **Clear Canvas (Press 'C')** – Erase and redraw without restarting  
✅ **Predict Digit (Press 'P')** – Get the result instantly  
✅ **Quit (Press 'Q')** – Exit the writing pad  

---

## 🛠️ Installation  

1. **Clone the repository:**  
   ```sh
   git clone https://github.com/your-username/handwritten-digit-recognition.git
   cd handwritten-digit-recognition
   ```

2. **Install dependencies:**  
   ```sh
   pip install opencv-python numpy tensorflow tkinter
   ```

3. **Download the trained model:**  
   - Place the pre-trained **handwriting_model.h5** inside the `models/` directory.  

---

## 🚀 Usage  

Run the program using:  
```sh
python main.py
```

You’ll see a **menu** asking you to:  
1️⃣ Upload an image  
2️⃣ Draw on a writing pad  

**For Image Upload:**  
- Select an image (`.png, .jpg, .jpeg`) containing a digit.  
- The system will process and display the predicted digit.  

**For Writing Pad:**  
- **Draw a digit** using your mouse.  
- **Press 'C'** to clear and redraw.  
- **Press 'P'** to predict the drawn digit.  
- **Press 'Q'** to exit.  

---

## 📂 Project Structure  

```
📂 handwritten-digit-recognition
│── 📂 models
│   └── handwriting_model.h5       # Pre-trained CNN model
│── digit_recognition.py           # Main program to Train & Run the Model
│── realtime_digit_recognition.py  # Program to Acces the Local File & Writting Pad
│── README.md                      # Project documentation
```

---

## 🧠 How It Works  

- **Image Processing:** Converts images to **grayscale, resizes (28x28), normalizes**, and inverts colors.  
- **CNN Model Prediction:** Trained on MNIST dataset, predicts digit with **softmax classification**.  
- **OpenCV GUI:** Provides an interactive writing pad for drawing digits.  

---

## 🖥️ Demo  

### **📸 Screenshot 2: Writing Pad in Action**
![Screenshot 2025-02-15 174001](https://github.com/user-attachments/assets/4e3767d4-b468-4722-b1ae-2686dd537385)


### **📸 Screenshot 3: Image Upload Prediction**
![Screenshot 2025-02-15 173810](https://github.com/user-attachments/assets/cb75d207-64df-4f1f-aaf5-6588f2730c58)

---

## 🤝 Contributing  

Want to improve this project? Feel free to fork and submit a pull request!  

---

## 📜 License  

This project is **open-source** and free to use! 🚀 
