<h1 align="center">🧠 Two Digit Recognition using Convolutional Neural Network (CNN)</h1>

<p align="center">
  <b>Author:</b> Dr. Poulami Nandi | 
  <a href="https://www.linkedin.com/in/poulami-nandi-a8a12917b/">LinkedIn</a> | 
  <a href="https://scholar.google.com/citations?user=KKlKH0kAAAAJ&hl=en">Google Scholar</a>
</p>

---

## 📌 Project Overview  
This project develops a robust Convolutional Neural Network (CNN) to recognize **two handwritten digits** from a single image. Unlike the standard MNIST challenge, each image contains two digits, turning this into a **multi-label, multi-output classification** problem. The CNN is architected with a shared convolutional base and separate output heads to simultaneously predict both digits.

---

## 🧪 Objective  
- Build a PyTorch-based CNN capable of dual-digit classification.  
- Optimize the model for balanced accuracy across both digit positions.  
- Handle data preprocessing, training, evaluation, and submission generation.

---

## 🧰 Tools & Technologies Used  

| Category               | Tools & Libraries                                                                 |
|------------------------|------------------------------------------------------------------------------------|
| **Programming**        | Python 3.8+                                                                        |
| **Deep Learning**      | PyTorch, Torchvision                                                               |
| **Data Handling**      | Pandas, NumPy                                                                      |
| **Visualization**      | Matplotlib, Seaborn                                                                |
| **Training/Evaluation**| CrossEntropyLoss, Adam Optimizer, custom DatasetLoader                            |
| **Deployment Ready**   | Jupyter Notebook, CSV Submission, Image Output Preview                            |

---

## 📂 Project Structure  
```bash
twoDigRecognizeCNN/
├── assets/
│   ├── training_curves.png
│   ├── sample_output_1.png
├── LICENSE
├── README.md
├── digit-recognizer.zip
├── submission_2d.csv
├── twoDigRecognizeCNN.ipynb
├── twoDigitRecogCNN_model.h5
└── twoDigitRecogCNN_model_trainingHist.pkl

```

---
## 🧠 Model Architecture  

- **Shared Convolutional Backbone**: 3 blocks of Conv2D → ReLU → MaxPool  
- **Regularization**: Dropout layers after conv and dense blocks  
- **Two Dense Heads**: Each outputs a prediction for one digit  
- **Loss Function**: Combined loss using CrossEntropy for each output  
- **Optimizer**: Adam with learning rate scheduling  
### 🔹 Model Architecture Diagram  
<p align="center">
  <img src="https://github.com/Poulami-Nandi/twoDigRecognizeCNN/blob/main/assets/model_architecture.png" width="600"/>
</p>   

---

## 🔄 Training & Evaluation Strategy  

- Trained for multiple epochs with stratified shuffling  
- Tracked accuracy per digit position  
- Generated a CSV submission format expected for Kaggle-style competition  
- Used custom PyTorch Dataset and DataLoader classes for loading two-label samples

---

## Inside the Notebook
| Section | Highlights | Figure |
|---------|------------|--------|
| **EDA & Visualisation** | pixel-intensity histograms, class balance | ![EDA](assets/image_4.png) |
| **Data Pipeline** | NumPy stacks → `tf.data.Dataset` → on-the-fly augmentation | ![Pipeline](assets/image_5.png) |
| **Training** | `ReduceLROnPlateau`, `ModelCheckpoint`, `TensorBoard` callbacks | ![Training](assets/image_6.png) |
| **Error Analysis** | visualise top-k mis-classifications | ![Errors](assets/image_7.png) |

*(12 additional figures are stored in `assets/` for reference.)*

---

## Research Extensions
* **Quantum-mechanics imaging** – port the same CNN to classify snapshots of **Hubbard-model** simulations (work in progress in my lab notebook).
* **Sequence models** – replace concatenation with an attention-based recogniser for variable-length digit strings.
* **Hardware deployment** – convert the Keras model to **TensorFlow Lite** for on-device digit reading (embedded meters / industrial scanners).

---

## Citation
If you use this codebase, please cite:

```bibtex
@misc{nandi2025twodigcnn,
  author  = {Poulami Nandi},
  title   = {Two consecutive Digit Recognition Using Convolutional Neural Network},
  year    = {2025},
  url     = {https://github.com/Poulami-Nandi/twoDigRecognizeCNN}
}
```

## 📈 Results & Visuals  

### 🔹 Training Loss Curves  
<p align="center">
  <img src="https://github.com/Poulami-Nandi/twoDigRecognizeCNN/blob/main/assets/training_curves.png" width="600"/>
</p>

### 🔹 Prediction Examples  
<p align="center">
  <img src="https://github.com/Poulami-Nandi/twoDigRecognizeCNN/blob/main/assets/sample_output_1.png" width="600"/>
</p>

---

## 🧾 Submission File Format  

The output CSV format used for submissions is as follows:

Id,First Digit,Second Digit
0,8,2
1,5,7
2,1,3
...

---
### 📊 Final Model Performance

| Metric                   | Value     |
|--------------------------|-----------|
| Final Training Accuracy  | **0.9984** |
| Final Validation Accuracy| **0.9804** |
| Final Training Loss      | **0.0051** |
| Final Validation Loss    | **0.0917** |

---

## 🚀 Future Work  

- Integrate ResNet/CNN variants to improve precision  
- Use data augmentation and regularization tuning  
- Deploy using Streamlit for live handwritten input inference  
- Grad-CAM for model interpretability  

---

## 👩‍💻 Author Info  

**Author**: [Dr. Poulami Nandi](https://www.linkedin.com/in/poulami-nandi/)  
<img src="https://github.com/Poulami-Nandi/IV_surface_analyzer/raw/main/images/own/own_image.jpg" alt="Profile" width="150"/>  
Physicist · Quant Researcher · Data Scientist  
[University of Pennsylvania](https://live-sas-physics.pantheon.sas.upenn.edu/people/poulami-nandi) | [IIT Kanpur](https://www.iitk.ac.in/) | [TU Wien](http://www.itp.tuwien.ac.at/CPT/index.htm?date=201838&cats=xbrbknmztwd)

📧 [nandi.poulami91@gmail.com](mailto:nandi.poulami91@gmail.com), [pnandi@sas.upenn.edu](mailto:pnandi@sas.upenn.edu)  
🔗 [LinkedIn](https://www.linkedin.com/in/poulami-nandi-a8a12917b/) • [GitHub](https://github.com/Poulami-Nandi) • [Google Scholar](https://scholar.google.co.in/citations?user=bOYJeAYAAAAJ&hl=en)

---

## 📝 License  

This project is licensed under the MIT License. See [LICENSE](https://github.com/Poulami-Nandi/twoDigRecognizeCNN/blob/main/LICENSE) for details.

---

## 🙏 Acknowledgments  

- Kaggle platform for dataset  
- PyTorch community for tutorials  
- Open source community contributors

---
