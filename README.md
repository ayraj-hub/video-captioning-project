# 🧠 Video Captioning Project
This project is designed to generate captions for videos using a deep learning model. The core functionality of the project involves loading a pre-trained BLIP model, setting up a Gradio application, and defining the interface for user interaction. The project utilizes the MSR-VTT dataset for training and testing the model. The key features of the project include video captioning, frame sampling, and model training.

## 🚀 Features
- **Video Captioning**: The project uses a pre-trained BLIP model to generate captions for videos.
- **Frame Sampling**: The project includes a frame sampling functionality to extract frames from videos at uniform intervals.
- **Model Training**: The project involves training a video captioning model using the MSR-VTT dataset.
- **Gradio Application**: The project sets up a Gradio application to provide a user interface for interacting with the video captioning model.

## 🛠️ Tech Stack
- **Frontend**: Gradio
- **Backend**: PyTorch
- **Database**: None
- **AI Tools**: Transformers, BLIP model
- **Build Tools**: None
- **Dependencies**: 
  - `gradio` for creating the web application
  - `torch` for deep learning operations
  - `transformers` for the BLIP model and its processor
  - `cv2` and `PIL` for image and video processing
  - `datasets` for loading the MSR-VTT dataset
  - `pandas` for data manipulation and analysis
  - `numpy` for numerical operations

## 📦 Installation
To install the required dependencies, run the following command:
```bash
pip install gradio torch transformers cv2 pillow datasets pandas numpy
```

## 💻 Usage
To run the project, follow these steps:
1. Clone the repository: `git clone https://github.com/your-repo/video-captioning.git`
2. Navigate to the project directory: `cd video-captioning`
3. Install the required dependencies: `pip install -r requirements.txt`
4. Run the Gradio application: `python app.ipynb`

## 📂 Project Structure
```markdown
video-captioning/
├── app.ipynb
├── dataset.ipynb
├── training.ipynb
├── frame_sampling.ipynb
├── videomaegpt2train.ipynb
├── requirements.txt
└── README.md
```
## 📊 Evaluation Results

The model was evaluated against a baseline using standard language modeling metrics (BLEU, ROUGE-L, and CIDEr). The final model shows a substantial leap in performance across all categories.

| Metric | Baseline Score | Final Model | Improvement |
| :--- | :---: | :---: | :---: |
| **Bleu_1** | 70.21 | **89.50** | +19.29 |
| **Bleu_2** | 50.34 | **81.08** | +30.74 |
| **Bleu_3** | 33.74 | **72.10** | +38.36 |
| **Bleu_4** | 22.36 | **62.99** | +40.63 |
| **ROUGE_L** | 47.86 | **73.08** | +25.22 |
| **CIDEr** | 29.92 | **87.39** | **+57.47** |


### 📈 Performance Analysis
* **Contextual Accuracy**: The **CIDEr** score saw the most significant jump (+57.47), indicating the final model is much better at capturing the specific consensus of the video content rather than just matching common words.
* **Structural Fluency**: The massive increase in **Bleu_4** (+40.63) shows that the model has progressed from generating simple fragments to high-quality, continuous 4-gram sequences.
* **Overall Recall**: A **ROUGE_L** score of 73.08 suggests the generated captions maintain high structural similarity to the ground truth references.

## 📸 Screenshots

## 🤝 Contributing
To contribute to the project, please follow these steps:
1. Fork the repository: `git fork https://github.com/your-repo/video-captioning.git`
2. Create a new branch: `git branch your-branch`
3. Make changes and commit: `git commit -m "your-commit-message"`
4. Push changes: `git push origin your-branch`
5. Create a pull request: `git pull-request`

