# 🗣️ An AI-Driven Approach for Personalized Articulation Therapy

This repository contains the complete pipeline and models for the paper:  
**"An AI-Driven Approach for Evaluating and Recommending Personalized Therapy for Articulation Speech Disorders Using Audio-to-Text"**  
Submitted to **ACL 2025 - Linguistic Annotation Workshop (LAW XIX)**.

---

## 📌 Overview

This project focuses on the automatic evaluation of articulation disorders and personalized therapy recommendations. The system combines:

- **ASR (Automatic Speech Recognition)**
- **Phoneme-level IPA conversion**
- **Error detection and alignment**
- **LLM-based interpretation**
- **Psychologically informed therapy exercises**

---

## 🧠 Architecture

The pipeline consists of three major modules:

### 1. Phoneme Alphabetic Conversion
Grapheme-to-Phoneme (G2P) transformation is performed using multiple LLMs (LLaMA 8B, Mistral 7B, LLaMA 7B), followed by CoT (Chain-of-Thought) reasoning via a Gamma 9B model to select the most appropriate IPA transcription.

### 2. Interpretation Module
Using the same LLM ensemble, the IPA output is analyzed for phonetic inconsistencies and articulation errors. This step aids in identifying speech patterns and phoneme-level deviations in disordered speech.

### 3. Final Decision & Therapy Recommendation
Combining phoneme errors with psychological inputs, the system generates **personalized therapy exercises** tailored to individual needs. The psychological question bank and logic for therapy generation are available in this repository[^1].

---

> 🔗 **View Psychological Assessment Form**  
> ![Psychological Form](/assets/images/psychological_form.png)  
> *(Form details and question mappings are available in the repository’s `/data` folder.)*


## 🧪 Dataset Links

- **TORGO Database**: [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)  
- **UA-Speech Corpus**: [UA-Speech](https://isle.illinois.edu/sst/data/ua-speech/)  
- **LibriSpeech (Baseline)**: [LibriSpeech](https://www.openslr.org/12)

# 🚀 Project Setup Guide

---

## 📁 Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/ai-articulation-therapy.git
cd ai-articulation-therapy
```

---

## 🛠 Step 2: Run the Setup Script

Make the setup script executable and run it by providing a name for your Conda environment:

```bash
chmod +x setup.sh
./setup.sh lip
```

- Replace `lip` with your desired environment name.
- The script will:
  - Install required system packages
  - Ensure CUDA is properly configured
  - Create a Conda environment with Python 3.10
  - Install `pip` and `llama-cpp-python` with CUDA support

---

## 🐍 Step 3: Activate the Conda Environment

After the setup script finishes successfully, activate the environment:

```bash
conda activate lip
```

---

## 📦 Step 4: Install Python Dependencies

Install additional dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## ✅ You’re All Set!

You can now run the application or start development. If you face any issues:

- Ensure CUDA 12.4 is installed and available at `/usr/local/cuda-12.4`
- Make sure Anaconda or Miniconda is installed and accessible via the `conda` command


---

## 🤖 Running the Project

To run the app using your preferred GGUF-format LLMs, follow these steps:

### 1. 🧠 Add Your LLM Models

Download your GGUF-format LLM models (e.g., from Hugging Face or other sources) and place them inside the `model/` directory.

> Example:
> ```
> model/llama-chat-3.1-q8.gguf
> ```

### 2. ✏️ Update `app.py`

Edit the `model_paths` list in `app.py` with your actual model filenames. You can use different models or multiple copies of the same model, depending on your setup.

```python
model_paths = [
    "model/your-model-name.gguf",
    "model/your-model-name.gguf",
    "model/your-model-name.gguf",
    "model/your-model-name.gguf",
]
```
### 3. 🚀 Run the App

Use **Streamlit** to launch the app:

```bash
streamlit run app.py
```