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
> ![Psychological Form](/assets/images/psychological_form.jpg)  
> *(Form details and question mappings are available in the repository’s `/data` folder.)*


## 🧪 Dataset Links

- **TORGO Database**: [TORGO](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html)  
- **UA-Speech Corpus**: [UA-Speech](https://isle.illinois.edu/sst/data/ua-speech/)  
- **LibriSpeech (Baseline)**: [LibriSpeech](https://www.openslr.org/12)

---

## 🛠️ How to Run

```bash
git clone https://github.com/your-username/ai-articulation-therapy.git
cd ai-articulation-therapy

# Install dependencies
pip install -r requirements.txt

# Run pipeline
python run_pipeline.py --input audio.wav
