# process_text.py
from llama_model import LlamaModelManager
import re, json
import streamlit as st
from typing import List, Dict, Optional
from audiototext import audio_to_text_whisper

def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    return re.sub(r'\s+', ' ', text).strip() if text else ""

def extract_ipa(text: str) -> Optional[str]:
    """Extract IPA between slashes from model response."""
    matches = re.findall(r'/([^/]+)/', text)
    return f"/{matches[0]}/" if matches else None

def load_model(model_path: str, is_mixtral: bool = False) -> Optional[LlamaModelManager]:
    """Load the LLaMA or Mixtral model from GGUF file."""
    print(f"Attempting to load model from {model_path}")
    try:
        model = LlamaModelManager(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except ValueError as e:
        print(f"Model loading failed: {str(e)}")
        LlamaModelManager.cleanup()
        print("Retrying after cleanup...")
        try:
            model = LlamaModelManager(model_path)
            print(f"Model loaded successfully on retry from {model_path}")
            return model
        except ValueError as e:
            print(f"Permanent model loading failure: {str(e)}")
            return None
    except Exception as e:
        print(f"Unexpected error loading model: {str(e)}")
        return None

def generate_ipa(model: LlamaModelManager, text: str) -> Optional[str]:
    if not text:
        return None
    prompt = f"""You are an expert phonetician. Convert this text to International Phonetic Alphabets (IPA):
Text: "{text}"

Rules:
1. Use /slashes/ 
2. Return ONLY the IPA between / /

Example: 
"butter" → /ˈbʌtər/ or /ˈbʌɾɚ/
"the quick fox" → /ðə kwɪk fɑks/

IPA:
"""
    for _ in range(2):
        response = model.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            stop=['\n', 'Text:']
        ).strip()
        print(f"IPA response: {response}")
        ipa_start = response.find("/ ")
        ipa_end = response.rfind(" /")
        if ipa_start != -1 and ipa_end != -1 and ipa_start < ipa_end:
            return response[ipa_start + 1:ipa_end + 1]
        elif "/" in response:
            ipa = response.split("/")[-2] if response.count("/") >= 2 else response.strip("/")
            if ipa:
                return f"/{ipa}/"
    return None

def evaluate_transcriptions(model: LlamaModelManager, 
                           original_text: str,
                           transcribed_text: str,
                           original_ipas: List[str],
                           transcribed_ipas: List[str]) -> Dict:
    prompt = f"""You are a phonetics expert helping evaluate IPA transcriptions.

Original Text: "{original_text}"
Original IPA Options:
1. {original_ipas[0]}
2. {original_ipas[1]}
3. {original_ipas[2]}

Transcribed Text: "{transcribed_text}"
Transcribed IPA Options:
1. {transcribed_ipas[0]}
2. {transcribed_ipas[1]}
3. {transcribed_ipas[2]}

Evaluation Instructions (Think Step by Step):
Step 1: Break down the original text phonetically. Think about the pronunciation of each syllable and determine what the ideal IPA transcription should look like.
Step 2: Compare each of the 3 original IPA options to the ideal IPA. For each, explain what phonemes match or deviate, and choose the best match.
Step 3: Repeat the same phonetic breakdown for the transcribed text. Analyze it sound-by-sound and mentally reconstruct the expected IPA.
Step 4: Compare the 3 transcribed IPA options to your expected transcription. Discuss how each one aligns or deviates. Choose the best match.
Step 5: For both selections, consider phonetic accuracy, stress, articulation, and typical variation patterns in speech.
Step 6: Assign a confidence score (1 to 10) based on clarity, precision, and how strongly the choice matches the expected IPA.
Step 7: Based on stylistic or phonetic cues, determine which model most likely generated the selected IPA transcriptions.

Output Instructions:
- Your reasoning should follow each step above.
- After all steps, output strictly JSON in the format below.
- DO NOT add explanations after the JSON.
- DO NOT include any non-JSON text before or after.
- Only the JSON should be in the final output.

Required JSON Format:
<<CAI-DSVV-IAI>>{{
  "best_ipa_original": "...",
  "best_ipa_transcript": "...",
  "confidence": ...,
  "selected_model": ...
}}<<CAI-DSVV-IAI>>

Now Think Step by Step and Give Only the Final JSON Output:
"""
    response = model.generate(
        prompt=prompt,
        max_tokens=400,
        temperature=0.0,
        stop=["\nNote:", "\nExplanation:", "\nNow give", "\nOutput:", "Note:", "Explanation:"]
    ).strip() + '}'
    try:
        print("----------------------"*5)
        print('Evaluation response:', response)
        print("----------------------"*5)
        return json.loads(response.split("<<CAI-DSVV-IAI>>")[1])
    except:
        return {
            "best_ipa_original": original_ipas[0],
            "best_ipa_transcript": transcribed_ipas[0],
            "confidence": 5
        }

def analyze_articulation_errors(model: LlamaModelManager,
                               original_text: str,
                               original_ipa: str,
                               transcribed_text: str,
                               transcribed_ipa: str) -> Dict:
    errors_prompt = f"""
Analyze the transcription for articulation errors using the SODA framework.

Original Text: "{original_text}"
Original IPA: {original_ipa}

Transcribed Text: "{transcribed_text}"
Transcribed IPA: {transcribed_ipa}

Instructions:
1. Compare the Original IPA and Transcribed IPA carefully.
2. Identify all articulation errors using:
   - Substitution
   - Omission
   - Distortion
   - Addition
3. For each error, provide:
   - type
   - original_sound
   - transcribed_sound
   - position (phoneme index or word index)
4. Output ONLY the errors list in JSON format.
5. DO NOT include affected speech organs yet.
6. Wrap JSON between <<ERRORS>> and <<ERRORS>> tags.

Output ONLY this format:
<<ERRORS>>
{{
  "errors": [
    {{
      "type": "Substitution" | "Omission" | "Distortion" | "Addition",
      "original_sound": "IPA symbol(s)",
      "transcribed_sound": "IPA symbol(s)",
      "position": "phoneme index or word index"
    }}
  ]
}}
<<ERRORS>>
"""
    errors_response = model.generate(
        prompt=errors_prompt,
        max_tokens=1000,
        temperature=0.0,
        stop=['<<ERRORS>>']
    )
    try:
        print("--------------"*5)
        print("Errors response:", errors_response)
        print("--------------"*5)
        errors_data = json.loads(errors_response)
    except:
        errors_data = {"errors": []}

    if errors_data["errors"]:
        organs_prompt = f"""
You are an expert in phonetics and speech articulation.

Analyze the following articulation errors and identify which human speech organs are likely responsible for the errors.

Errors:
{json.dumps(errors_data["errors"], indent=2, ensure_ascii=False)}

Possible Organs: lips, teeth, tongue, palate, velum, glottis

Instructions:
1. Use the types of errors and the phonemes involved to determine which organs are affected.
2. Choose only from this list: lips, teeth, tongue, palate, velum, glottis.
3. Do NOT explain or output placeholder text.
4. Output only valid JSON, exactly in the format shown below.
5. Wrap the JSON between <<ORGANS>> and <<ORGANS>>. Do not add any extra text.

Format Example:
<<ORGANS>>
{{
  "affected_speech_organs": ["tongue", "palate"]
}}
<<ORGANS>>
"""
        organs_response = model.generate(
            prompt=organs_prompt,
            max_tokens=300,
            temperature=0.0,
            stop=['<<ORGANS>>']
        )
        try:
            print("--------------"*5)
            print("Organs response:", organs_response)
            print("--------------"*5)
            organs_data = json.loads(organs_response)
        except:
            organs_data = {"affected_speech_organs": []}
    else:
        organs_data = {"affected_speech_organs": []}

    return {
        "errors": errors_data.get("errors", []),
        "affected_speech_organs": organs_data.get("affected_speech_organs", [])
    }

def evaluate_soda_analyses(model: LlamaModelManager, analyses: List[Dict]) -> Dict:
    prompt = f"""
Evaluate these SODA analyses and select the most accurate one:

Analysis Options:
1. {analyses[0]}
2. {analyses[1]}
3. {analyses[2]}

Rules:
1. Consider completeness of error identification
2. Consider accuracy of speech organ attribution
3. Return ONLY JSON with:
{{
  "selected_analysis": (index 0-2),
  "confidence": (1-10),
  "consolidated_analysis": (merged best findings)
}}

JSON Output:
"""
    response = model.generate(
        prompt=prompt,
        max_tokens=500,
        temperature=0.1,
        stop=['}\n']
    ).strip() + '}'
    try:
        print("--------------"*5)
        print("SODA evaluation response:", response)
        print("--------------"*5)
        return json.loads(response, ensure_ascii=False)
    except:
        return {
            "selected_analysis": 0,
            "confidence": 5,
            "consolidated_analysis": analyses[0]
        }

def generate_soda_summary(model: LlamaModelManager, 
                          original_text: str,
                          transcribed_text: str,
                          best_ipa_original: str,
                          best_ipa_transcript: str,
                          soda_analysis: Dict,
                          psychological_profile: Dict = None) -> Dict:
    print(f"Psychological profile: {psychological_profile}")
    prompt = f"""
You are a clinical speech-language pathologist and AI assistant.

You will analyze articulation error data (SODA analysis) along with a psychological profile to generate a highly focused JSON summary. The goal is to improve articulation outcomes by tailoring suggestions based on actual speech errors and user-reported emotional or cognitive states.

Input Data:
- Original Text: "{original_text}"
- Transcribed Text: "{transcribed_text}"
- Original IPA: "{best_ipa_original}"
- Transcribed IPA: "{best_ipa_transcript}"
- SODA Analysis: {json.dumps(soda_analysis, indent=2, ensure_ascii=False)}
- Psychological Profile: {json.dumps(psychological_profile, indent=2, ensure_ascii=False) if psychological_profile else {}}

Instructions:
1. Use **SODA analysis** to count articulation errors and identify their types.
2. Determine which organs (e.g., tongue, lips, palate) are most affected.
3. Assess articulation accuracy as "High", "Moderate", or "Low".
4. Analyze the **psychological profile** only for factors that affect speech motivation or consistency (e.g., anxiety, frustration, avoidance). **Do NOT include raw survey answers or preferences.**
5. Create **custom articulation exercises** based on the specific SODA issues (e.g., substitution of /s/ → /ʃ/) and psychological profile. Exercises must be detailed and **NOT copied** from user preferences. Be creative and therapeutic.
6. Output must be **pure JSON**, structured like this:

Output Format (Strictly JSON):
<<CAI-DSVV-IAI>>{{
  "total_errors": int,
  "error_breakdown": {{
    "substitution": int,
    "omission": int,
    "distortion": int,
    "addition": int
  }},
  "most_affected_organs": ["tongue", "palate"],
  "psychological_insights": "Brief summary of how emotional state affects articulation effort or consistency.",
  "articulation_accuracy": "High|Moderate|Low",
  "personalized_exercises": [
    "Exercise 1: ...",
    "Exercise 2: ..."
  ]
}}<<CAI-DSVV-IAI>>

IMPORTANT:
- Do not echo user preferences like 'interactive' or '10 minutes'.
- Do not include irrelevant psychological commentary.
- Only output valid JSON — no extra text.
- Wrap the entire JSON output between the tags:
    - <<CAI-DSVV-IAI>> and <<CAI-DSVV-IAI>>.
"""
    response = model.generate(
        prompt=prompt,
        max_tokens=300,
        temperature=0.0,
        stop=['}\n']
    ).strip() + '}'
    try:
        print("-><-"*5)
        print("SODA summary response:", response)
        print("-><-"*5)
        return json.loads(response.split('<<CAI-DSVV-IAI>>')[1])
    except:
        return {
            "total_errors": len(soda_analysis["errors"]),
            "error_breakdown": {
                "substitution": sum(1 for e in soda_analysis["errors"] if e["type"] == "Substitution"),
                "omission": sum(1 for e in soda_analysis["errors"] if e["type"] == "Omission"),
                "distortion": sum(1 for e in soda_analysis["errors"] if e["type"] == "Distortion"),
                "addition": sum(1 for e in soda_analysis["errors"] if e["type"] == "Addition")
            },
            "most_affected_organs": soda_analysis["affected_speech_organs"] or ["unknown"],
            "psychological_insights": "No significant psychological impact noted" if not psychological_profile else f"Based on profile: {psychological_profile.get('speech_impact', 'Moderate')}",
            "articulation_accuracy": "Moderate",
            "personalized_exercises": ["Practice minimal pair words.", "Repeat challenging phonemes in isolation."]
        }

def process_inputs(audio_path: str, 
                   original_text: str,
                   model_paths: List[str]) -> Optional[Dict]:
    """Full processing pipeline with evaluation."""
    model_instances = []
    try:
        raw_text = audio_to_text_whisper(audio_path)
        transcribed_text = clean_text(raw_text)
        if not transcribed_text:
            st.error("Audio transcription failed")
            return None
        
        original_ipas = []
        transcribed_ipas = []
        
        with st.spinner("Generating IPA variants..."):
            for i in range(3):
                model = load_model(model_paths[i])
                if model:
                    model_instances.append(model)
                    ipa = generate_ipa(model, original_text)
                    if ipa:
                        original_ipas.append(ipa)
                        st.write(f"Original IPA {i+1}: {ipa}")
            
            for i in range(3):
                model = load_model(model_paths[i])
                if model and model not in model_instances:
                    model_instances.append(model)
                ipa = generate_ipa(model, transcribed_text)
                if ipa:
                    transcribed_ipas.append(ipa)
                    st.write(f"Transcribed IPA {i+1}: {ipa}")
        
        if len(original_ipas) < 3 or len(transcribed_ipas) < 3:
            st.error("Insufficient IPA variants generated")
            return None
        
        with st.spinner("Evaluating transcriptions..."):
            evaluation_model = load_model(model_paths[3])
            if evaluation_model:
                model_instances.append(evaluation_model)
            evaluation = evaluate_transcriptions(
                evaluation_model,
                original_text,
                transcribed_text,
                original_ipas,
                transcribed_ipas
            )
        
        results = {
            "original_text": original_text,
            "transcribed_text": transcribed_text,
            "original_ipas": original_ipas,
            "transcribed_ipas": transcribed_ipas,
            "evaluation": evaluation
        }
        
        with st.spinner("Performing SODA analysis..."):
            best_original_ipa = evaluation["best_ipa_original"]
            best_transcribed_ipa = evaluation["best_ipa_transcript"]
            
            soda_analyses = []
            for i in range(3):
                model = load_model(model_paths[i])
                if model and model not in model_instances:
                    model_instances.append(model)
                analysis = analyze_articulation_errors(
                    model,
                    original_text,
                    best_original_ipa,
                    transcribed_text,
                    best_transcribed_ipa
                )
                soda_analyses.append(analysis)
                st.write(f"SODA Analysis {i+1}: {json.dumps(analysis, indent=2, ensure_ascii=False)}")
            
            evaluation_model = load_model(model_paths[3])
            if evaluation_model and evaluation_model not in model_instances:
                model_instances.append(evaluation_model)
            soda_evaluation = evaluate_soda_analyses(
                evaluation_model,
                soda_analyses
            )
            
            results["soda_analyses"] = soda_analyses
            results["soda_evaluation"] = soda_evaluation
        
        return results
    
    finally:
        for _ in range(len(model_instances)):
            LlamaModelManager.cleanup()
        print("All models cleaned up at end of process_inputs.")