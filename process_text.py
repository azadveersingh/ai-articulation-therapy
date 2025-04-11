from llama_model import LlamaModelManager
import re, json
from typing import List, Dict, Optional
# ===== CORE FUNCTIONS ===== 
def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    return re.sub(r'\s+', ' ', text).strip() if text else ""

def extract_ipa(text: str) -> Optional[str]:
    """Extract IPA between slashes from model response."""
    matches = re.findall(r'/([^/]+)/', text)
    return f"/{matches[0]}/" if matches else None

# ===== MODEL FUNCTIONS =====
def load_model(model_path: str, is_mixtral: bool = False) -> Optional[LlamaModelManager]:
    """Load and cache the LLaMA or Mixtral model from GGUF file."""
    print(f"Attempting to load model from {model_path}")
    try:
        # Parameters for loading GGUF files
        params = {
            "model_path": model_path,
            "n_ctx": 2048,
            "n_gpu_layers": 0 if is_mixtral else -1,  # Offload Mixtral to CPU, LLaMA to GPU
            "verbose": False
        }
        # Load the GGUF file directly
        # Wrap in LlamaModelManager (assuming it can take a preloaded model)
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
    """Generate IPA transcription with retries."""
    """Generate IPA transcription."""
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
    
    for _ in range(2):  # 2 attempts
        response = model.generate(
            prompt=prompt,
            max_tokens=100,
            temperature=0.1,
            stop=['\n', 'Text:']
        ).strip()
        print(response)
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
    """Evaluate which IPA transcriptions are best."""
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
    """Analyze phoneme errors using SODA framework (split into two parts)."""
    # Part 1: Get the errors list
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
    // Add more errors if needed
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
        errors_data = json.loads(errors_response)
    except:
        errors_data = {"errors": []}
    
    # Part 2: Get affected speech organs
    if errors_data["errors"]:
        organs_prompt = f"""
You are an expert in phonetics and speech articulation.

Analyze the following articulation errors and identify which human speech organs are likely responsible for the errors.

Errors:
{json.dumps(errors_data["errors"], indent=2,ensure_ascii=False)}

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
            organs_data = json.loads(organs_response)
        except:
            organs_data = {"affected_speech_organs": []}
    else:
        organs_data = {"affected_speech_organs": []}
    
    # Merge results
    return {
        "errors": errors_data.get("errors", []),
        "affected_speech_organs": organs_data.get("affected_speech_organs", [])
    }

def evaluate_soda_analyses(model: LlamaModelManager,
                         analyses: List[Dict]) -> Dict:
    """Evaluate multiple SODA analyses and select the best one."""
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

JSON Output:"""
    
    response = model.generate(
        prompt=prompt,
        max_tokens=500,
        temperature=0.1,
        stop=['}\n']
    ).strip() + '}'
    
    try:
        return json.loads(response,ensure_ascii=False )
    except:
        return {
            "selected_analysis": 0,
            "confidence": 5,
            "consolidated_analysis": analyses[0]
        }

# def generate_soda_summary(model: LlamaModelManager, 
#                          original_text: str,
#                          transcribed_text: str,
#                          best_ipa_original: str,
#                          best_ipa_transcript: str,
#                          soda_analysis: Dict) -> Dict:
#     """Generate a final summary of the SODA analysis."""
#     prompt = f"""
# You are an expert in phonetics and speech-language pathology.
# Your task is to generate a concise final summary based on the SODA analysis of articulation errors.

# Original Text: "{original_text}"
# Transcribed Text: "{transcribed_text}"
# Original IPA: "{best_ipa_original}"
# Transcribed IPA: "{best_ipa_transcript}"
# SODA Analysis: {json.dumps(soda_analysis, indent=2, ensure_ascii=False)}

# Instructions:
# - Step 1: Review the SODA analysis, including identified errors and affected speech organs.
# - Step 2: Summarize the total number of errors and their types (Substitution, Omission, Distortion, Addition).
# - Step 3: Highlight the most frequently affected speech organs.
# - Step 4: Provide a brief assessment of articulation accuracy (e.g., "High", "Moderate", "Low") based on error count and severity.
# - Step 5: Assign a confidence score (1-10) for the summary based on the clarity and consistency of the analysis.

# Output Format:
# - Output strictly JSON in the format below.
# - NO explanations. NO extra text.
# - OUTPUT JSON ONLY.

# Required JSON Format:
# {{
#   "total_errors": ...,
#   "error_breakdown": {{
#     "substitution": ...,
#     "omission": ...,
#     "distortion": ...,
#     "addition": ...
#   }},
#   "most_affected_organs": ["...", "..."],
#   "articulation_accuracy": "High|Moderate|Low",
#   "confidence": ...
# }}
# """
    
#     response = model.generate(
#         prompt=prompt,
#         max_tokens=300,
#         temperature=0.0,
#         stop=['}\n']
#     ).strip() + '}'
    
#     try:
#         return json.loads(response)
#     except:
#         return {
#             "total_errors": len(soda_analysis["errors"]),
#             "error_breakdown": {
#                 "substitution": sum(1 for e in soda_analysis["errors"] if e["type"] == "Substitution"),
#                 "omission": sum(1 for e in soda_analysis["errors"] if e["type"] == "Omission"),
#                 "distortion": sum(1 for e in soda_analysis["errors"] if e["type"] == "Distortion"),
#                 "addition": sum(1 for e in soda_analysis["errors"] if e["type"] == "Addition")
#             },
#             "most_affected_organs": soda_analysis["affected_speech_organs"] or ["unknown"],
#             "articulation_accuracy": "Moderate",
#             "confidence": 5
#         }
                    
           
def generate_soda_summary(model: LlamaModelManager, 
                         original_text: str,
                         transcribed_text: str,
                         best_ipa_original: str,
                         best_ipa_transcript: str,
                         soda_analysis: Dict,
                         psychological_profile: Dict = None) -> Dict:
    """Generate a final summary of the SODA analysis with psychological input."""
    prompt = f"""
You are an expert in phonetics and speech-language pathology.
Your task is to generate a concise final summary based on the SODA analysis of articulation errors and psychological profile.

Original Text: "{original_text}"
Transcribed Text: "{transcribed_text}"
Original IPA: "{best_ipa_original}"
Transcribed IPA: "{best_ipa_transcript}"
SODA Analysis: {json.dumps(soda_analysis, indent=2, ensure_ascii=False)}
Psychological Profile: {json.dumps(psychological_profile, indent=2, ensure_ascii=False) if psychological_profile else {{}}}

Instructions:
- Step 1: Review the SODA analysis, including identified errors and affected speech organs.
- Step 2: Summarize the total number of errors and their types (Substitution, Omission, Distortion, Addition).
- Step 3: Highlight the most frequently affected speech organs.
- Step 4: Incorporate psychological profile data (e.g., self-assessment, emotional impact) to assess motivation and therapy needs.
- Step 5: Provide a brief assessment of articulation accuracy (e.g., "High", "Moderate", "Low") based on error count, severity, and psychological factors.
- Step 6: Assign a confidence score (1-10) for the summary based on the clarity, consistency, and relevance of the analysis.

Output Format:
- Output strictly JSON in the format below.
- NO explanations. NO extra text.
- OUTPUT JSON ONLY.

Required JSON Format:
{{
  "total_errors": ...,
  "error_breakdown": {{
    "substitution": ...,
    "omission": ...,
    "distortion": ...,
    "addition": ...
  }},
  "most_affected_organs": ["...", "..."],
  "articulation_accuracy": "High|Moderate|Low",
  "psychological_insights": "...",
  "confidence": ...
}}
"""
    
    response = model.generate(
        prompt=prompt,
        max_tokens=300,
        temperature=0.0,
        stop=['}\n']
    ).strip() + '}'
    
    try:
        return json.loads(response)
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
            "articulation_accuracy": "Moderate",
            "psychological_insights": "No significant psychological impact noted" if not psychological_profile else f"Based on profile: {psychological_profile.get('speech_impact', 'Moderate')}",
            "confidence": 5
        }         
                    
                    
                    
                    
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                












# Prompt:1



# prompt = f"""You are a phonetics expert helping evaluate IPA transcriptions.

# Original Text: "{original_text}"
# Original IPA Options:
# 1. {original_ipas[0]}
# 2. {original_ipas[1]}
# 3. {original_ipas[2]}

# Transcribed Text: "{transcribed_text}"
# Transcribed IPA Options:
# 1. {transcribed_ipas[0]}
# 2. {transcribed_ipas[1]}
# 3. {transcribed_ipas[2]}

# Evaluation Instructions:
# - Step 1: Think step-by-step to generate the correct IPA transcription for the original text.
# - Step 2: Compare each original IPA option to the correct IPA, and select the most accurate one.
# - Step 3: Think step-by-step to determine the expected IPA for the transcribed text.
# - Step 4: Compare each transcribed IPA option to the expected one, and select the most accurate.
# - Step 5: Based on phonetic similarity, stress patterns, and articulation, select one final transcription for each.
# - Step 6: Assign a confidence score (1 to 10) based on consistency, clarity, and match quality.
# - Step 7: Identify which model generated the selected transcriptions.

# Output Format:
# - Output strictly JSON in the format below.
# - NO explanations. NO extra text.
# - DO NOT say "Answer", "Response", or "Output".
# - OUTPUT JSON ONLY.

# Required JSON Format:
# <<CAI-DSVV-IAI>>{{
#   "best_ipa_original": "...",
#   "best_ipa_transcript": "...",
#   "confidence": ...,
#   "selected_model": ...
# }}<<CAI-DSVV-IAI>>
# """






# prompt: 2

# prompt = f"""You are an expert in speech-language pathology and phonetics.
# Your task is to interpret phoneme-level articulation errors between the original and transcribed IPA sequences.
# Use the SODA classification system (Substitution, Omission, Distortion, Addition).
# Also infer articulatory features like affected organ (e.g., tongue, lips, glottis) and part-of-speech context.

# Original IPA: "{best_ipa_original}"
# Transcribed IPA: "{best_ipa_transcript}"

# Instructions:
# - Step 1: Compare each phoneme in both IPA sequences.
# - Step 2: Identify phoneme mismatches and categorize each as Substitution, Omission, Distortion, or Addition.
# - Step 3: For each error, determine the likely articulatory organ involved.
# - Step 4: Indicate the part-of-speech of the word containing the error (if identifiable).
# - Step 5: Summarize findings with error type counts and linguistic patterns.
# - Step 6: Assign a confidence score (1 to 10) for the overall analysis.

# Output Format:
# - Output strictly JSON in the format below.
# - NO explanations. NO extra text.
# - DO NOT say "Answer", "Response", or "Output".
# - OUTPUT JSON ONLY.

# Required JSON Format:
# <<CAI-DSVV-IAI>>{{
#   "errors": [
#     {{
#       "original_phoneme": "...",
#       "transcribed_phoneme": "...",
#       "error_type": "Substitution|Omission|Distortion|Addition",
#       "articulatory_organ": "...",
#       "part_of_speech": "noun|verb|adj|adv|other"
#     }}
#     ...
#   ],
#   "summary": {{
#     "total_errors": ...,
#     "substitution": ...,
#     "omission": ...,
#     "distortion": ...,
#     "addition": ...
#   }},
#   "confidence": ...
# }}<<CAI-DSVV-IAI>>
# """




# prompt 3:-

# prompt = f"""You are an expert in speech-language therapy and cognitive psychology.
# Your task is to generate personalized articulation therapy recommendations based on phoneme-level errors and psychological profile.
# Use insights from prior IPA interpretation and responses to psychological assessment questions.

# IPA Error Summary:
# {ipa_error_summary}

# Psychological Profile:
# {psychological_answers}

# Instructions:
# - Step 1: Analyze the IPA error types (Substitution, Omission, Distortion, Addition) and frequency.
# - Step 2: Consider the part-of-speech and affected articulatory organs for each error.
# - Step 3: Integrate psychological assessment data to determine user's emotional and cognitive needs.
# - Step 4: Generate 3 personalized therapy exercises targeting both articulation errors and psychological support.
# - Step 5: Justify each recommendation in one sentence.
# - Step 6: Assign an overall therapy priority level (Low | Medium | High).

# Output Format:
# - Output strictly JSON in the format below.
# - NO explanations. NO extra text.
# - DO NOT say "Answer", "Response", or "Output".
# - OUTPUT JSON ONLY.

# Required JSON Format:
# <<CAI-DSVV-FTM>>{{
#   "recommendations": [
#     {{
#       "exercise": "...",
#       "justification": "..."
#     }},
#     ...
#   ],
#   "therapy_priority": "Low|Medium|High"
# }}<<CAI-DSVV-FTM>>
# """
