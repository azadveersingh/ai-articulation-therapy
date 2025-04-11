import streamlit as st
from audiototext import audio_to_text_whisper
from process_text import *
import json
from typing import List, Dict, Optional

def process_inputs(audio_path: str, 
                  original_text: str,
                  model_paths: List[str]) -> Optional[Dict]:
    """Full processing pipeline with evaluation."""
    # 1. Audio Transcription
    raw_text = audio_to_text_whisper(audio_path)
    transcribed_text = clean_text(raw_text)
    if not transcribed_text:
        st.error("Audio transcription failed")
        return None
    
    # 2. Load all models
    models = [load_model(path) for path in model_paths]
    if not all(models):
        st.error("Some models failed to load")
        return None
    
    # 3. Generate IPA variants (3 for original, 3 for transcribed)
    original_ipas = []
    transcribed_ipas = []
    
    with st.spinner("Generating IPA variants..."):
        # Generate 3 original IPAs
        for i, model in enumerate(models[:3]):
            ipa = generate_ipa(model, original_text)
            if ipa:
                original_ipas.append(ipa)
                st.write(f"Original IPA {i+1}: {ipa}")
        
        # Generate 3 transcribed IPAs
        for i, model in enumerate(models[:3]):
            ipa = generate_ipa(model, transcribed_text)
            if ipa:
                transcribed_ipas.append(ipa)
                st.write(f"Transcribed IPA {i+1}: {ipa}")
    
    if len(original_ipas) < 3 or len(transcribed_ipas) < 3:
        st.error("Insufficient IPA variants generated")
        return None
    
    # 4. Evaluation with 4th model
    with st.spinner("Evaluating transcriptions..."):
        evaluation = evaluate_transcriptions(
            models[3],  # Evaluation model
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
    
    # 5. SODA Analysis
    with st.spinner("Performing SODA analysis..."):
        # Get the selected IPAs from evaluation
        best_original_ipa = evaluation["best_ipa_original"]
        best_transcribed_ipa = evaluation["best_ipa_transcript"]
        
        # Generate 3 SODA analyses using first 3 models
        soda_analyses = []
        for i, model in enumerate(models[:3]):
            analysis = analyze_articulation_errors(
                model,
                original_text,
                best_original_ipa,
                transcribed_text,
                best_transcribed_ipa
            )
            soda_analyses.append(analysis)
            st.write(f"SODA Analysis {i+1}: {json.dumps(analysis, indent=2, ensure_ascii=False)}")
        
        # Evaluate SODA analyses with 4th model
        soda_evaluation = evaluate_soda_analyses(
            models[3],
            soda_analyses
        )
        
        # Add to results
        results["soda_analyses"] = soda_analyses
        results["soda_evaluation"] = soda_evaluation
    
    # 6. Generate Final SODA Summary
    with st.spinner("Generating SODA summary..."):
        selected_soda_analysis = soda_evaluation["consolidated_analysis"]
        soda_summary = generate_soda_summary(
            models[3],  # Use the evaluation model
            original_text,
            transcribed_text,
            best_original_ipa,
            best_transcribed_ipa,
            selected_soda_analysis
        )
        results["soda_summary"] = soda_summary

    return results

# ===== STREAMLIT UI =====

# if __name__ == '__main__':
#     st.set_page_config(layout="wide", page_title="Advanced IPA Transcriber")
#     st.title("🔊 Multi-Model IPA Transcription")

# #     model_paths = [
# #     "/media/cairuser1/b916a6fe-2106-41d4-98be-bbdd1d3bcb16/model/llama-chat-3.1-q8.gguf",  # Model 1
# #     "/media/cairuser1/b916a6fe-2106-41d4-98be-bbdd1d3bcb16/model/llama-2-7b-chat.Q8_0.gguf",  # Model 2
# #     "/media/cairuser1/b916a6fe-2106-41d4-98be-bbdd1d3bcb16/model/Ministral-8B-Instruct-2410-Q8_0.gguf",  # Model 3
# #     "/media/cairuser1/b916a6fe-2106-41d4-98be-bbdd1d3bcb16/model/Ministral-8B-Instruct-2410-Q8_0.gguf"  # Model 4
# # ]
#     model_paths = [
#     "model/llama-chat-3.1-q8.gguf",  # Model 1
#     "model/llama-chat-3.1-q8.gguf",  # Model 1
#     "model/llama-chat-3.1-q8.gguf",  # Model 1
#     "model/llama-chat-3.1-q8.gguf",  # Model 1
#     ]

#     col1, col2 = st.columns(2)
#     with col1:
#         original_text = st.text_area(
#             "Reference Text",
#             value="I saw Sam sitting on a bus",
#             height=100
#         )
#     with col2:
#         audio_file = st.file_uploader(
#             "Upload Audio (WAV/MP3)",
#             type=["wav", "mp3"]
#         )

#     if audio_file and original_text.strip():
#         with st.status("Processing...", expanded=True) as status:
#             results = process_inputs(
#                 audio_file,
#                 original_text,
#                 model_paths
#             )
            
#             if results:
#                 status.update(label="Processing Complete", state="complete")
#                 st.divider()
#                 st.subheader("Final Evaluation")
                
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.markdown("**Original Text**")
#                     st.code(results["original_text"], language="text")
#                     st.markdown("**Selected Original IPA**")
#                     st.code(results["evaluation"]["best_ipa_original"], language="text")
                
#                 with col2:
#                     st.markdown("**Transcribed Text**")
#                     st.code(results["transcribed_text"], language="text")
#                     st.markdown("**Selected Transcribed IPA**")
#                     st.code(results["evaluation"]["best_ipa_transcript"], language="text")
#                     st.markdown(f"*Confidence: {results['evaluation']['confidence']}/10*")
        
#         if results:
#             with st.expander("All IPA Variants"):
#                 st.markdown("**Original IPAs**")
#                 for i, ipa in enumerate(results["original_ipas"]):
#                     st.write(f"Model {i+1}: {ipa}")
                
#                 st.markdown("**Transcribed IPAs**")
#                 for i, ipa in enumerate(results["transcribed_ipas"]):
#                     st.write(f"Model {i+1}: {ipa}")
            
#             st.divider()
#             st.subheader("Articulation Error Analysis (SODA)")
            
#             selected_idx = results["soda_evaluation"]["selected_analysis"]
#             st.json(results["soda_analyses"][selected_idx])
            
#             st.markdown(f"*Confidence: {results['soda_evaluation']['confidence']}/10*")
            
#             with st.expander("All SODA Analyses"):
#                 for i, analysis in enumerate(results["soda_analyses"]):
#                     st.write(f"Analysis {i+1}:")
#                     st.json(analysis)
            
#             st.divider()
#             st.subheader("Final SODA Summary")
#             st.json(results["soda_summary"])
#             st.markdown(f"*Summary Confidence: {results['soda_summary']['confidence']}/10*")

# import streamlit as st
# from audiototext import audio_to_text_whisper
# from process_text import *

# # ... (existing imports and functions remain unchanged)

# # ===== STREAMLIT UI =====

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Advanced IPA Transcriber")
    st.title("🔊 Multi-Model IPA Transcription")

    model_paths = [
        "model/llama-chat-3.1-q8.gguf",
        "model/llama-chat-3.1-q8.gguf",
        "model/llama-chat-3.1-q8.gguf",
        "model/llama-chat-3.1-q8.gguf",
    ]

    # Existing columns for original text and audio upload
    col1, col2 = st.columns(2)
    with col1:
        original_text = st.text_area(
            "Reference Text",
            value="I saw Sam sitting on a bus",
            height=100
        )
    with col2:
        audio_file = st.file_uploader(
            "Upload Audio (WAV/MP3)",
            type=["wav", "mp3"]
        )

    # New section for the form
    st.header("Speech Assessment Form")
    with st.form(key="speech_assessment_form"):
        # Personal Information
        st.subheader("Personal Information")
        full_name = st.text_input("Full Name")
        gender = st.radio("Gender:", ["Male", "Female", "Other"])
        native_language = st.text_input("Native Language")
        prior_speech_therapy = st.radio("Prior Speech Therapy Experience?", ["Yes", "No"])

        # Self-Assessment of Speech Difficulties
        st.subheader("Self-Assessment of Speech Difficulties")
        self_assess_freq = st.radio(
            "How often do people ask you to repeat yourself?",
            ["Never", "Rarely", "Sometimes", "Often", "Always"]
        )
        difficulty_pronounce = st.radio(
            "Do you find it difficult to pronounce certain sounds or words?",
            ["Yes", "No"]
        )
        if difficulty_pronounce == "Yes":
            difficulty_type = st.multiselect(
                "If yes, which type of difficulty do you experience the most? (Select all that apply)",
                ["Substitutions (Replacing one sound with another, e.g., 'wabbit' instead of 'rabbit')",
                 "Omissions/Deletions (Leaving out sounds, e.g., 'cat' instead of 'cat')",
                 "Distortions (Producing sounds incorrectly, e.g., slurred sound of 'black')",
                 "Additions (Adding extra sounds, e.g., 'buhlack' instead of 'black')"]
            )
        speech_impact = st.radio(
            "How much does your speech difficulty affect your daily life?",
            ["Not at all", "Slightly", "Moderately", "Significantly", "Severely"]
        )

        # Psychological & Emotional Impact
        st.subheader("Psychological & Emotional Impact")
        anxious_speaking = st.radio(
            "Do you feel self-conscious or anxious while speaking?",
            ["Never", "Rarely", "Sometimes", "Often", "Always"]
        )
        avoid_difficulties = st.radio(
            "Have you avoided social interactions due to speech difficulties?",
            ["Never", "Rarely", "Sometimes", "Often", "Always"]
        )
        misunderstood = st.radio(
            "Do you feel frustrated when people don't understand you?",
            ["Never", "Rarely", "Sometimes", "Often", "Always"]
        )

        # Speech Context and Triggers
        st.subheader("Speech Context and Triggers")
        context_struggle = st.multiselect(
            "In which situations do you struggle the most?",
            ["Talking to friends/family", "Professional setting", "Public speaking", "Talking on the phone", "Reading aloud"]
        )
        stressed_tired = st.radio(
            "Do you notice more difficulty when stressed or tired?",
            ["Yes", "No"]
        )

        # Speech Exercise Preferences
        st.subheader("Speech Exercise Preferences")
        exercise_type = st.multiselect(
            "What type of exercises do you prefer?",
            ["Visual-based (lip-reading)", "Audio-based (listening)", "Interactive AI-assisted", "Repetitive articulation drills"]
        )
        time_dedicate = st.radio(
            "How much time can you dedicate to speech exercises daily?",
            ["<10 min", "10-20 min", "20-30 min", ">30 min"]
        )

        # Final Comments
        st.subheader("Final Comments")
        final_comments = st.text_area("Is there anything else you'd like to share?")

        # Submit button for the form
        submit_button = st.form_submit_button(label="Submit Assessment")

    # Process form and audio data
    if audio_file and original_text.strip() and submit_button:
        with st.status("Processing...", expanded=True) as status:
            results = process_inputs(
                audio_file,
                original_text,
                model_paths
            )
            
            if results:
                # Incorporate form data into SODA summary
                form_data = {
                    "full_name": full_name,
                    "gender": gender,
                    "native_language": native_language,
                    "prior_speech_therapy": prior_speech_therapy,
                    "self_assess_freq": self_assess_freq,
                    "difficulty_pronounce": difficulty_pronounce,
                    "difficulty_type": difficulty_type if difficulty_pronounce == "Yes" else [],
                    "speech_impact": speech_impact,
                    "anxious_speaking": anxious_speaking,
                    "avoid_difficulties": avoid_difficulties,
                    "misunderstood": misunderstood,
                    "context_struggle": context_struggle,
                    "stressed_tired": stressed_tired,
                    "exercise_type": exercise_type,
                    "time_dedicate": time_dedicate,
                    "final_comments": final_comments
                }

                # Update the soda_summary with form data
                results["soda_summary"] = generate_soda_summary(
                    model=load_model(model_paths[3]),  # Using the evaluation model
                    original_text=results["original_text"],
                    transcribed_text=results["transcribed_text"],
                    best_ipa_original=results["evaluation"]["best_ipa_original"],
                    best_ipa_transcript=results["evaluation"]["best_ipa_transcript"],
                    soda_analysis=results["soda_evaluation"]["consolidated_analysis"],
                    psychological_profile=form_data  # Pass form data as additional input
                )

                status.update(label="Processing Complete", state="complete")
                st.divider()
                st.subheader("Final Evaluation")

                # ... (rest of the existing display logic remains unchanged)

                # Update SODA Summary display with form influence
                st.divider()
                st.subheader("Final SODA Summary")
                st.json(results["soda_summary"])
                st.markdown(f"*Summary Confidence: {results['soda_summary']['confidence']}/10*")

# ... (rest of the script remains unchanged)