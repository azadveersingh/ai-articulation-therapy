# app.py
import streamlit as st
from process_text import *
from llama_model import LlamaModelManager

if __name__ == '__main__':
    st.set_page_config(layout="wide", page_title="Advanced IPA Transcriber")
    st.title("🔊 Multi-Model IPA Transcription")

    model_paths = [
        "model/llama-chat-3.1-q8.gguf",
        "model/llama-chat-3.1-q8.gguf",
        "model/llama-chat-3.1-q8.gguf",
        "model/llama-chat-3.1-q8.gguf",
    ]

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

    st.header("Speech Assessment Form")
    with st.form(key="speech_assessment_form"):
        st.subheader("Personal Information")
        full_name = st.text_input("Full Name")
        gender = st.radio("Gender:", ["Male", "Female", "Other"])
        native_language = st.text_input("Native Language")
        prior_speech_therapy = st.radio("Prior Speech Therapy Experience?", ["Yes", "No"])

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

        st.subheader("Speech Context and Triggers")
        context_struggle = st.multiselect(
            "In which situations do you struggle the most?",
            ["Talking to friends/family", "Professional setting", "Public speaking", "Talking on the phone", "Reading aloud"]
        )
        stressed_tired = st.radio(
            "Do you notice more difficulty when stressed or tired?",
            ["Yes", "No"]
        )

        st.subheader("Speech Exercise Preferences")
        exercise_type = st.multiselect(
            "What type of exercises do you prefer?",
            ["Visual-based (lip-reading)", "Audio-based (listening)", "Interactive AI-assisted", "Repetitive articulation drills"]
        )
        time_dedicate = st.radio(
            "How much time can you dedicate to speech exercises daily?",
            ["<10 min", "10-20 min", "20-30 min", ">30 min"]
        )

        st.subheader("Final Comments")
        final_comments = st.text_area("Is there anything else you'd like to share?")

        submit_button = st.form_submit_button(label="Submit Assessment")

    if audio_file and original_text.strip() and submit_button:
        try:
            with st.status("Processing...", expanded=True) as status:
                results = process_inputs(audio_file, original_text, model_paths)
                if results:
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

                    # Generate SODA summary with cleanup
                    evaluation_model = load_model(model_paths[3])
                    if evaluation_model:
                        results["soda_summary"] = generate_soda_summary(
                            model=evaluation_model,
                            original_text=results["original_text"],
                            transcribed_text=results["transcribed_text"],
                            best_ipa_original=results["evaluation"]["best_ipa_original"],
                            best_ipa_transcript=results["evaluation"]["best_ipa_transcript"],
                            soda_analysis=results["soda_evaluation"]["consolidated_analysis"],
                            psychological_profile=form_data
                        )
                        LlamaModelManager.cleanup()  # Clean up evaluation model

                    status.update(label="Processing Complete", state="complete")
            st.divider()
            st.subheader("Final Evaluation")

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Text**")
                st.code(results["original_text"], language="text")
                st.markdown("**Selected Original IPA**")
                st.code(results["evaluation"]["best_ipa_original"], language="text")
            with col2:
                st.markdown("**Transcribed Text**")
                st.code(results["transcribed_text"], language="text")
                st.markdown("**Selected Transcribed IPA**")
                st.code(results["evaluation"]["best_ipa_transcript"], language="text")
                st.markdown(f"*Confidence: {results['evaluation']['confidence']}/10*")

            with st.expander("All IPA Variants"):
                st.markdown("**Original IPAs**")
                for i, ipa in enumerate(results["original_ipas"]):
                    st.write(f"Model {i+1}: {ipa}")
                st.markdown("**Transcribed IPAs**")
                for i, ipa in enumerate(results["transcribed_ipas"]):
                    st.write(f"Model {i+1}: {ipa}")

            st.divider()
            st.subheader("Articulation Error Analysis (SODA)")
            selected_idx = results["soda_evaluation"]["selected_analysis"]
            st.json(results["soda_analyses"][selected_idx])
            st.markdown(f"*Confidence: {results['soda_evaluation']['confidence']}/10*")

            with st.expander("All SODA Analyses"):
                for i, analysis in enumerate(results["soda_analyses"]):
                    st.write(f"Analysis {i+1}:")
                    st.json(analysis)

            st.divider()
            st.subheader("Final SODA Summary")
            st.json(results["soda_summary"])
            # Note: Confidence may not be present in fallback case
            confidence = results["soda_summary"].get("confidence", 5)
            st.markdown(f"*Summary Confidence: {confidence}/10*")
        finally:
            LlamaModelManager.cleanup()  # Final cleanup to catch any stray instances
            print("Final cleanup completed in app.py.")