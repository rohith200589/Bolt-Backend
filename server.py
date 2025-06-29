from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import google.generativeai as genai
import requests
from dotenv import load_dotenv
import re
import json
import io
import tempfile
from PyPDF2 import PdfReader
from docx import Document
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable
import traceback # For detailed error logging

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# --- API Configurations ---

# Configure Google Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in a .env file.")
genai.configure(api_key=GEMINI_API_KEY)

# Use gemini-2.0-flash as specified for general text generation and test generation
gemini_model_text_gen = genai.GenerativeModel('gemini-2.0-flash')
gemini_model_test_gen = genai.GenerativeModel('gemini-2.0-flash') # Can use the same model instance

# Configure Eleven Labs API
ELEVEN_LABS_API_KEY = os.getenv("ELEVEN_LABS_API_KEY")
if not ELEVEN_LABS_API_KEY:
    raise ValueError("ELEVEN_LABS_API_KEY not found in environment variables.")

# Choose a voice ID from Eleven Labs (e.g., Rachel)
ELEVEN_LABS_VOICE_ID = os.getenv("ELEVEN_LABS_VOICE_ID")

# --- Helper Functions for Content Extraction ---

def extract_text_from_pdf(pdf_file_path: str) -> str:
    """Extracts text from a PDF file."""
    text = ""
    try:
        with open(pdf_file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or ""
    except Exception as e:
        raise ValueError(f"Could not read PDF file: {e}")
    return text

def extract_text_from_docx(docx_file_path: str) -> str:
    """Extracts text from a DOCX file."""
    text = ""
    try:
        document = Document(docx_file_path)
        for paragraph in document.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        raise ValueError(f"Could not read DOCX file: {e}")
    return text

def get_youtube_transcript(youtube_url: str) -> str:
    """Fetches the transcript from a YouTube video URL."""
    video_id = None
    try:
        # Robustly extract video ID from common YouTube URL formats
        # This regex handles watch?v=, embed/, v/, and youtu.be/ formats.
        match = re.search(r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})(?:\S+)?', youtube_url)
        if match:
            video_id = match.group(1)

        if not video_id:
            raise ValueError(f"Could not extract a valid 11-character YouTube video ID from URL: {youtube_url}. Please ensure it's a standard YouTube video link.")

        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        transcript = " ".join([t['text'] for t in transcript_list])
        return transcript

    except NoTranscriptFound:
        raise ValueError("No transcript found for this YouTube video. It might be disabled by the uploader.")
    except TranscriptsDisabled:
        raise ValueError("Transcripts are disabled for this video.")
    except VideoUnavailable:
        raise ValueError("The video is unavailable or private.")
    except Exception as e:
        # Catch any other unexpected errors during transcript fetching
        raise ValueError(f"Error fetching YouTube transcript: {e}")

# --- Flask Routes ---

@app.route('/')
def home():
    """Basic home route to confirm backend is running."""
    return "Knovia AI Backend is running!"

@app.route('/generate-text', methods=['POST'])
def generate_text():
    """
    Generates text using the Google Gemini API based on a provided question.
    Cleans up markdown formatting from the response.
    """
    data = request.get_json()
    question = data.get('question')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        gemini_prompt = f"The user asked: '{question}'. Provide a clear, concise answer in under 200 words."
        gemini_response = gemini_model_text_gen.generate_content(gemini_prompt)
        raw_text = gemini_response.text.strip()

        # Clean markdown/formatting characters
        clean_text = re.sub(r"[*_`~#>-]", "", raw_text)
        clean_text = re.sub(r"\\n+", "\n", clean_text).strip() # Use single newline for readability

        return jsonify({"text": clean_text})
    except Exception as e:
        print(f"Gemini text generation error: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    """
    Converts text to speech using the ElevenLabs API.
    Streams the audio response back to the client.
    """
    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "Text is required"}), 400

    try:
        eleven_labs_url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_LABS_VOICE_ID}/stream"

        headers = {
            "Accept": "audio/mpeg",
            "xi-api-key": ELEVEN_LABS_API_KEY,
            "Content-Type": "application/json"
        }

        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1", # Changed from 'eleven_monolingual_v1' to 'eleven_turbo_v2' if available and preferred
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75
            }
        }

        response = requests.post(eleven_labs_url, headers=headers, json=payload, stream=True)

        if response.status_code == 200:
            # Stream the audio content directly to the client
            return Response(stream_with_context(response.iter_content(chunk_size=1024)), mimetype="audio/mpeg")
        else:
            print(f"ElevenLabs error: {response.text}")
            return jsonify({"error": response.text}), response.status_code

    except Exception as e:
        print(f"Audio generation error: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": str(e)}), 500

@app.route('/generate_test', methods=['POST'])
def generate_test():
    """
    Generates a mock test (MCQ, True/False, or Fill-in-the-Blanks)
    from various content types (topic, file, text, YouTube URL) using Gemini.
    """
    try:
        # Determine if the request is JSON or FormData
        if request.is_json:
            data = request.json
        else:
            data = request.form # For form data (including file uploads via multipart/form-data)

        content_type = data.get('type')
        difficulty = data.get('difficulty')
        num_questions = int(data.get('numQuestions'))
        # Ensure boolean conversion for include_explanations
        include_explanations = str(data.get('includeExplanations')).lower() == 'true'
        question_type = data.get('questionType')
        time_limit = int(data.get('timeLimit'))

        extracted_content = ""

        if content_type == 'topic':
            topic = data.get('content')
            if not topic:
                return jsonify({"error": "Topic content is required."}), 400
            extracted_content = f"Topic: {topic}"

        elif content_type == 'file':
            if 'file' not in request.files:
                return jsonify({"error": "No file part in the request."}), 400
            file = request.files['file']
            if file.filename == '':
                return jsonify({"error": "No selected file."}), 400

            # Use a temporary file to save the uploaded content
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                file.save(tmp_file.name)
                temp_file_path = tmp_file.name

            try:
                file_extension = os.path.splitext(file.filename)[1].lower()
                if file_extension == '.pdf':
                    extracted_content = extract_text_from_pdf(temp_file_path)
                elif file_extension in ['.docx', '.doc']: # Basic .doc handling, can be limited
                    extracted_content = extract_text_from_docx(temp_file_path)
                elif file_extension == '.txt':
                    with open(temp_file_path, 'r', encoding='utf-8') as f:
                        extracted_content = f.read()
                else:
                    return jsonify({"error": "Unsupported file type. Please upload PDF, DOCX, DOC, or TXT."}), 400
            finally:
                os.remove(temp_file_path) # Clean up the temporary file

        elif content_type == 'text':
            pasted_text = data.get('content')
            if not pasted_text:
                return jsonify({"error": "Pasted text content is required."}), 400
            extracted_content = pasted_text

        elif content_type == 'youtube':
            youtube_url = data.get('content')
            if not youtube_url:
                return jsonify({"error": "YouTube URL is required."}), 400
            extracted_content = get_youtube_transcript(youtube_url)

        else:
            return jsonify({"error": "Invalid content type specified."}), 400

        if not extracted_content.strip():
            return jsonify({"error": "Could not extract sufficient content for test generation. Content might be empty or unreadable."}), 400

        # Construct the prompt for Gemini
        prompt_parts = [
            f"You are an AI assistant designed to create mock test questions. Generate a mock test based on the following content:\n\n",
            f"Content: {extracted_content}\n\n",
            f"Instructions:\n",
            f"- Create {num_questions} {question_type} questions.\n",
            f"- The difficulty should be {difficulty}.\n",
            f"- For each question, provide a unique ID (integer), the question text, an array of options (for MCQs), and the correct answer.\n"
        ]

        if question_type == 'mcq':
            prompt_parts.append("- For MCQ questions, provide exactly 4 options.\n")
        elif question_type == 'trueFalse':
            prompt_parts.append("- For True/False questions, options should be ['True', 'False'].\n")
        elif question_type == 'fillInTheBlanks':
            prompt_parts.append("- For Fill-in-the-Blanks questions, the question should contain a blank indicated by '___', and the correct answer should be the word or phrase that fills the blank. Do not provide options.\n")


        if include_explanations:
            prompt_parts.append("- Include a brief explanation for the correct answer for each question.\n")

        prompt_parts.append(
            f"- Format the output as a JSON object. Ensure the JSON is valid and complete. Do not include any extra text or markdown outside the JSON.\n"
            "```json\n"
            "{\n"
            "   \"title\": \"[Generated Test Title]\",\n"
            "   \"description\": \"[Generated Test Description]\",\n"
            "   \"questions\": [\n"
            "     {\n"
            "       \"id\": 1,\n"
            "       \"question\": \"[Question text]?\",\n"
            "       \"options\": [\"Option A\", \"Option B\", \"Option C\", \"Option D\"],\n"
            "       \"correctAnswer\": \"[Correct Option]\",\n"
            "       \"explanation\": \"[Explanation for correct answer]\"\n"
            "     },\n"
            "     // ... more questions\n"
            "   ]\n"
            "}\n"
            "```\n"
            "Ensure 'options' array is only for MCQ/TrueFalse. For Fill-in-the-Blanks, omit 'options'."
        )

        full_prompt = "".join(prompt_parts)

        # Generate content using Gemini
        response = gemini_model_test_gen.generate_content(full_prompt)
        raw_gemini_response_text = response.text.strip() # Strip leading/trailing whitespace

        quiz_data = {}
        # Attempt to parse directly first
        try:
            quiz_data = json.loads(raw_gemini_response_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from markdown block
            match = re.search(r'```json\s*(\{.*\})\s*```', raw_gemini_response_text, re.DOTALL)
            if match:
                json_string = match.group(1)
                try:
                    quiz_data = json.loads(json_string)
                except json.JSONDecodeError as e:
                    print(f"JSON Decode Error (after regex extraction): {e}")
                    print(f"Extracted JSON string (first 500 chars): {json_string[:500]}...")
                    return jsonify({"error": "Failed to parse AI response. Invalid JSON format after markdown extraction."}), 500
            else:
                # If no markdown block found, it might still be malformed or empty
                print(f"JSON Decode Error: No direct JSON or markdown JSON block found.")
                print(f"Raw Gemini Response (no JSON or markdown block detected, first 500 chars): {raw_gemini_response_text[:500]}...")
                return jsonify({"error": "Failed to parse AI response. Unexpected response format from AI. It did not return valid JSON."}), 500

        # Add timeLimitMinutes from frontend settings, as Gemini doesn't set this directly
        quiz_data['timeLimitMinutes'] = time_limit
        print(quiz_data)

        return jsonify(quiz_data)

    except ValueError as ve:
        # Catches errors like invalid integer conversion, missing required fields, extraction errors
        print(f"Validation Error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Catch any other unexpected errors
        print(f"An unexpected server error occurred: {e}")
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({"error": f"An internal server error occurred: {str(e)}. Please check backend logs."}), 500

if __name__ == '__main__':
    # For development, run on localhost:5000
    # In a production environment, use a production-ready WSGI server like Gunicorn or uWSGI
    app.run(host='0.0.0.0', port=5000, debug=True)

