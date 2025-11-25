from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from itertools import cycle
from src.document_processor import DocumentProcessor
from src.chat_manager import ChatManager
from src.vector_store import VectorStore
from src.audio_service import AudioService
from src.scraper import WebScraper, ScrapeConfig
from flask import Response, stream_with_context
from flask_cors import CORS

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize components
vector_store = VectorStore()
document_processor = DocumentProcessor(vector_store)
chat_manager = ChatManager(vector_store)
audio_service = AudioService()
scraper = WebScraper(vector_store)

PREDEFINED_ANSWERS = [
    {
        "question": "Executive benefits overview",
        "answer": (
            "**Executive Onboarding Snapshot**\n"
            "\n"
            "Welcome to Advantis! Your Executive benefits include:\n"
            "\n"
            "- **Medical**: Outpatient reimbursement, hospitalization & surgical cover, critical illness protection, Healthscan mobile ambulance, spectacle reimbursement.\n"
            "- **Life & Accident**: Group Term Life Assurance plus Personal Accident Insurance.\n"
            "- **Statutory**: Private Provident Fund, Employees’ Trust Fund, and gratuity after 5 years.\n"
            "- **Loans**: Personal, academic/professional, motorcycle, and PPF-backed housing/education facilities.\n"
            "- **Recreation**: Annual company trips, holiday bungalows, hotel discounts, subsidized gym memberships.\n"
            "- **HGRC Membership**: Funeral fund, wedding & newborn gifts, educational achievement rewards, blood donation support, and community events.\n"
            "- **Recognition & Growth**: Long-service awards, examination gifts, and support for relevant professional memberships.\n"
            "- **Other Perks**: Company wedding gift, club membership eligibility, holiday/night meal allowances (role dependent).\n"
            "\n"
            "Reach out to HR anytime you need help activating a benefit."
        ),
    },
    {
        "question": "Raise IT support ticket for Outlook access",
        "answer": (
            "**Need Outlook Access?**\n"
            "\n"
            "1. Open the IT request portal: <https://helpdesk.hayleysadvantis.com/app/itdesk/HomePage.do>.\n"
            "2. Click **New Request** in the top bar.\n"
            "3. Complete the form and note the issue as “Not receiving emails in Outlook / Need access enabled”.\n"
            "4. Submit the ticket so the IT Support Team can review and respond.\n"
            "\n"
            "Tip: attach screenshots or error messages to speed up troubleshooting."
        ),
    },
    {
        "question": "GMC members overview",
        "answer": (
            "**Group Management Committee (GMC) Snapshot**\n"
            "\n"
            "- **Managing Director**: Ruwan Waidyaratne\n"
            "- **Deputy Managing Directors**: Asanka Ratnayake, Shano Sabar\n"
            "- **Board Directors**: Janitha Jayanetti, Virendra Perera, Binupa Liyanage, Vishwanath Daluwatte\n"
            "- **GMC Members**: Gerard Victoria, Shamindra Wickremesooriya, Chintaka De Zoysa, Sheran Abeysundere, Sagara Peiris, Chamila Bandara, Arosha Fernando, Tharanga Perera\n"
            "\n"
            "These leaders steer Advantis strategy, investments, and cross-division alignment."
        ),
    },
]

predefined_cycle = cycle(PREDEFINED_ANSWERS)

@app.route('/upload', methods=['POST'])
def upload_document():
    """
    Endpoint to upload and process documents
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        document_processor.process_document(file)
        return jsonify({'message': 'Document processed successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/stream', methods=['GET'])
def chat_stream():
    """Server-Sent Events stream of answer tokens.
    Query params: question, session_id (optional).
    """
    question = request.args.get('question')
    session_id = request.args.get('session_id')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        sid, token_gen = chat_manager.stream_response(question, session_id)

        def sse():
            for tok in token_gen:
                # Normalize token: send exactly as received from LLM
                # Frontend should concatenate tokens directly without adding spaces
                # GPT-4 already includes proper spacing in tokens
                yield f"data: {tok}\n\n"
            yield f"event: done\ndata: {sid}\n\n"

        headers = {
            'Content-Type': 'text/event-stream',
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive'
        }
        return Response(stream_with_context(sse()), headers=headers)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/scrape', methods=['POST'])
def scrape():
    """Crawl a website and index textual content into the vector store.
    Body JSON: { "url": "https://site" , "max_pages": 50, "max_depth": 3 }
    """
    data = request.json or {}
    url = data.get('url')
    if not url:
        return jsonify({'error': 'Missing url'}), 400
    cfg = ScrapeConfig(
        max_pages=int(data.get('max_pages', 50)),
        max_depth=int(data.get('max_depth', 3)),
    )
    try:
        # Do not index by default; return scraped items so user can inspect
        result = scraper.scrape_and_index(url, cfg, index=bool(data.get('index', False)), include_text=bool(data.get('include_text', True)))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint to handle chat interactions
    """
    data = request.json
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    session_id = data.get('session_id', None)
    question = data['question']
    
    try:
        response = chat_manager.get_response(question, session_id)
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/chat/history', methods=['GET'])
def get_chat_history():
    """
    Endpoint to retrieve chat history for a session
    """
    session_id = request.args.get('session_id')
    if not session_id:
        return jsonify({'error': 'No session ID provided'}), 400
    
    try:
        history = chat_manager.get_chat_history(session_id)
        return jsonify(history), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stt', methods=['POST'])
def speech_to_text():
    """Speech-to-Text: accepts a file field 'audio'. Language is fixed to English."""
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    audio = request.files['audio']
    if audio.filename == '':
        return jsonify({'error': 'No audio file selected'}), 400
    try:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            tmp_path = tmp.name
            audio.save(tmp_path)
        text = audio_service.speech_to_text(tmp_path)
        os.unlink(tmp_path)
        return jsonify({'text': text}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Text-to-Speech: accepts JSON {"text": "...", "voice": "alloy"} and returns mp3 bytes."""
    data = request.json or {}
    text = data.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    voice = data.get('voice', 'alloy')
    try:
        audio_bytes = audio_service.text_to_speech(text, voice=voice)
        from flask import Response
        return Response(audio_bytes, mimetype='audio/mpeg')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predefined-answer', methods=['GET'])
def get_predefined_answer():
    """Return one of three hardcoded answers in round-robin order."""
    # Use next() so repeated calls cycle through the predefined list
    answer_payload = next(predefined_cycle)
    # Return a shallow copy to avoid exposing iterator internals
    return jsonify(dict(answer_payload)), 200

if __name__ == '__main__':
    app.run(debug=True)
