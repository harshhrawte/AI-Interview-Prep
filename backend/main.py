import os
import uuid
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
import faiss
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
print(f"[DEBUG] GEMINI_API_KEY loaded: {GEMINI_API_KEY}")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
session_store = {}

# Embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Gemini setup
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

def extract_text_from_pdf(file) -> str:
    """Extract text from a PDF file."""
    reader = PdfReader(io.BytesIO(file))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text

def chunk_text(text, chunk_size=500):
    """Split text into chunks of a given size (words)."""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...)):
    print("[DEBUG] Received upload request")
    try:
        contents = await file.read()
        print(f"[DEBUG] Read file bytes, size: {len(contents)} bytes")
        text = extract_text_from_pdf(contents)
        print(f"[DEBUG] Extracted text from PDF, length: {len(text)} characters")
        chunks = chunk_text(text)
        print(f"[DEBUG] Chunked text into {len(chunks)} chunks")

        embeddings = embedder.encode(chunks)
        print(f"[DEBUG] Created embeddings, shape: {embeddings.shape}")

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        print("[DEBUG] FAISS index created and embeddings added")

        session_id = str(uuid.uuid4())
        session_store[session_id] = {
            "faiss_index": index,
            "chunks": chunks,
            "embeddings": embeddings
        }
        print(f"[DEBUG] Session stored with ID: {session_id}")

        return {"session_id": session_id}
    except Exception as e:
        print(f"[ERROR] Upload error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to process PDF: {str(e)}"})

@app.post("/generate_questions")
async def generate_questions(session_id: str = Form(...), job_description: str = Form(...)):
    if session_id not in session_store:
        return JSONResponse(status_code=404, content={"error": "Session not found"})
    try:
        # Extract keywords from job description
        keywords = [w.strip('.,') for w in job_description.split() if len(w) > 3]

        # Embed keywords
        keyword_embeds = embedder.encode([" ".join(keywords)])

        # Query FAISS for top 3 relevant resume chunks
        index = session_store[session_id]["faiss_index"]
        D, I = index.search(keyword_embeds, 3)
        relevant_chunks = [
            session_store[session_id]["chunks"][i]
            for i in I[0] if i < len(session_store[session_id]["chunks"])
        ]

        # Build prompt safely (avoid f-string backslash error)
        resume_context = "\n".join(relevant_chunks)
        prompt = f"""
You are an interview coach. Based on the following resume and job description, generate 10 tailored interview preparation questions:

Resume Context:
{resume_context}

Job Description:
{job_description}
"""

        # Call Gemini
        if not GEMINI_API_KEY:
            return JSONResponse(status_code=500, content={"error": "Gemini API key not set."})

        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)

        # Ensure response is parsed safely
        questions = [q.strip() for q in response.text.split("\n") if q.strip()]

        return {"questions": questions}
    except Exception as e:
        print(f"[ERROR] Generate questions error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Failed to generate questions: {str(e)}"})
