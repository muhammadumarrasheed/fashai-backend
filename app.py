from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor
import logging
import uuid
from typing import Optional
from dotenv import load_dotenv
import os
from google.api_core.exceptions import ResourceExhausted

from vertexai import init as vertexai_init
from vertexai.generative_models import GenerativeModel, SafetySetting, Part
from fastapi.middleware.cors import CORSMiddleware

# ============ Load Environment Variables ============
load_dotenv()

# ============ FastAPI App Initialization ============
app = FastAPI()

# ============ CORS Configuration ============
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace '*' with your frontend's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============ Logging Setup ============
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ ThreadPoolExecutor for Multitasking ============
executor = ThreadPoolExecutor(max_workers=10)

# ============ Vertex AI Model Initialization ============
PROJECT_ID = os.getenv("PROJECT_ID", "your-project-id")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not GOOGLE_APPLICATION_CREDENTIALS or not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
    raise EnvironmentError("Invalid GOOGLE_APPLICATION_CREDENTIALS path. Check your .env file.")

# Initialize Vertex AI
vertexai_init(project=PROJECT_ID, location="europe-west2")

model = GenerativeModel("gemini-1.5-flash")

# ============ Safety Settings ============
safety_settings = [
    SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
    SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
    SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
    SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF"),
]

# ============ In-Memory Conversation Store ============
context_store = {}

def generate_thread_id() -> str:
    """Generate a unique thread ID using UUID."""
    return str(uuid.uuid4())

def build_prompt(
        conversation_history: list,
        assistant_name: str,
        assistant_description: str,
        new_user_message: str
) -> str:
    """
    Build a combined prompt from:
      - conversation_history (list of strings, representing past messages)
      - assistant_name
      - assistant_description
      - new_user_message (current user request)
    """
    history_text = "\n".join(conversation_history)
    prompt = (
        f"Conversation so far:\n{history_text}\n\n"
        f"You are an assistant named {assistant_name}. "
        f"Description: {assistant_description}\n\n"
        f"User says: {new_user_message}\n"
        "Please respond accordingly."
    )
    return prompt

@app.post("/create_assistant_advice")
async def create_assistant_advice(
        file: UploadFile = File(...),
        assistant_name: str = Form(...),
        assistant_description: str = Form(...),
        thread_id: Optional[str] = Form(None),
):
    """
    API for users uploading an image for assistant analysis.
    """
    try:
        if not thread_id:
            thread_id = generate_thread_id()
            context_store[thread_id] = []
        else:
            if thread_id not in context_store:
                context_store[thread_id] = []

        image_contents = await file.read()
        image_part = Part.from_data(mime_type="image/jpeg", data=image_contents)

        user_request = (
            "Please analyze the uploaded image and provide feedback specific to it. "
            "Focus on aspects like color coordination, style, fit, and overall aesthetic appeal. "
            "Provide a concise response in plain text, max 150 words."
        )

        prompt = build_prompt(
            conversation_history=context_store[thread_id],
            assistant_name=assistant_name,
            assistant_description=assistant_description,
            new_user_message=user_request
        )

        response_generator = model.generate_content(
            [image_part, prompt],
            generation_config={"max_output_tokens": 3873, "temperature": 1, "top_p": 0.95},
            safety_settings=safety_settings,
            stream=True
        )

        response_text = ""
        for part in response_generator:
            response_text += part.text

        response_text = response_text.strip()

        context_store[thread_id].append(f"User (image request): {user_request}")
        context_store[thread_id].append(f"Assistant: {response_text}")

        return JSONResponse(
            content={
                "assistant_advice": response_text,
                "thread_id": thread_id
            }
        )

    except ResourceExhausted as quota_error:
        logger.error("Quota exceeded: %s", quota_error)
        return JSONResponse(
            content={
                "error": "Quota exceeded. Please wait before making more requests, or contact support to increase your quota.",
                "details": str(quota_error),
            },
            status_code=429,  # HTTP 429 Too Many Requests
        )
    except Exception as e:
        logger.error(f"Error in create_assistant_advice: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/create_assistant_advice_text")
async def create_assistant_advice_text(
        assistant_name: str = Form(...),
        assistant_description: str = Form(...),
        user_text: str = Form(...),
        thread_id: Optional[str] = Form(None),
):
    """
    API for users who only send text for assistant analysis.
    """
    try:
        if not thread_id:
            thread_id = generate_thread_id()
            context_store[thread_id] = []
        else:
            if thread_id not in context_store:
                context_store[thread_id] = []

        prompt = build_prompt(
            conversation_history=context_store[thread_id],
            assistant_name=assistant_name,
            assistant_description=assistant_description,
            new_user_message=user_text
        )

        response_generator = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": 3873, "temperature": 1, "top_p": 0.95},
            safety_settings=safety_settings,
            stream=True
        )

        response_text = "".join(part.text for part in response_generator)

        context_store[thread_id].append(f"User (text request): {user_text}")
        context_store[thread_id].append(f"Assistant: {response_text}")

        return JSONResponse(
            content={
                "assistant_response": response_text,
                "thread_id": thread_id
            }
        )

    except ResourceExhausted as quota_error:
        logger.error("Quota exceeded: %s", quota_error)
        return JSONResponse(
            content={
                "error": "Quota exceeded. Please wait before making more requests, or contact support to increase your quota.",
                "details": str(quota_error),
            },
            status_code=429,  # HTTP 429 Too Many Requests
        )
    except Exception as e:
        logger.error(f"Error in create_assistant_advice_text: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": str(e)}, status_code=500)
