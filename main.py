import os
import uuid
import numpy as np
import cv2
import json
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
import firebase_admin
from firebase_admin import credentials, firestore
import cloudinary
import cloudinary.uploader

# Import your custom ML logic from model.py
from model import extract_embeddings, match_faces

app = FastAPI()
db = None
# --- 1. FIREBASE INITIALIZATION ---
try:
    if not firebase_admin._apps:
        fb_json = os.getenv("FIREBASE_SERVICE_ACCOUNT_JSON")
        if fb_json:
            # Use service account JSON if provided in Render Env
            service_account_info = json.loads(fb_json)
            cred = credentials.Certificate(service_account_info)
            firebase_admin.initialize_app(cred)
        else:
            # Fallback to default (works if configured in Firebase console)
            firebase_admin.initialize_app()
    db = firestore.client()
    print("✅ Firebase Connected")
except Exception as e:
    print(f"❌ Firebase Error: {e}")

# --- 2. CLOUDINARY CONFIGURATION ---
# Note: We only need Cloud Name for Unsigned uploads, but keeping config for SDK stability
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

# This MUST match the preset name in your Cloudinary Dashboard
UPLOAD_PRESET = "face_flutter_app"


@app.get("/")
def home():
    return {"message": "Anti Gravity Face API is Live 🚀", "mode": "Unsigned-Simple"}


# --- 3. BACKGROUND WORKER FOR POOL INGESTION ---
def process_and_index_pool(event_id: str, file_contents_list: List[bytes]):
    """Processes images in background to avoid Render timeouts."""
    for contents in file_contents_list:
        try:
            # A. Process Image
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: continue

            # B. Extract Face Vectors
            embeddings = extract_embeddings([img])
            if not embeddings: continue

            # C. Unsigned Upload to Cloudinary
            upload_result = cloudinary.uploader.unsigned_upload(
                contents,
                upload_preset=UPLOAD_PRESET,
                folder=f"events/{event_id}/pool"
            )
            secure_url = upload_result.get("secure_url")

            # D. Save to Firestore
            # We wrap vectors in a dict to avoid Firestore nested array errors
            vector_data = [{"vector": e} for e in embeddings]

            db.collection('images').add({
                'event_id': event_id,
                'image_url': secure_url,
                'face_vectors': vector_data,
                'createdAt': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"❌ background_worker error: {e}")


# --- 4. ENDPOINTS ---

@app.post("/admin/upload")
async def upload_to_pool(
        background_tasks: BackgroundTasks,
        event_id: str = Form(...),
        images: List[UploadFile] = File(...)
):
    """Admin bulk upload (Asynchronous)."""
    try:
        file_contents = [await file.read() for file in images]
        background_tasks.add_task(process_and_index_pool, event_id, file_contents)
        return {"status": "success", "message": f"Processing {len(images)} images in background."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/reference")
async def register_user_face(user_id: str = Form(...), image: UploadFile = File(...)):
    """Saves user's face identity (Synchronous)."""
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        embeddings = extract_embeddings([img])
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected in selfie.")

        # SIMPLE UNSIGNED UPLOAD
        upload_result = cloudinary.uploader.unsigned_upload(
            contents,
            upload_preset=UPLOAD_PRESET,
            folder=f"users/{user_id}/reference"
        )
        ref_url = upload_result.get("secure_url")

        # Update User in Firestore
        vector_data = [{"vector": e} for e in embeddings]
        user_ref = db.collection('users').document(user_id)
        user_ref.set({
            'reference_vectors': firestore.ArrayUnion(vector_data),
            'reference_images': firestore.ArrayUnion([ref_url]),
            'last_updated': firestore.SERVER_TIMESTAMP
        }, merge=True)

        return {"status": "success", "image_url": ref_url}
    except Exception as e:
        print(f"❌ Reference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/search")
async def search_event_pool(
        event_id: str = Form(...),
        user_id: str = Form(...),
        operation: str = Form("OR")
):
    """Search images where the user appears."""
    try:
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User not found")

        # Unwrap vectors from Firestore format
        user_data = user_doc.to_dict()
        ref_vectors = [item["vector"] for item in user_data.get('reference_vectors', [])]

        pool_query = db.collection('images').where('event_id', '==', event_id).stream()

        matched_urls = []
        for doc in pool_query:
            data = doc.to_dict()
            # Unwrap pool vectors
            db_vectors = [item["vector"] for item in data.get('face_vectors', [])]

            if match_faces(ref_vectors, db_vectors, operation.upper()):
                matched_urls.append(data.get('image_url'))

        return {"matches_found": len(matched_urls), "images": matched_urls}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))