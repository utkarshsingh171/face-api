import os
import uuid
import numpy as np
import cv2
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
import firebase_admin
from firebase_admin import credentials, firestore
import cloudinary
import cloudinary.uploader

# Import your custom logic from model.py
from model import extract_embeddings, match_faces

# --- 1. FIREBASE INITIALIZATION ---
try:
    if not firebase_admin._apps:
        # On Render, ensure you've set GOOGLE_APPLICATION_CREDENTIALS
        # or initialized with a service account file.
        firebase_admin.initialize_app()
    db = firestore.client()
except Exception as e:
    print(f"⚠️ Firebase Warning: {e}")

# --- 2. CLOUDINARY CONFIGURATION ---
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)
print(
    f"--- CLOUDINARY AUDIT: Cloud={os.getenv('CLOUDINARY_CLOUD_NAME')}, Secret_Loaded={bool(os.getenv('CLOUDINARY_API_SECRET'))} ---")

# The name of the unsigned preset you created
UPLOAD_PRESET = "face_flutter_app"

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Anti Gravity Face API is Live 🚀"}


# --- 3. BACKGROUND WORKER FOR POOL INGESTION ---
def process_and_index_pool(event_id: str, file_contents_list: List[bytes]):
    """Processes images, uploads to Cloudinary, and indexes vectors in Firestore."""
    for contents in file_contents_list:
        try:
            # A. Convert bytes to OpenCV image
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if img is None: continue

            # B. Extract Embeddings (Vectors)
            # This uses your global FaceAnalysis model in model.py
            embeddings = extract_embeddings([img])
            if not embeddings:
                continue  # Skip images where no faces are detected

            # Prepare for Firestore: list of objects to avoid nested array errors
            firestore_embeddings = [{"vector": e} for e in embeddings]

            # C. Upload to Cloudinary using Dynamic Event Folders
            print(f"--- CLOUDINARY AUDIT: Uploading image to pool for event {event_id} ---")
            upload_result = cloudinary.uploader.upload(
                contents,
                folder=f"events/{event_id}/pool",
                upload_preset=UPLOAD_PRESET,
                resource_type="image"
            )
            secure_url = upload_result.get("secure_url")

            # D. Save Metadata to Firestore
            print(f"--- FIRESTORE AUDIT: Saving metadata for event {event_id} ---")
            image_id = str(uuid.uuid4())
            db.collection('images').document(image_id).set({
                'event_id': event_id,
                'image_url': secure_url,
                'face_embeddings': firestore_embeddings,  # Array of objects
                'createdAt': firestore.SERVER_TIMESTAMP
            })
        except Exception as e:
            print(f"❌ Error indexing image for event {event_id}: {e}")


# --- 4. ENDPOINTS ---

@app.post("/admin/upload")
async def upload_to_pool(
        background_tasks: BackgroundTasks,
        event_id: str = Form(...),
        images: List[UploadFile] = File(...)
):
    """Admin bulk upload endpoint with date-wise selection support."""
    try:
        # Read file bytes immediately before the request stream closes
        file_contents = [await file.read() for file in images]

        # Move heavy ML and Upload work to background
        background_tasks.add_task(process_and_index_pool, event_id, file_contents)

        return {
            "status": "success",
            "message": f"Queued {len(images)} images for processing.",
            "event_id": event_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.post("/user/reference")
async def register_user_face(user_id: str = Form(...), image: UploadFile = File(...)):
    """Saves user's face identity to their profile for cross-device searching."""
    try:
        contents = await image.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Extract reference vector
        embeddings = extract_embeddings([img])
        if not embeddings:
            raise HTTPException(status_code=400, detail="No face detected in selfie.")

        # Convert to flat list of objects to prevent Firestore Nested Array errors
        firestore_embeddings = [{"vector": e} for e in embeddings]

        # CLOUDINARY STEP
        try:
            print(f"--- CLOUDINARY AUDIT: Uploading reference for user {user_id} ---")
            # We use no trailing slash in 'folder' to prevent signature errors
            upload_result = cloudinary.uploader.upload(
                contents,
                folder=f"users/{user_id}/reference",
                upload_preset=UPLOAD_PRESET
            )
            ref_url = upload_result.get("secure_url")
        except Exception as c_err:
            print(f"❌ Cloudinary Failed: {c_err}")
            raise HTTPException(status_code=500, detail=f"Cloudinary Error: {str(c_err)}")

        # FIRESTORE STEP
        try:
            print(f"--- FIRESTORE AUDIT: Saving reference to Firestore for user {user_id} ---")
            user_ref = db.collection('users').document(user_id)
            user_ref.set({
                # Ensure embeddings is a standard list of objects
                'reference_embeddings': firestore.ArrayUnion(firestore_embeddings),
                'reference_images': firestore.ArrayUnion([ref_url]),
                'last_updated': firestore.SERVER_TIMESTAMP
            }, merge=True)
        except Exception as f_err:
            print(f"❌ Firestore Failed: {f_err}")
            raise HTTPException(status_code=500, detail=f"Firestore Error: {str(f_err)}")

        return {"status": "success", "image_url": ref_url}

    except Exception as e:
        print(f"🔥 System Crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/user/search")
async def search_event_pool(
        event_id: str = Form(...),
        user_id: str = Form(...),
        operation: str = Form("OR")  # "AND" or "OR"
):
    """Instant search using pre-indexed vectors in Firestore."""
    try:
        # 1. Get User's Identity (Reference Vectors)
        print(f"--- FIRESTORE AUDIT: Fetching reference for user {user_id} ---")
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found.")

        ref_embeddings_raw = user_doc.to_dict().get('reference_embeddings', [])
        # Extract the flat vectors from the objects
        ref_embeddings = [item["vector"] for item in ref_embeddings_raw if "vector" in item]

        if not ref_embeddings:
            raise HTTPException(status_code=400, detail="User has no reference faces.")

        # 2. Get all images in the Event Pool
        # Note: For massive events (>2000 pics), consider paginating or using a Vector DB
        print(f"--- FIRESTORE AUDIT: Fetching pool for event {event_id} ---")
        pool_query = db.collection('images').where('event_id', '==', event_id).stream()

        matched_urls = []
        for doc in pool_query:
            data = doc.to_dict()
            db_vectors_raw = data.get('face_embeddings', [])
            # Extract the flat vectors from the objects
            db_vectors = [item["vector"] for item in db_vectors_raw if "vector" in item]
            image_url = data.get('image_url')

            if not db_vectors or not image_url: continue

            # 3. Compare math (using logic in model.py)
            if match_faces(ref_embeddings, db_vectors, operation.upper()):
                matched_urls.append(image_url)

        return {
            "event_id": event_id,
            "operation": operation,
            "matches_found": len(matched_urls),
            "images": matched_urls
        }
    except Exception as e:
        print(f"🔥 Search Crash: {e}")
        raise HTTPException(status_code=500, detail=str(e))