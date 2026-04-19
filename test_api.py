from fastapi import FastAPI, UploadFile, File, Form
from typing import List
import numpy as np
import cv2

from model import find_faces

app = FastAPI()


@app.get("/")
def home():
    return {"message": "Face API running 🚀"}


@app.post("/match")
async def match_faces(
        mode: str = Form(...),  # single / multiple
        operation: str = Form("OR"),

        # ✅ UNLIMITED IMAGES
        references: List[UploadFile] = File(...),
        groups: List[UploadFile] = File(...)
):
    ref_images = []
    group_images = []

    # -----------------------------
    # LOAD REFERENCE IMAGES
    # -----------------------------
    for file in references:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is not None:
            ref_images.append(img)

    # -----------------------------
    # LOAD GROUP IMAGES
    # -----------------------------
    for file in groups:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is not None:
            group_images.append(img)

    # -----------------------------
    # VALIDATION
    # -----------------------------
    if len(ref_images) == 0:
        return {"error": "No reference images"}

    if len(group_images) == 0:
        return {"error": "No group images"}

    # -----------------------------
    # MODE LOGIC
    # -----------------------------
    if mode.lower() == "single":
        operation = "OR"
    else:
        operation = operation.upper()
        if operation not in ["AND", "OR"]:
            return {"error": "Invalid operation"}

    # -----------------------------
    # RUN MODEL
    # -----------------------------
    results = find_faces(
        reference_imgs=ref_images,
        group_images=group_images,
        operation=operation
    )

    if isinstance(results, str):
        return {"error": results}

    return {
        "mode": mode,
        "operation": operation,
        "total_group_images": len(group_images),
        "matches_found": len(results),
        "matched_indices": results
    }
# checking the update