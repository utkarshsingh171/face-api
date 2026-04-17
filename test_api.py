from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2

from model import find_faces

app = FastAPI()


# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Face API running 🚀"}


# -----------------------------
# MAIN API
# -----------------------------
@app.post("/match")
async def match_faces(
    mode: str = Form(...), # single / multiple

    # reference images
    ref1: UploadFile = File(...),
    ref2: UploadFile = File(None),
    ref3: UploadFile = File(None),

    # group images
    grp1: UploadFile = File(...),
    grp2: UploadFile = File(None),
    grp3: UploadFile = File(None),
    grp4: UploadFile = File(None),

    operation: str = Form(...),  # only used in multiple
):

    ref_images = []
    group_images = []

    # -----------------------------
    # LOAD REFERENCE IMAGES
    # -----------------------------
    ref_files = [ref1, ref2, ref3]

    for file in ref_files:
        if file is not None:
            contents = await file.read()
            np_arr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if img is not None:
                ref_images.append(img)

    # -----------------------------
    # LOAD GROUP IMAGES
    # -----------------------------
    grp_files = [grp1, grp2, grp3, grp4]

    for file in grp_files:
        if file is not None:
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
    # MODE HANDLING
    # -----------------------------
    if mode.lower() == "single":
        operation = "OR"  # single person → OR logic automatically

    else:
        operation = operation.upper()
        if operation not in ["AND", "OR"]:
            return {"error": "Invalid operation (use AND / OR)"}

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

    # -----------------------------
    # RESPONSE
    # -----------------------------
    return {
        "mode": mode,
        "operation": operation,
        "total_group_images": len(group_images),
        "matches_found": len(results),
        "matched_indices": results
    }