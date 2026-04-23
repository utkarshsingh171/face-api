import numpy as np
from insightface.app import FaceAnalysis

SIMILARITY_THRESHOLD = 0.40

# -----------------------------
# LOAD MODEL (ONLY ONCE)
# -----------------------------
app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))


# -----------------------------
# MAIN FUNCTION
# -----------------------------
def find_faces(reference_imgs, group_images, operation="AND"):

    ref_embeddings = []

    # -----------------------------
    # GET REFERENCE EMBEDDINGS
    # -----------------------------
    for ref_img in reference_imgs:

        faces = app.get(ref_img)

        if len(faces) == 0:
            return "No face found in one of the reference images"

        emb = faces[0].embedding
        emb = emb / np.linalg.norm(emb)

        ref_embeddings.append(emb)

    matched_indices = []

    # -----------------------------
    # PROCESS GROUP IMAGES
    # -----------------------------
    for i, img in enumerate(group_images):

        faces = app.get(img)

        if len(faces) == 0:
            continue

        face_embeddings = []

        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            face_embeddings.append(emb)

        match_found = False

        # -----------------------------
        # AND LOGIC
        # -----------------------------
        if operation == "AND":

            match_found = all(
                any(np.dot(ref_emb, face_emb) > SIMILARITY_THRESHOLD for face_emb in face_embeddings)
                for ref_emb in ref_embeddings
            )

        # -----------------------------
        # OR LOGIC
        # -----------------------------
        elif operation == "OR":

            match_found = any(
                np.dot(ref_emb, face_emb) > SIMILARITY_THRESHOLD
                for ref_emb in ref_embeddings
                for face_emb in face_embeddings
            )

        # -----------------------------
        # STORE RESULT
        # -----------------------------
        if match_found:
            matched_indices.append(i)

    return matched_indices