import numpy as np
from insightface.app import FaceAnalysis

SIMILARITY_THRESHOLD = 0.45

# -----------------------------
# LOAD MODEL (ONLY ONCE)
# -----------------------------
app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(320, 320))


# -----------------------------
# EXTRACT EMBEDDINGS
# -----------------------------
def extract_embeddings(images):
    """
    Extracts face embeddings from a list of images.
    Returns a list of lists of floats.
    """
    all_embeddings = []

    for img in images:
        faces = app.get(img)
        if len(faces) == 0:
            continue

        for face in faces:
            emb = face.embedding
            emb = emb / np.linalg.norm(emb)
            all_embeddings.append(emb.tolist())

    return all_embeddings


# -----------------------------
# MATCH FACES
# -----------------------------
def match_faces(ref_embeddings, db_vectors, operation="AND"):
    """
    Matches reference embeddings against a list of pre-fetched database vectors.
    ref_embeddings: List of numpy arrays (or lists of floats) representing the reference faces.
    db_vectors: List of lists of floats representing the faces in a database image.
    operation: 'AND' or 'OR'.
    Returns boolean indicating if there is a match.
    """
    if not ref_embeddings or not db_vectors:
        return False

    # Convert inputs to numpy arrays if they aren't already
    ref_embeddings_np = [np.array(e) if isinstance(e, list) else e for e in ref_embeddings]
    db_vectors_np = [np.array(e) if isinstance(e, list) else e for e in db_vectors]

    match_found = False

    if operation.upper() == "AND":
        match_found = all(
            any(np.dot(ref_emb, db_vec) > SIMILARITY_THRESHOLD for db_vec in db_vectors_np)
            for ref_emb in ref_embeddings_np
        )
    else:  # OR
        match_found = any(
            np.dot(ref_emb, db_vec) > SIMILARITY_THRESHOLD
            for ref_emb in ref_embeddings_np
            for db_vec in db_vectors_np
        )

    return match_found