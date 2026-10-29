# Smart Attendance System — Portable Edition

## Run on another Windows PC
1. Install Python 3.10+ (64-bit) and PostgreSQL.
2. Copy this entire folder to the new PC.
3. Double-click `run.bat`. The first run creates `.venv` and installs the Python dependencies.
4. Edit `.env` with that PC's PostgreSQL settings.
5. Run `run.bat` again.

## Included
- Professional frontend with Attendance and Registration buttons
- Face detection (SCRFD)
- ArcFace embeddings + FAISS index
- PostgreSQL attendance database integration
- Supertonic voice notifications
- 60-minute attendance window
- Persistent WhatsApp queue using PyWhatKit

## Important
- The project uses a local PostgreSQL database; PostgreSQL itself is not bundled.
- The `.env` file containing database credentials is intentionally excluded from portable distribution. Use `.env.example` as the template.
- A camera must be available as camera index 0, unless the camera module is changed.
- WhatsApp Web automation requires WhatsApp Web login on the target PC.
- The `data/face_index.faiss` file is machine-independent, but the student records it references must exist in the target PostgreSQL database.
