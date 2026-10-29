import os
import sys
import ctypes
import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import psycopg
import pywhatkit

from dotenv import load_dotenv

from src.app.camera.camera import Camera
from src.app.embedding.scrfd import SCRFD
from src.app.alignment.alignment import FaceAligner
from src.app.embedding.arc_face import ArcFace
from src.vector_store import VectorStore
from src.app.tts import TTSEngine
from src.app.tts.service import TTSService

from template.registration_template import (
    student_registered_template,
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attendance_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD")


# =========================================================
# MODEL PATHS
# =========================================================

SCRFD_MODEL = Path(
    "models/scrfd/2.5g_bnkps.onnx"
)

ARCFACE_MODEL = Path(
    "models/arcface/w600k_r50.onnx"
)


# =========================================================
# SETTINGS
# =========================================================

FACE_MATCH_THRESHOLD = 0.60

WINDOW_NAME = "Student Registration"

VK_SPACE = 0x20
VK_Q = 0x51

WHATSAPP_WAIT_TIME = 15
WHATSAPP_TAB_CLOSE = True
WHATSAPP_CLOSE_TIME = 3

# Keep the registration camera live for at least five seconds after
# the success voice notification finishes.
REGISTRATION_COOLDOWN_SECONDS = 5.0


# =========================================================
# WINDOWS KEYBOARD
# =========================================================

def is_key_down(
    virtual_key: int,
) -> bool:
    """
    Check whether a Windows virtual key is currently pressed.

    User32 is used to detect SPACE and Q while the OpenCV
    registration window is active.
    """

    return bool(
        ctypes.windll.user32.GetAsyncKeyState(
            virtual_key
        ) & 0x8000
    )


# =========================================================
# DATABASE
# =========================================================

def get_db_connection():
    """
    Create a PostgreSQL database connection.

    Credentials are read from the environment so database
    passwords are not stored directly in the Python file.
    """

    if not DB_PASSWORD:
        raise RuntimeError(
            "DB_PASSWORD is not configured."
        )

    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def create_student(
    name: str,
    phone_number: str,
) -> int:
    """
    Insert a new student into PostgreSQL.

    PostgreSQL generates the student ID and returns it so that
    the same ID can be attached to the student's FAISS embedding.
    """

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO students (
                    name,
                    phone_number
                )
                VALUES (
                    %s,
                    %s
                )
                RETURNING student_id
                """,
                (
                    name.strip(),
                    phone_number.strip(),
                ),
            )

            row = cursor.fetchone()

        connection.commit()

        if row is None:
            raise RuntimeError(
                "Student ID was not returned."
            )

        return int(row[0])

    except Exception:

        connection.rollback()
        raise

    finally:

        connection.close()


def delete_student(
    student_id: int,
):
    """
    Delete a newly created PostgreSQL student record.

    This rollback helper prevents PostgreSQL and FAISS from becoming
    inconsistent when FAISS insertion fails after database insertion.
    """

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                DELETE FROM students
                WHERE student_id = %s
                """,
                (student_id,),
            )

        connection.commit()

    finally:

        connection.close()


def find_existing_student(name: str, phone_number: str):
    """
    Reuse a PostgreSQL student only when both name and normalized phone
    match. This repairs a missing FAISS embedding without creating a new ID.
    """
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT student_id, name, phone_number
                FROM students
                WHERE lower(trim(name)) = lower(trim(%s))
                  AND phone_number = %s
                ORDER BY student_id
                """,
                (name.strip(), phone_number.strip()),
            )
            rows = cursor.fetchall()
        if len(rows) > 1:
            raise RuntimeError(
                "Multiple PostgreSQL rows match this name and phone. "
                "Resolve duplicate student records before registering."
            )
        return rows[0] if rows else None
    finally:
        connection.close()


# =========================================================
# PHONE
# =========================================================

def normalize_phone_number(
    phone_number: str,
) -> str:
    """
    Normalize a Pakistani phone number.

    Local 03XXXXXXXXX numbers are converted to +92XXXXXXXXXX
    format before storage and WhatsApp delivery.
    """

    phone = (
        phone_number
        .strip()
        .replace(" ", "")
        .replace("-", "")
        .replace("(", "")
        .replace(")", "")
    )

    if not phone:

        raise ValueError(
            "Phone number cannot be empty."
        )

    if phone.startswith("00"):

        phone = "+" + phone[2:]

    elif phone.startswith("03"):

        phone = "+92" + phone[1:]

    elif phone.startswith("92"):

        phone = "+" + phone

    elif not phone.startswith("+"):

        raise ValueError(
            "Invalid phone number format."
        )

    return phone


# =========================================================
# INPUT
# =========================================================

def collect_student_name() -> str:
    """
    Ask for a student name.

    The function continues until the operator enters
    a non-empty name.
    """

    while True:

        name = input(
            "Student name: "
        ).strip()

        if name:
            return name

        print(
            "Name cannot be empty."
        )


def collect_phone_number() -> str:
    """
    Ask for and validate the student's phone number.

    The returned value is already normalized to international format.
    """

    while True:

        phone_number = input(
            "Student phone number: "
        ).strip()

        try:

            return normalize_phone_number(
                phone_number
            )

        except ValueError as exc:

            print(
                "Invalid phone number:",
                exc,
            )


# =========================================================
# WHATSAPP
# =========================================================

def send_registration_whatsapp(
    phone_number: str,
    student_name: str,
    student_id: int,
) -> bool:
    """
    Send the registration confirmation through WhatsApp Web.

    PyWhatKit opens the WhatsApp Web conversation and automatically
    sends the generated registration template.
    """

    message = (
        student_registered_template(
            student_name=student_name,
            student_id=student_id,
        )
    )

    try:

        pywhatkit.sendwhatmsg_instantly(
            phone_no=phone_number,
            message=message,
            wait_time=WHATSAPP_WAIT_TIME,
            tab_close=WHATSAPP_TAB_CLOSE,
            close_time=WHATSAPP_CLOSE_TIME,
        )

        print(
            "Registration WhatsApp message sent."
        )

        return True

    except Exception as exc:

        print(
            "WhatsApp registration error:",
            exc,
        )

        return False


# =========================================================
# DRAW
# =========================================================

def draw_detection(
    frame: np.ndarray,
    detection: dict,
):
    """
    Draw a face bounding box and five landmarks.

    This provides visual feedback during student registration.
    """

    x1, y1, x2, y2 = (
        detection["box"]
        .astype(int)
    )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    for landmark in detection["landmarks"]:

        x, y = landmark.astype(int)

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 0, 255),
            -1,
        )


# =========================================================
# CAMERA
# =========================================================

def close_camera(
    camera,
):
    """
    Release the camera and destroy OpenCV windows.

    Cleanup is performed when registration capture ends.
    """

    try:
        camera.release()

    except Exception as exc:

        print(
            "Camera release warning:",
            exc,
        )

    cv2.destroyAllWindows()


def capture_registration_frame(
    camera,
    detector: SCRFD,
):
    """
    Capture one valid registration frame.

    SPACE accepts a frame only when exactly one face is detected.
    Q cancels the registration workflow.
    """

    previous_space = False
    previous_q = False

    selected_frame = None

    # Allow the registration preview image to scale with the window.
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:

        while True:

            frame = camera.read()

            if frame is None:

                raise RuntimeError(
                    "Could not read camera frame."
                )

            detections = detector.detect(
                frame
            )

            display = frame.copy()

            for detection in detections:

                draw_detection(
                    display,
                    detection,
                )

            if len(detections) == 0:

                status = (
                    "No face detected"
                )

            elif len(detections) == 1:

                status = (
                    "One face - Ready"
                )

            else:

                status = (
                    f"{len(detections)} faces - "
                    "ONLY ONE ALLOWED"
                )

            cv2.putText(
                display,
                status,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                display,
                "SPACE = Register | Q = Quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            # TOPMOST intentionally disabled: do not let OpenCV compete with Camera Emulator / Imou.

            space_state = is_key_down(
                VK_SPACE
            )

            space_pressed = (
                space_state
                and not previous_space
            )

            previous_space = (
                space_state
            )

            q_state = is_key_down(
                VK_Q
            )

            q_pressed = (
                q_state
                and not previous_q
            )

            previous_q = q_state

            if space_pressed:

                if len(detections) == 1:

                    selected_frame = (
                        frame.copy()
                    )

                    break

                print(
                    "Registration requires exactly one face."
                )

            if q_pressed:

                break

            cv2.waitKey(1)

    finally:
        # Do NOT release the camera here. Registration must keep the
        # camera alive until the operator presses Q.
        pass

    return selected_frame


# =========================================================
# EMBEDDING
# =========================================================

def keep_registration_camera_open(
    camera,
    detector: SCRFD,
    hold_seconds: float = REGISTRATION_COOLDOWN_SECONDS,
):
    """
    Keep the registration camera live after a capture/registration.
    The hold timer starts when this function is entered, so the
    configured minimum is measured after the success notification
    has completed. The camera is released only when Q is pressed.
    """
    previous_q = False
    cooldown_until = time.monotonic() + max(0.0, hold_seconds)

    while True:
        frame = camera.read()

        if frame is None:
            print("Could not read camera frame.")
            break

        display = frame.copy()
        detections = detector.detect(frame)

        for detection in detections:
            draw_detection(display, detection)

        if time.monotonic() < cooldown_until:
            remaining = max(
                0.0,
                cooldown_until - time.monotonic(),
            )
            status = f"Registration cooldown: {remaining:.1f}s"
        else:
            status = "Registration camera ready"

        cv2.putText(
            display,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            display,
            "Q = Quit",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.imshow(WINDOW_NAME, display)

        # TOPMOST intentionally disabled: do not let OpenCV compete with Camera Emulator / Imou.

        q_state = is_key_down(VK_Q)
        q_pressed = q_state and not previous_q
        previous_q = q_state

        if q_pressed:
            break

        cv2.waitKey(1)


def generate_embedding(
    frame: np.ndarray,
    detector: SCRFD,
    aligner: FaceAligner,
    arcface: ArcFace,
):
    """
    Detect, align and encode one face.

    The resulting normalized 512-dimensional vector is prepared
    for FAISS similarity search and storage.
    """

    detections = detector.detect(
        frame
    )

    if len(detections) != 1:

        raise RuntimeError(
            "Exactly one face is required."
        )

    detection = detections[0]

    aligned_face = aligner.align(
        frame,
        detection["landmarks"],
    )

    embedding = arcface.get_embedding(
        aligned_face
    )

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(512)

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        raise RuntimeError(
            "ArcFace returned a zero embedding."
        )

    return (
        embedding / norm
    ).astype(
        np.float32
    )


# =========================================================
# MAIN REGISTRATION WORKFLOW
# =========================================================

def register_student(
    provided_name: str | None = None,
    provided_phone: str | None = None,
):
    """
    Run the complete student enrollment workflow.

    Optional name and phone arguments allow the desktop frontend
    to collect student details before opening the camera.

    A face is checked against FAISS first. Only an unknown face can
    be registered. The function then creates PostgreSQL and FAISS
    records and sends the registration confirmation through WhatsApp.
    """

    # Always resolve relative project resources from this script's folder,
    # regardless of the directory from which register_student.py is launched.
    project_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
    os.chdir(project_root)

    print()
    print("=" * 60)
    print("STUDENT REGISTRATION")
    print("=" * 60)

    # Initialize TTS before face capture so the student is prompted
    # to look at the camera and duplicate registrations can be announced.
    try:

        tts = TTSService(
            engine=TTSEngine(),
            output_path=Path("data/tts_output.wav"),
        )

        tts.speak(
            "Please look at the camera.",
            language="en",
        )

        print("Registration TTS initialized.")

    except Exception as exc:

        tts = None

        print("Registration TTS initialization warning:", exc)

    camera = Camera()

    detector = SCRFD(
        str(SCRFD_MODEL),
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )

    aligner = FaceAligner()

    arcface = ArcFace(
        str(ARCFACE_MODEL)
    )

    vector_store = VectorStore()

    frame = capture_registration_frame(
        camera,
        detector,
    )

    if frame is None:
        print("Registration cancelled.")
        close_camera(camera)
        return False

    print(
        "Generating face embedding..."
    )

    embedding = generate_embedding(
        frame,
        detector,
        aligner,
        arcface,
    )

    print(
        "Checking whether this face is already registered..."
    )

    existing = (
        vector_store.find_existing(
            embedding,
            threshold=FACE_MATCH_THRESHOLD,
        )
    )

    if existing is not None:

        print()
        print(
            "REGISTRATION REJECTED"
        )

        print(
            "Face is already registered."
        )

        print(
            "Existing Student ID:",
            int(existing["vector_id"]),
        )

        print(
            "Similarity:",
            float(existing["similarity"]),
        )

        if tts is not None:

            tts.speak(
                "This student is already registered.",
                language="en",
            )

        return False

    print()

    name = (
        provided_name.strip()
        if provided_name
        else collect_student_name()
    )

    phone_number = (
        provided_phone.strip()
        if provided_phone
        else collect_phone_number()
    )

    phone_number = normalize_phone_number(
        phone_number
    )

    student_id = None
    created_student = False
    mapping_committed = False

    try:
        # Reuse an exact existing PostgreSQL identity when its face vector
        # is missing; never mint a second ID for the same student.
        existing_student = find_existing_student(name, phone_number)
        if existing_student is not None:
            student_id = int(existing_student[0])
            print(
                f"Repairing missing face mapping for existing "
                f"student ID {student_id}."
            )
        else:
            student_id = create_student(
                name=name,
                phone_number=phone_number,
            )
            created_student = True

        vector_store.add_vector(
            student_id,
            embedding,
        )
        # From this point onward, the DB row and FAISS ID are linked.
        # Notification/window errors must not delete either record.
        mapping_committed = True

        print()
        print("=" * 60)
        print("REGISTRATION SUCCESSFUL")
        print("=" * 60)

        print(
            "Student ID:",
            student_id,
        )

        print(
            "Name:",
            name,
        )

        print(
            "Phone:",
            phone_number,
        )

        print(
            "No attendance was recorded."
        )

        # ---------------------------------------------
        # TTS VOICE NOTIFICATION
        # ---------------------------------------------

        if tts is not None:

            try:

                tts.speak(
                    text=f"{name} is now registered.",
                    language="en",
                )

                print(
                    "Registration success TTS message spoken."
                )

            except Exception as exc:

                print(
                    "Registration TTS warning:",
                    exc,
                )

        # ---------------------------------------------
        # WHATSAPP CONFIRMATION
        # ---------------------------------------------

        send_registration_whatsapp(
            phone_number=phone_number,
            student_name=name,
            student_id=student_id,
        )

        # Keep the camera open after registration. The operator closes
        # the registration session explicitly with Q.
        keep_registration_camera_open(
            camera,
            detector,
        )

        close_camera(camera)
        return True

    except Exception as exc:

        print(
            "Registration failed:",
            exc,
        )

        # Roll back only an incomplete mapping. Once both the PostgreSQL
        # row and FAISS vector are saved, later TTS/WhatsApp/window failures
        # must not delete the student's identity or leave an orphan vector.
        if not mapping_committed and student_id is not None:
            # add_vector performs its own rollback if persistence fails.
            # Never remove by ID here: a failed duplicate-ID rejection must
            # not delete a pre-existing student's valid FAISS mapping.
            if created_student:
                try:
                    delete_student(student_id)
                except Exception as cleanup_exc:
                    print(
                        "Database rollback failed:",
                        cleanup_exc,
                    )

        close_camera(camera)
        return False


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student registration workflow"
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--phone", default=None)
    args = parser.parse_args()

    register_student(
        provided_name=args.name,
        provided_phone=args.phone,
    )