import sys
import os
import ctypes
import json
from datetime import datetime
import time
import threading
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
from src.app.tts.notifier import TTSNotifier

from template.attendance_template import (
    attendance_success_template,
)

load_dotenv()

DB_HOST = os.getenv(
    "DB_HOST",
    "localhost",
)

DB_PORT = os.getenv(
    "DB_PORT",
    "5432",
)

DB_NAME = os.getenv(
    "DB_NAME",
    "attendance_db",
)

DB_USER = os.getenv(
    "DB_USER",
    "postgres",
)

DB_PASSWORD = os.getenv(
    "DB_PASSWORD"
)


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

WINDOW_NAME = (
    "Face Recognition Attendance"
)

VK_Q = 0x51


# =========================================================
# WHATSAPP SETTINGS
# =========================================================

WHATSAPP_WAIT_TIME = 15

# Delay between queued WhatsApp notifications. Attendance collection is
# completed first; only then are queued messages sent one by one.
WHATSAPP_QUEUE_DELAY_SECONDS = 10

WHATSAPP_TAB_CLOSE = True

WHATSAPP_CLOSE_TIME = 3

# Minimum time between spoken warnings for unknown faces.
UNKNOWN_TTS_COOLDOWN_SECONDS = 5.0

# Briefly suppress an unknown-face warning after a registered face
# disappears. A suddenly hidden face can produce a few partial/poor
# detections before SCRFD reports no face at all. Those transient
# detections must not be announced as an unregistered person.
FACE_LOSS_GRACE_SECONDS = 1.5

# Run recognition more frequently while keeping the display thread independent.
RECOGNITION_INTERVAL_SECONDS = 0.30

# Keep the camera live and resume recognition quickly after a successful mark.
NEW_STUDENT_ATTENDANCE_COOLDOWN_SECONDS = 0.75

# Give the already-marked instruction sooner, without changing duplicate protection.
ALREADY_ATTENDED_CONFIRMATION_SECONDS = 1.5

# Attendance is accepted only during the first 60 minutes after startup.
# Recognition continues after that window so late registered students can
# receive the spoken late-arrival message.
ATTENDANCE_WINDOW_SECONDS = 60 * 60


# =========================================================
# WINDOWS KEYBOARD
# =========================================================

def is_key_down(
    virtual_key: int,
) -> bool:
    """
    Check whether a Windows keyboard key is currently pressed.

    User32 is used for non-blocking Q-key detection while the
    OpenCV attendance window is active.
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
    Create and return a PostgreSQL database connection.

    Connection credentials are loaded from the .env file so that
    database secrets are not hard-coded in the application.
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


def get_student(
    student_id: int,
):
    """
    Retrieve a registered student from PostgreSQL.

    The returned record contains the database student ID, name,
    and registered phone number used by attendance processing.
    """

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                SELECT
                    student_id,
                    name,
                    phone_number
                FROM students
                WHERE student_id = %s
                """,
                (student_id,),
            )

            return cursor.fetchone()

    finally:

        connection.close()


# =========================================================
# ATTENDANCE
# =========================================================

def has_attendance_for_date(
    student_id: int,
    attendance_date,
) -> bool:
    """Return whether this student already has an attendance row for the date."""
    connection = get_db_connection()
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT 1
                FROM attendance
                WHERE student_id = %s AND date = %s
                LIMIT 1
                """,
                (student_id, attendance_date),
            )
            return cursor.fetchone() is not None
    finally:
        connection.close()


def record_attendance(
    student_id: int,
    attendance_datetime: datetime,
) -> bool:
    """
    Insert one attendance record for a registered student.

    Only an already verified student can reach this function.
    Unknown faces never call this database operation.
    """

    connection = get_db_connection()

    try:

        with connection.cursor() as cursor:

            cursor.execute(
                """
                INSERT INTO attendance (
                    student_id,
                    status,
                    date,
                    time
                )
                VALUES (
                    %s,
                    %s,
                    %s,
                    %s
                )
                """,
                (
                    student_id,
                    "Present",
                    attendance_datetime.date(),
                    attendance_datetime.strftime(
                        "%H:%M:%S"
                    ),
                ),
            )

        connection.commit()

        print(
            f"Attendance recorded for "
            f"student {student_id}."
        )

        return True

    except Exception as exc:

        connection.rollback()

        print(
            "Attendance database error:",
            exc,
        )

        return False

    finally:

        connection.close()


# =========================================================
# WHATSAPP
# =========================================================

def normalize_phone_number(
    phone_number: str,
) -> str:
    """
    Normalize a Pakistani phone number into international format.

    The normalized value is suitable for passing to PyWhatKit
    when opening a WhatsApp Web conversation.
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


def send_attendance_whatsapp(
    phone_number: str,
    student_name: str,
    student_id: int,
    attendance_datetime: datetime,
) -> bool:
    """
    Send the registered student's attendance confirmation.

    PyWhatKit opens WhatsApp Web and sends the message generated
    by the dedicated attendance template module.
    """

    phone = normalize_phone_number(
        phone_number
    )

    message = (
        attendance_success_template(
            student_name=student_name,
            student_id=student_id,
            attendance_datetime=attendance_datetime,
        )
    )

    try:

        pywhatkit.sendwhatmsg_instantly(
            phone_no=phone,
            message=message,
            wait_time=WHATSAPP_WAIT_TIME,
            tab_close=WHATSAPP_TAB_CLOSE,
            close_time=WHATSAPP_CLOSE_TIME,
        )

        print(
            f"WhatsApp sent to {student_name}."
        )

        return True

    except Exception as exc:

        print(
            f"WhatsApp error for {student_name}:",
            exc,
        )

        return False

def save_whatsapp_queue(
    queue: list[dict],
    queue_path: Path,
) -> None:
    """
    Persist the current session's WhatsApp queue.

    The queue is written after every successful attendance so the
    pending notifications are not kept only in RAM.
    """

    queue_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with queue_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            queue,
            file,
            indent=2,
            ensure_ascii=False,
        )


def send_whatsapp_queue(
    queue: list[dict],
    queue_path: Path,
) -> None:
    """
    Send queued attendance notifications one by one after the
    attendance window has ended.
    """

    if not queue:
        print(
            "WhatsApp queue is empty. No messages to send."
        )
        return

    print(
        f"Attendance window ended. Sending {len(queue)} "
        "WhatsApp notification(s)..."
    )

    for index, item in enumerate(queue, start=1):

        print(
            f"Sending WhatsApp {index}/{len(queue)} "
            f"to {item['name']}..."
        )

        try:

            sent = send_attendance_whatsapp(
                phone_number=item["phone_number"],
                student_name=item["name"],
                student_id=int(item["student_id"]),
                attendance_datetime=datetime.fromisoformat(
                    item["attendance_datetime"]
                ),
            )

        except Exception as exc:

            sent = False

            print(
                f"WhatsApp queue error for {item['name']}:",
                exc,
            )

        item["status"] = (
            "sent"
            if sent
            else "failed"
        )

        save_whatsapp_queue(
            queue,
            queue_path,
        )

        # Keep a fixed 10-second gap between queued recipients. Do not
        # delay after the final message because there is no next recipient.
        if index < len(queue):
            time.sleep(WHATSAPP_QUEUE_DELAY_SECONDS)


# =========================================================
# DRAW FACE
# =========================================================

def draw_face(
    frame: np.ndarray,
    detection: dict,
    display_name: str | None = None,
):
    """
    Draw a detected face and optionally display its student name.

    Registered students are rendered as <name> above their bounding
    box, while unknown faces can be visually ignored.
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
            3,
            (0, 0, 255),
            -1,
        )

    if display_name:

        label = (
            f"{display_name}'s attendance marked!"
        )

        text_y = max(
            30,
            y1 - 10,
        )

        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )


# =========================================================
# CAMERA CLEANUP
# =========================================================

def close_camera(
    camera,
):
    """
    Release the camera and destroy all OpenCV windows.

    Cleanup is used when attendance processing ends or the
    operator presses Q.
    """

    try:

        camera.release()

    except Exception as exc:

        print(
            "Camera release warning:",
            exc,
        )

    cv2.destroyAllWindows()


# =========================================================
# FACE EMBEDDING
# =========================================================

def generate_embedding(
    frame: np.ndarray,
    detection: dict,
    aligner: FaceAligner,
    arcface: ArcFace,
):
    """
    Generate one normalized ArcFace embedding for one detected face.

    Detection is already performed by SCRFD, so this function only
    aligns that specific face and converts it into a 512-D vector.
    """

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

    embedding = (
        embedding / norm
    ).astype(
        np.float32
    )

    return embedding


# =========================================================
# PROCESS ONE FACE
# =========================================================

def process_one_face(
    frame: np.ndarray,
    detection: dict,
    aligner: FaceAligner,
    arcface: ArcFace,
    vector_store: VectorStore,
    attended_students: set[int],
    student_cache: dict[int, tuple],
    tts: TTSNotifier | None,
    attendance_window_open: bool = True,
):
    """
    Process one detected face independently.

    The face is converted to an embedding and searched against
    registered FAISS vectors. Unknown faces are ignored; a valid
    registered match is verified in PostgreSQL before attendance.
    """

    try:

        embedding = generate_embedding(
            frame=frame,
            detection=detection,
            aligner=aligner,
            arcface=arcface,
        )

        # -------------------------------------------------
        # FAISS SEARCH
        # -------------------------------------------------

        match = (
            vector_store.find_existing(
                embedding,
                threshold=FACE_MATCH_THRESHOLD,
            )
        )

        # -------------------------------------------------
        # UNKNOWN FACE
        # -------------------------------------------------

        if match is None:

            # Completely ignore strangers.
            return {
                "status": "unknown",
                "name": None,
                "student_id": None,
            }

        # -------------------------------------------------
        # REGISTERED MATCH
        # -------------------------------------------------

        student_id = int(
            match["vector_id"]
        )

        similarity = float(
            match["similarity"]
        )

        # -------------------------------------------------
        # LATE ARRIVAL AFTER THE ATTENDANCE WINDOW
        # -------------------------------------------------

        if not attendance_window_open:

            student = student_cache.get(
                student_id
            )

            if student is None:

                student = get_student(
                    student_id
                )

                if student is not None:

                    student_cache[student_id] = student

            if student is not None:

                return {
                    "status": "late",
                    "name": (
                        student[1].strip()
                        if student[1]
                        else None
                    ),
                    "student_id": student_id,
                    "similarity": similarity,
                }

            return {
                "status": "database_error",
                "name": None,
                "student_id": student_id,
            }

        # -------------------------------------------------
        # PREVENT DUPLICATE ATTENDANCE
        # -------------------------------------------------

        if student_id in attended_students:

            student = student_cache.get(
                student_id
            )

            if student is None:

                student = get_student(
                    student_id
                )

                if student is not None:

                    student_cache[student_id] = student

            if student is not None:

                return {
                    "status": "already_attended",
                    "name": (
                        student[1].strip()
                        if student[1]
                        else None
                    ),
                    "student_id": student_id,
                    "similarity": similarity,
                }

            return {
                "status": "database_error",
                "name": None,
                "student_id": student_id,
            }

        # -------------------------------------------------
        # POSTGRESQL VERIFICATION
        # -------------------------------------------------

        student = student_cache.get(
            student_id
        )

        if student is None:

            student = get_student(
                student_id
            )

            if student is not None:

                student_cache[student_id] = student

        if student is None:

            print(
                f"FAISS matched student {student_id}, "
                "but PostgreSQL record was not found."
            )

            return {
                "status": "database_error",
                "name": None,
                "student_id": student_id,
            }

        database_student_id = int(
            student[0]
        )

        student_name = (
            student[1].strip()
            if student[1]
            else ""
        )

        phone_number = (
            student[2].strip()
            if student[2]
            else ""
        )

        # -------------------------------------------------
        # REQUIRED STUDENT DATA
        # -------------------------------------------------

        if not student_name:

            print(
                f"Student {database_student_id} "
                "has no name."
            )

            return {
                "status": "invalid_student",
                "name": None,
                "student_id": database_student_id,
                "similarity": similarity,
            }

        if not phone_number:

            print(
                f"{student_name} has no phone number."
            )

            return {
                "status": "invalid_student",
                "name": student_name,
                "student_id": database_student_id,
                "similarity": similarity,
            }

        # -------------------------------------------------
        # PREVENT DUPLICATE ATTENDANCE ACROSS APP RESTARTS
        # -------------------------------------------------

        current_datetime = datetime.now()
        if has_attendance_for_date(
            database_student_id,
            current_datetime.date(),
        ):
            print(
                f"Attendance already exists today for student "
                f"{database_student_id}; duplicate skipped."
            )
            return {
                "status": "already_attended",
                "name": student_name,
                "student_id": database_student_id,
                "similarity": similarity,
            }

        # -------------------------------------------------
        # ATTENDANCE TIME
        # -------------------------------------------------

        attendance_datetime = current_datetime

        # -------------------------------------------------
        # RECORD ATTENDANCE
        # -------------------------------------------------

        attendance_success = (
            record_attendance(
                student_id=database_student_id,
                attendance_datetime=attendance_datetime,
            )
        )

        if not attendance_success:

            return {
                "status": "attendance_failed",
                "name": student_name,
                "student_id": database_student_id,
                "similarity": similarity,
            }

        # -------------------------------------------------
        # MARK AS ATTENDED
        # -------------------------------------------------

        attended_students.add(
            database_student_id
        )

        # -------------------------------------------------
        # TTS VOICE NOTIFICATION
        # -------------------------------------------------

        if tts is not None:

            tts.speak(
                f"Hello, {student_name}. "
                "Your attendance has been marked."
            )

        # -------------------------------------------------
        # WHATSAPP QUEUE DATA
        # -------------------------------------------------

        # WhatsApp is intentionally NOT sent here. The notification
        # is queued by main() and sent only after the 60-minute
        # attendance window has ended.

        return {
            "status": "attended",
            "name": student_name,
            "student_id": database_student_id,
            "phone_number": phone_number,
            "attendance_datetime": attendance_datetime.isoformat(),
            "similarity": similarity,
        }

    except Exception as exc:

        print(
            "Face processing error:",
            exc,
        )

        return {
            "status": "error",
            "name": None,
            "student_id": None,
        }


# =========================================================
# MAIN
# =========================================================

def main():
    """
    Run multi-student attendance recognition.

    The camera/display loop is intentionally independent from the expensive
    recognition pipeline. SCRFD, ArcFace, FAISS, PostgreSQL, and TTS run in
    a background worker so the visible camera keeps refreshing continuously.
    """

    project_root = (
        Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
        if getattr(sys, "frozen", False)
        else Path(__file__).resolve().parent
    )
    os.chdir(project_root)

    print("Starting attendance system...")

    camera = Camera()

    initial_frame = camera.read()
    if initial_frame is None:
        raise RuntimeError("Could not initialize the camera.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 960, 540)
    cv2.moveWindow(WINDOW_NAME, 40, 40)
    cv2.imshow(WINDOW_NAME, initial_frame)
    # TOPMOST intentionally disabled: do not let OpenCV compete with Camera Emulator / Imou.
    cv2.waitKey(30)

    detector = SCRFD(
        str(SCRFD_MODEL),
        confidence_threshold=0.5,
        nms_threshold=0.4,
    )

    aligner = FaceAligner()
    arcface = ArcFace(str(ARCFACE_MODEL))
    vector_store = VectorStore()

    try:
        tts = TTSNotifier()
        print("TTS initialized.")
    except Exception as exc:
        tts = None
        print("TTS initialization warning:", exc)

    print("System initialized.")
    if tts is not None:
        tts.speak("Initializing Smart Attendance system.")

    print("Faces will be checked from nearest to farthest.")
    print("Only one student can be marked per frame.")
    print("Unknown faces will receive a spoken warning.")
    print("Attendance window: 60 minutes.")
    print("After 60 minutes, the camera closes and the WhatsApp queue starts.")
    print("WhatsApp queue delay: 10 seconds between recipients.")
    print("Q = quit")

    previous_q_state = False
    last_unknown_tts_time = 0.0
    last_known_face_time = 0.0

    attended_students: set[int] = set()
    student_cache: dict[int, tuple] = {}
    already_attended_visible_since: dict[int, float] = {}
    already_attended_warned: set[int] = {}
    late_students_warned: set[int] = set()

    attendance_window_start = time.monotonic()
    attendance_window_open = True

    whatsapp_queue: list[dict] = []
    whatsapp_queue_path = (
        project_root
        / "data"
        / (
            "whatsapp_queue_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
    )

    # Shared worker state. The display thread never waits for recognition.
    state_lock = threading.Lock()
    state = {
        "busy": False,
        "cooldown_until": 0.0,
        "events": [],
    }

    def run_recognition_pass(frame_to_process):
        nonlocal last_known_face_time
        nonlocal last_unknown_tts_time

        try:
            detections = detector.detect(frame_to_process)

            detections = sorted(
                detections,
                key=lambda detection: max(
                    0.0,
                    float(detection["box"][2] - detection["box"][0]),
                )
                * max(
                    0.0,
                    float(detection["box"][3] - detection["box"][1]),
                ),
                reverse=True,
            )

            currently_visible_already_attended: set[int] = set()
            events = []

            for detection in detections:
                result = process_one_face(
                    frame=frame_to_process,
                    detection=detection,
                    aligner=aligner,
                    arcface=arcface,
                    vector_store=vector_store,
                    attended_students=attended_students,
                    student_cache=student_cache,
                    tts=tts,
                    attendance_window_open=attendance_window_open,
                )

                if result["status"] in {
                    "attended",
                    "already_attended",
                    "late",
                }:
                    last_known_face_time = time.monotonic()

                if result["status"] == "attended":
                    with state_lock:
                        state["cooldown_until"] = (
                            time.monotonic()
                            + NEW_STUDENT_ATTENDANCE_COOLDOWN_SECONDS
                        )

                    whatsapp_queue.append(
                        {
                            "student_id": result["student_id"],
                            "name": result["name"],
                            "phone_number": result["phone_number"],
                            "attendance_datetime": result["attendance_datetime"],
                            "status": "pending",
                        }
                    )
                    save_whatsapp_queue(
                        whatsapp_queue,
                        whatsapp_queue_path,
                    )
                    events.append((detection, result))
                    break

                if result["status"] == "already_attended":
                    student_id = int(result["student_id"])
                    currently_visible_already_attended.add(student_id)
                    now = time.monotonic()

                    if student_id not in already_attended_visible_since:
                        already_attended_visible_since[student_id] = now

                    if (
                        tts is not None
                        and student_id not in already_attended_warned
                        and (
                            now
                            - already_attended_visible_since[student_id]
                        )
                        >= ALREADY_ATTENDED_CONFIRMATION_SECONDS
                    ):
                        tts.speak(
                            "Your attendance is already marked. "
                            "Please go from here."
                        )
                        already_attended_warned.add(student_id)

                    events.append((detection, result))
                    break

                if result["status"] == "late":
                    student_id = int(result["student_id"])

                    if (
                        tts is not None
                        and student_id not in late_students_warned
                    ):
                        tts.speak(
                            "Sorry, you are too late for attendance."
                        )
                        late_students_warned.add(student_id)

                    events.append((detection, result))
                    break

                if result["status"] == "unknown":
                    now = time.monotonic()

                    face_loss_grace_expired = (
                        last_known_face_time == 0.0
                        or (
                            now - last_known_face_time
                            >= FACE_LOSS_GRACE_SECONDS
                        )
                    )

                    if (
                        tts is not None
                        and face_loss_grace_expired
                        and (
                            now - last_unknown_tts_time
                            >= UNKNOWN_TTS_COOLDOWN_SECONDS
                        )
                    ):
                        tts.speak("You are not registered.")
                        last_unknown_tts_time = now

            for student_id in list(
                already_attended_visible_since
            ):
                if (
                    student_id
                    not in currently_visible_already_attended
                ):
                    already_attended_visible_since.pop(
                        student_id,
                        None,
                    )

            with state_lock:
                state["events"] = events

        except Exception as exc:
            print("Background recognition error:", exc)
        finally:
            with state_lock:
                state["busy"] = False

    try:
        last_recognition_time = 0.0

        while True:
            if (
                attendance_window_open
                and (
                    time.monotonic()
                    - attendance_window_start
                    >= ATTENDANCE_WINDOW_SECONDS
                )
            ):
                attendance_window_open = False

                print("60-minute attendance window ended.")
                print("No further attendance will be recorded.")

                close_camera(camera)

                # Stop and discard all pending speech as soon as the camera closes.
                if tts is not None:
                    tts.shutdown(wait=True)

                send_whatsapp_queue(
                    whatsapp_queue,
                    whatsapp_queue_path,
                )

                print("WhatsApp queue processing finished.")
                break

            frame = camera.read()

            if frame is None:
                time.sleep(0.003)
                cv2.waitKey(1)
                continue

            display = frame.copy()

            # Consume only the latest completed recognition result.
            with state_lock:
                completed_events = state["events"]
                state["events"] = []
                recognition_busy = state["busy"]
                cooldown_until = state["cooldown_until"]

            if completed_events:
                for detection, result in completed_events:
                    if result["status"] in {
                        "attended",
                        "already_attended",
                    }:
                        draw_face(
                            display,
                            detection,
                            display_name=result["name"],
                        )

            now = time.monotonic()

            if (
                not recognition_busy
                and now - last_recognition_time
                >= RECOGNITION_INTERVAL_SECONDS
                and now >= cooldown_until
            ):
                last_recognition_time = now

                with state_lock:
                    if not state["busy"]:
                        state["busy"] = True
                        start_worker = True
                    else:
                        start_worker = False

                if start_worker:
                    worker_frame = frame.copy()
                    threading.Thread(
                        target=run_recognition_pass,
                        args=(worker_frame,),
                        daemon=True,
                        name="attendance-recognition-worker",
                    ).start()

            cv2.putText(
                display,
                f"Registered attended: {len(attended_students)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                display,
                "Q = Quit",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # CRITICAL: refresh the visible camera before any expensive
            # recognition work can run. Recognition is now background-only.
            cv2.imshow(
                WINDOW_NAME,
                display,
            )

            # TOPMOST intentionally disabled: do not let OpenCV compete with Camera Emulator / Imou.

            q_state = is_key_down(VK_Q)
            q_pressed = (
                q_state
                and not previous_q_state
            )
            previous_q_state = q_state

            cv2.waitKey(1)

            if q_pressed:
                break

    finally:
        close_camera(camera)

        if tts is not None:
            tts.shutdown(wait=True)

    print()
    print("=" * 60)
    print("ATTENDANCE SESSION ENDED")
    print("Students attended:", len(attended_students))
    print("=" * 60)


if __name__ == "__main__":
    main()
