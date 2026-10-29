# =========================================================
# ATTENDANCE WHATSAPP TEMPLATES
# =========================================================

from datetime import datetime


def attendance_success_template(
    student_name: str,
    student_id: int,
    attendance_datetime: datetime,
) -> str:
    """
    Create the WhatsApp attendance confirmation message.

    The template receives the exact attendance datetime so that the
    message reflects the same date and time stored in PostgreSQL.
    """

    return (
        f"Assalam o Alaikum {student_name},\n\n"
        f"Your attendance has been marked successfully.\n\n"
        f"Student ID: {student_id}\n"
        f"Date: {attendance_datetime.strftime('%d-%m-%Y')}\n"
        f"Time: {attendance_datetime.strftime('%I:%M:%S %p')}\n\n"
        f"Thank you."
    )


def student_not_registered_template() -> str:
    """
    Create the message used when a face is not found in the
    registered FAISS database.

    This message is displayed locally and is not sent to an
    unregistered student's phone because no registered phone exists.
    """

    return (
        "Student is not registered.\n\n"
        "Attendance has NOT been recorded.\n"
        "Please register the student first."
    )