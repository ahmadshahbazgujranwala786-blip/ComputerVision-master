# =========================================================
# REGISTRATION WHATSAPP TEMPLATES
# =========================================================

def student_registered_template(
    student_name: str,
    student_id: int,
) -> str:
    """
    Create the WhatsApp message sent after successful registration.

    The template confirms that the student has been successfully
    enrolled and includes the generated student ID.
    """

    return (
        "Assalam o Alaikum,\n\n"
        f"Your student registration has been completed successfully.\n\n"
        f"Student ID: {student_id}\n\n"
        f"Your face is now registered in the attendance system.\n\n"
        f"Thank you."
    )


def registration_failed_template() -> str:
    """
    Create a message for a failed registration attempt.

    This template is available for future notification handling but
    is not sent automatically when the registration process fails.
    """

    return (
        "Student registration could not be completed.\n\n"
        "Please contact the administrator."
    )