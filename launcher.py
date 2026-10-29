import os
import sys
from pathlib import Path

BASE_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
os.chdir(BASE_DIR)

def main():
    from db_setup import ensure_database_setup
    if not ensure_database_setup():
        return
    mode = sys.argv[1].lower() if len(sys.argv) > 1 else "frontend"
    if mode == "attendance":
        from attendance import main as attendance_main
        return attendance_main()
    if mode == "registration":
        from register_student import register_student
        name = None
        phone = None
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == "--name" and i + 1 < len(sys.argv):
                name = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == "--phone" and i + 1 < len(sys.argv):
                phone = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        return register_student(provided_name=name, provided_phone=phone)
    from frontend import AttendanceFrontend
    import tkinter as tk
    root = tk.Tk()
    AttendanceFrontend(root)
    root.mainloop()

if __name__ == "__main__":
    main()
