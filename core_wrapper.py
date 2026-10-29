import os
import sys
from pathlib import Path

BASE = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent)) if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
os.chdir(BASE)


def main():
    args = sys.argv[1:]

    if "--attendance" in args:
        from attendance import main as attendance_main
        sys.argv = [sys.argv[0]] + [x for x in args if x != "--attendance"]
        return attendance_main()

    if "--registration" in args:
        from register_student import register_student
        forwarded = [x for x in args if x != "--registration"]
        sys.argv = [sys.argv[0]] + forwarded
        name = None
        phone = None
        i = 0
        while i < len(forwarded):
            if forwarded[i] == "--name" and i + 1 < len(forwarded):
                name = forwarded[i + 1]
                i += 2
            elif forwarded[i] == "--phone" and i + 1 < len(forwarded):
                phone = forwarded[i + 1]
                i += 2
            else:
                i += 1
        return register_student(provided_name=name, provided_phone=phone)

    from frontend import SmartAttendanceFrontend
    import tkinter as tk
    root = tk.Tk()
    SmartAttendanceFrontend(root)
    root.mainloop()


if __name__ == "__main__":
    main()

