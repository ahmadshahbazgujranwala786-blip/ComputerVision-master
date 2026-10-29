import os
import subprocess
import sys
from datetime import datetime, time as dt_time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import psycopg
from dotenv import load_dotenv

from src.vector_store import VectorStore

PROJECT_ROOT = Path(__file__).resolve().parent
ATTENDANCE_SCRIPT = PROJECT_ROOT / "attendance.py"
REGISTRATION_SCRIPT = PROJECT_ROOT / "register_student.py"

load_dotenv(PROJECT_ROOT / ".env", override=True)

# ---------------------------------------------------------------------------
# Futuristic desktop UI
# ---------------------------------------------------------------------------
BG = "#070b14"
PANEL = "#0d1422"
PANEL_2 = "#111b2d"
PANEL_3 = "#17243a"
TEXT = "#eef4ff"
MUTED = "#a7b4c8"
ACCENT = "#5aa2ff"
ACCENT_2 = "#8b7cff"
SUCCESS = "#38d39f"
DANGER = "#ff6378"
WARNING = "#f5bd55"
BORDER = "#22324d"
FONT = "Segoe UI"

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attendance_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

REQUIRED_COLUMNS = {
    "students": {"student_id", "name", "phone_number"},
    "attendance": {"id", "student_id", "date", "time", "status"},
}
PROTECTED_COLUMNS = {
    "students": {"student_id", "name", "phone_number"},
    "attendance": {"id", "student_id", "date", "time", "status"},
}
TYPE_OPTIONS = [
    "TEXT",
    "INTEGER",
    "BIGINT",
    "BOOLEAN",
    "DATE",
    "TIME",
    "TIMESTAMP",
]


def db_connect():
    if not DB_PASSWORD:
        raise RuntimeError("DB_PASSWORD is not configured in .env.")
    return psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def quote_identifier(name):
    return '"' + str(name).replace('"', '""') + '"'


def get_columns(table):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT column_name, data_type, is_nullable, column_default
                FROM information_schema.columns
                WHERE table_schema = 'public' AND table_name = %s
                ORDER BY ordinal_position
                """,
                (table,),
            )
            return cur.fetchall()


def get_rows(table, columns):
    if not columns:
        return []
    col_sql = ", ".join(quote_identifier(c[0]) for c in columns)
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT {col_sql} FROM {quote_identifier(table)} ORDER BY 1"
            )
            return cur.fetchall()


def table_count(table):
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}")
            return int(cur.fetchone()[0])


def current_attendance_window():
    """Return the local 12-hour attendance window containing now."""
    now = datetime.now()
    start_hour = 0 if now.hour < 12 else 12
    start = datetime.combine(now.date(), dt_time(start_hour, 0, 0))
    end = datetime.combine(now.date(), dt_time(12 if start_hour == 0 else 0, 0, 0))
    if end <= start:
        from datetime import timedelta
        end += timedelta(days=1)
    return start, end


def current_attendance_count():
    """Count attendance records only inside the active 12-hour window."""
    start, end = current_attendance_window()
    with db_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(*)
                FROM attendance
                WHERE (date + time) >= %s
                  AND (date + time) < %s
                """,
                (start, end),
            )
            return int(cur.fetchone()[0]), start, end


class SmartAttendanceFrontend:
    def __init__(self, root):
        self.root = root
        self.root.title("Smart Attendance • Command Center")
        self.root.geometry("1280x780")
        self.root.minsize(1080, 680)
        self.root.configure(bg=BG)

        self.child_processes = {}
        self.db_window = None
        self.current_db_table = None
        self.db_tree = None
        self.db_columns = []

        self._configure_styles()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.build_ui()
        self.refresh_status()

    # ------------------------------------------------------------------
    # Styling
    # ------------------------------------------------------------------
    def _configure_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Modern.Treeview",
            background=PANEL_2,
            fieldbackground=PANEL_2,
            foreground=TEXT,
            rowheight=36,
            borderwidth=0,
            font=(FONT, 10),
        )
        style.map(
            "Modern.Treeview",
            background=[("selected", "#274d7c")],
            foreground=[("selected", "#ffffff")],
        )
        style.configure(
            "Modern.Treeview.Heading",
            background=PANEL_3,
            foreground="#b9c9df",
            relief="flat",
            borderwidth=0,
            font=(FONT, 9, "bold"),
            padding=(10, 10),
        )
        style.configure(
            "Modern.TCombobox",
            fieldbackground=PANEL_3,
            background=PANEL_3,
            foreground=TEXT,
            arrowcolor=TEXT,
            bordercolor=BORDER,
            lightcolor=PANEL_3,
            darkcolor=PANEL_3,
            padding=8,
        )
        style.map(
            "Modern.TCombobox",
            fieldbackground=[("readonly", PANEL_3)],
            foreground=[("readonly", TEXT)],
        )

    def make_button(self, parent, text, command, accent=ACCENT, width=None):
        options = dict(
            text=text,
            command=command,
            font=(FONT, 10, "bold"),
            fg="#ffffff",
            bg=accent,
            activebackground=accent,
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=18,
            pady=11,
        )
        if width:
            options["width"] = width
        return tk.Button(parent, **options)

    def make_outline_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=(FONT, 9, "bold"),
            fg=TEXT,
            bg=PANEL_3,
            activebackground="#203452",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            cursor="hand2",
            padx=14,
            pady=9,
        )

    # ------------------------------------------------------------------
    # Main command-center UI
    # ------------------------------------------------------------------
    def build_ui(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=34, pady=(28, 18))

        brand = tk.Frame(top, bg=BG)
        brand.pack(side="left")

        tk.Label(
            brand,
            text="SMART ATTENDANCE",
            font=(FONT, 27, "bold"),
            fg=TEXT,
            bg=BG,
        ).pack(anchor="w")
        tk.Label(
            brand,
            text="COMMAND CENTER  /  FACE • DATABASE • OPERATIONS",
            font=(FONT, 9, "bold"),
            fg=ACCENT,
            bg=BG,
        ).pack(anchor="w", pady=(5, 0))

        self.system_pill = tk.Label(
            top,
            text="●  SYSTEM READY",
            font=(FONT, 10, "bold"),
            fg=SUCCESS,
            bg=PANEL,
            padx=16,
            pady=9,
        )
        self.system_pill.pack(side="right", pady=8)

        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x", padx=34)

        # Scrollable main content area so the full Command Center remains
        # accessible on smaller/lower-resolution screens.
        main_area = tk.Frame(self.root, bg=BG)
        main_area.pack(fill="both", expand=True, padx=0, pady=0)

        body_canvas = tk.Canvas(main_area, bg=BG, highlightthickness=0, bd=0)
        body_scrollbar = ttk.Scrollbar(
            main_area, orient="vertical", command=body_canvas.yview
        )
        body = tk.Frame(body_canvas, bg=BG)

        body_window = body_canvas.create_window((0, 0), window=body, anchor="nw")
        body_canvas.configure(yscrollcommand=body_scrollbar.set)

        body_scrollbar.pack(side="right", fill="y", padx=(0, 8), pady=4)
        body_canvas.pack(side="left", fill="both", expand=True)

        def _update_body_scrollregion(_event=None):
            body_canvas.configure(scrollregion=body_canvas.bbox("all"))

        def _resize_body_width(event):
            body_canvas.itemconfigure(body_window, width=event.width)

        body.bind("<Configure>", _update_body_scrollregion)
        body_canvas.bind("<Configure>", _resize_body_width)

        # Mouse-wheel scrolling.
        body_canvas.bind_all(
            "<MouseWheel>",
            lambda event: body_canvas.yview_scroll(
                int(-1 * (event.delta / 120)), "units"
            ),
        )

        # Left rail
        rail = tk.Frame(body, bg=PANEL, width=225)
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)

        tk.Label(
            rail,
            text="OPERATIONS",
            font=(FONT, 9, "bold"),
            fg=MUTED,
            bg=PANEL,
        ).pack(anchor="w", padx=22, pady=(24, 14))

        tk.Label(
            rail,
            text="DATABASE",
            font=(FONT, 9, "bold"),
            fg=MUTED,
            bg=PANEL,
        ).pack(anchor="w", padx=22, pady=(30, 14))

        self.make_outline_button(
            rail, "ATTENDANCE DATABASE", lambda: self.open_database("attendance")
        ).pack(fill="x", padx=16, pady=4)

        self.make_outline_button(
            rail, "REGISTRATION DATABASE", lambda: self.open_database("students")
        ).pack(fill="x", padx=16, pady=4)

        tk.Frame(rail, bg=BORDER, height=1).pack(fill="x", padx=18, pady=25)

        tk.Label(
            rail,
            text="DATABASE LINK",
            font=(FONT, 8, "bold"),
            fg=MUTED,
            bg=PANEL,
        ).pack(anchor="w", padx=22)
        self.db_link = tk.Label(
            rail,
            text="PostgreSQL  •  checking...",
            font=(FONT, 9),
            fg=TEXT,
            bg=PANEL,
            wraplength=175,
            justify="left",
        )
        self.db_link.pack(anchor="w", padx=22, pady=(6, 0))

        # Main cards
        content = tk.Frame(body, bg=BG)
        content.pack(side="left", fill="both", expand=True, padx=(24, 0))

        welcome = tk.Frame(content, bg=BG)
        welcome.pack(fill="x", pady=(0, 20))
        tk.Label(
            welcome,
            text="Operations overview",
            font=(FONT, 19, "bold"),
            fg=TEXT,
            bg=BG,
        ).pack(anchor="w")
        tk.Label(
            welcome,
            text="Control live attendance and manage the connected PostgreSQL records from one interface.",
            font=(FONT, 10),
            fg=MUTED,
            bg=BG,
        ).pack(anchor="w", pady=(5, 0))

        cards = tk.Frame(content, bg=BG)
        cards.pack(fill="x")
        cards.columnconfigure(0, weight=1)
        cards.columnconfigure(1, weight=1)

        self.operation_card(
            cards, 0,
            "ATTENDANCE",
            "Live face-recognition session",
            "Recognize registered students and record attendance.",
            ACCENT,
            "▶",
            self.start_attendance,
        )
        self.operation_card(
            cards, 1,
            "REGISTRATION",
            "New student enrollment",
            "Capture a face and create the PostgreSQL + FAISS identity.",
            SUCCESS,
            "+",
            self.open_registration,
        )

        stats = tk.Frame(content, bg=BG)
        stats.pack(fill="x", pady=(22, 0))
        stats.columnconfigure(0, weight=1)
        stats.columnconfigure(1, weight=1)
        self.stat_card(stats, 0, "REGISTERED STUDENTS", "students_count")
        self.stat_card(stats, 1, "ATTENDANCE • CURRENT 12H", "attendance_count", "attendance_period")

        footer = tk.Frame(self.root, bg=BG)
        footer.pack(fill="x", padx=34, pady=(0, 22))
        self.status_label = tk.Label(
            footer, text="System ready",
            font=(FONT, 9), fg=SUCCESS, bg=BG
        )
        self.status_label.pack(side="left")
        tk.Label(
            footer,
            text="Q closes an active camera session",
            font=(FONT, 9),
            fg=MUTED,
            bg=BG,
        ).pack(side="right")

    def operation_card(self, parent, column, title, subtitle, detail, accent, symbol, command):
        card = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        card.grid(
            row=0, column=column, sticky="nsew",
            padx=(0, 10) if column == 0 else (10, 0)
        )

        icon = tk.Canvas(card, width=62, height=62, bg=PANEL, highlightthickness=0)
        icon.pack(anchor="w", padx=25, pady=(24, 15))
        icon.create_oval(4, 4, 58, 58, fill=accent, outline="")
        icon.create_text(31, 31, text=symbol, fill="white", font=(FONT, 24, "bold"))

        tk.Label(
            card, text=title, font=(FONT, 17, "bold"), fg=TEXT, bg=PANEL
        ).pack(anchor="w", padx=25)
        tk.Label(
            card, text=subtitle, font=(FONT, 10, "bold"), fg=accent, bg=PANEL
        ).pack(anchor="w", padx=25, pady=(5, 0))
        tk.Label(
            card, text=detail, font=(FONT, 9), fg=MUTED, bg=PANEL,
            wraplength=360, justify="left"
        ).pack(anchor="w", padx=25, pady=(8, 20))
        self.make_button(
            card, f"{symbol}  OPEN", command, accent
        ).pack(anchor="w", padx=25, pady=(0, 25))

    def stat_card(self, parent, column, title, attr, meta_attr=None):
        card = tk.Frame(parent, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        card.grid(row=0, column=column, sticky="ew",
                  padx=(0, 10) if column == 0 else (10, 0))
        tk.Label(
            card, text=title, font=(FONT, 9, "bold"), fg=TEXT, bg=PANEL
        ).pack(anchor="w", padx=20, pady=(16, 3))
        label = tk.Label(
            card, text="—", font=(FONT, 24, "bold"), fg=TEXT, bg=PANEL
        )
        label.pack(anchor="w", padx=20, pady=(0, 2))
        setattr(self, attr, label)
        if meta_attr:
            meta = tk.Label(
                card, text="", font=(FONT, 9, "bold"), fg=ACCENT, bg=PANEL
            )
            meta.pack(anchor="w", padx=20, pady=(0, 14))
            setattr(self, meta_attr, meta)

    # ------------------------------------------------------------------
    # Process launching
    # ------------------------------------------------------------------
    def start_attendance(self):
        command = (
            [sys.executable, "--attendance"]
            if getattr(sys, "frozen", False)
            else [sys.executable, str(ATTENDANCE_SCRIPT)]
        )
        self.launch("attendance", command)

    def open_registration(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Register Student")
        dialog.geometry("500x430")
        dialog.resizable(False, False)
        dialog.configure(bg=PANEL)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="NEW STUDENT", font=(FONT, 21, "bold"),
            fg=TEXT, bg=PANEL
        ).pack(anchor="w", padx=34, pady=(30, 4))
        tk.Label(
            dialog, text="Create the PostgreSQL identity, then capture the face.",
            font=(FONT, 10), fg=MUTED, bg=PANEL
        ).pack(anchor="w", padx=34, pady=(0, 24))

        fields = tk.Frame(dialog, bg=PANEL)
        fields.pack(fill="x", padx=34)
        name_var = tk.StringVar()
        phone_var = tk.StringVar()

        self.field(fields, "Student name", name_var)
        self.field(fields, "Phone number", phone_var)

        def submit():
            name = name_var.get().strip()
            phone = phone_var.get().strip()
            if not name or not phone:
                messagebox.showwarning(
                    "Missing information",
                    "Please enter both the student name and phone number.",
                    parent=dialog,
                )
                return
            dialog.destroy()
            command = (
                [sys.executable, "--registration", "--name", name, "--phone", phone]
                if getattr(sys, "frozen", False)
                else [sys.executable, str(REGISTRATION_SCRIPT), "--name", name, "--phone", phone]
            )
            self.launch("registration", command)

        self.make_button(dialog, "START REGISTRATION", submit, SUCCESS).pack(
            anchor="w", padx=34, pady=28
        )

    def field(self, parent, label, variable):
        tk.Label(
            parent, text=label, font=(FONT, 9, "bold"),
            fg=MUTED, bg=PANEL
        ).pack(anchor="w", pady=(0, 5))
        entry = tk.Entry(
            parent, textvariable=variable, font=(FONT, 11),
            fg=TEXT, bg=PANEL_3, insertbackground=TEXT,
            relief="flat", bd=0,
        )
        entry.pack(fill="x", ipady=11, pady=(0, 16))

    def launch(self, kind, command):
        process = self.child_processes.get(kind)
        if process and process.poll() is None:
            messagebox.showinfo(
                "Already running", f"{kind.title()} is already running.", parent=self.root
            )
            return
        try:
            process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            )
            self.child_processes[kind] = process
            self.status_label.config(text=f"{kind.title()} started", fg=SUCCESS)
        except Exception as exc:
            messagebox.showerror(
                "Could not start",
                f"Failed to start {kind}:\n\n{exc}",
                parent=self.root,
            )

    # ------------------------------------------------------------------
    # Database manager
    # ------------------------------------------------------------------
    def open_database(self, table):
        if self.db_window and self.db_window.winfo_exists():
            self.db_window.destroy()

        self.current_db_table = table
        title = "Attendance Database" if table == "attendance" else "Registration Database"

        self.db_window = tk.Toplevel(self.root)
        win = self.db_window
        win.title(f"Smart Attendance • {title}")
        win.geometry("1180x700")
        win.minsize(950, 580)
        win.configure(bg=BG)
        win.transient(self.root)

        header = tk.Frame(win, bg=BG)
        header.pack(fill="x", padx=28, pady=(22, 12))
        tk.Label(
            header, text=title.upper(), font=(FONT, 20, "bold"),
            fg=TEXT, bg=BG
        ).pack(side="left")
        self.db_count_label = tk.Label(
            header, text="", font=(FONT, 9, "bold"), fg=ACCENT, bg=BG
        )
        self.db_count_label.pack(side="left", padx=14, pady=5)

        toolbar = tk.Frame(win, bg=PANEL)
        toolbar.pack(fill="x", padx=28, pady=(0, 14))

        self.make_button(toolbar, "＋  ADD ROW", lambda: self.db_add_row(), SUCCESS).pack(
            side="left", padx=8, pady=9
        )
        self.make_outline_button(toolbar, "EDIT SELECTED", lambda: self.db_edit_row()).pack(
            side="left", padx=4, pady=9
        )
        self.make_outline_button(toolbar, "DELETE SELECTED", lambda: self.db_delete_row()).pack(
            side="left", padx=4, pady=9
        )
        self.make_outline_button(toolbar, "↻  REFRESH", self.db_refresh).pack(
            side="left", padx=4, pady=9
        )
        self.make_outline_button(toolbar, "COLUMN MANAGER", self.open_column_manager).pack(
            side="right", padx=8, pady=9
        )

        table_frame = tk.Frame(win, bg=PANEL, highlightthickness=1, highlightbackground=BORDER)
        table_frame.pack(fill="both", expand=True, padx=28, pady=(0, 12))

        self.db_tree = ttk.Treeview(
            table_frame,
            style="Modern.Treeview",
            show="headings",
            selectmode="browse",
        )
        yscroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.db_tree.yview)
        xscroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.db_tree.xview)
        self.db_tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.db_tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        self.db_tree.bind("<Double-1>", lambda _event: self.db_edit_row())
        self.db_refresh()

        footer = tk.Frame(win, bg=BG)
        footer.pack(fill="x", padx=28, pady=(0, 18))
        tk.Label(
            footer,
            text="Double-click a row to edit • Select a row then Delete • Column Manager changes the PostgreSQL schema",
            font=(FONT, 9), fg=MUTED, bg=BG
        ).pack(side="left")

    def db_refresh(self):
        if not self.db_tree or not self.db_tree.winfo_exists():
            return
        try:
            self.db_columns = get_columns(self.current_db_table)
            columns = [c[0] for c in self.db_columns]
            self.db_tree.delete(*self.db_tree.get_children())
            self.db_tree["columns"] = columns

            for column in columns:
                self.db_tree.heading(column, text=column.upper())
                self.db_tree.column(column, width=max(125, min(230, len(column) * 12 + 50)))

            rows = get_rows(self.current_db_table, self.db_columns)
            for row in rows:
                values = ["" if value is None else str(value) for value in row]
                self.db_tree.insert("", "end", values=values)

            count = len(rows)
            self.db_count_label.config(text=f"{count} RECORDS")
        except Exception as exc:
            messagebox.showerror("Database error", str(exc), parent=self.db_window)

    def selected_db_row(self):
        selection = self.db_tree.selection()
        if not selection:
            messagebox.showinfo("Select a row", "Please select a row first.", parent=self.db_window)
            return None
        values = self.db_tree.item(selection[0], "values")
        return dict(zip([c[0] for c in self.db_columns], values))

    def db_add_row(self):
        self.open_row_editor("ADD RECORD", None)

    def db_edit_row(self):
        row = self.selected_db_row()
        if row is not None:
            self.open_row_editor("EDIT RECORD", row)

    def open_row_editor(self, title, row):
        dialog = tk.Toplevel(self.db_window)
        dialog.title(f"{title} • {self.current_db_table}")
        dialog.geometry("560x620")
        dialog.configure(bg=PANEL)
        dialog.transient(self.db_window)
        dialog.grab_set()

        tk.Label(
            dialog, text=title, font=(FONT, 19, "bold"), fg=TEXT, bg=PANEL
        ).pack(anchor="w", padx=30, pady=(24, 3))
        tk.Label(
            dialog,
            text=f"Table: {self.current_db_table}",
            font=(FONT, 9, "bold"), fg=ACCENT, bg=PANEL
        ).pack(anchor="w", padx=30, pady=(0, 18))

        canvas = tk.Canvas(dialog, bg=PANEL, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        form = tk.Frame(canvas, bg=PANEL)
        form_id = canvas.create_window((0, 0), window=form, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True, padx=(25, 0))
        scrollbar.pack(side="right", fill="y", padx=(0, 20))
        form.bind("<Configure>", lambda _e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(form_id, width=e.width))

        variables = {}
        editable_columns = []

        for name, data_type, nullable, default in self.db_columns:
            # Identity/default primary keys are generated by PostgreSQL.
            generated = default and ("nextval(" in default or "identity" in default.lower())
            if generated and row is None:
                continue
            if name in {"student_id", "id"} and row is None:
                continue

            tk.Label(
                form, text=f"{name}  •  {data_type}",
                font=(FONT, 8, "bold"), fg=MUTED, bg=PANEL
            ).pack(anchor="w", pady=(7, 4))
            var = tk.StringVar(value="" if row is None else str(row.get(name, "")))
            entry = tk.Entry(
                form, textvariable=var, font=(FONT, 10),
                fg=TEXT, bg=PANEL_3, insertbackground=TEXT,
                relief="flat", bd=0,
            )
            entry.pack(fill="x", ipady=9, pady=(0, 8))
            variables[name] = var
            editable_columns.append(name)

        def save():
            try:
                values = {name: variables[name].get() for name in editable_columns}
                self.save_db_row(row, values)
                dialog.destroy()
                self.db_refresh()
                self.refresh_status()
            except Exception as exc:
                messagebox.showerror("Save failed", str(exc), parent=dialog)

        self.make_button(dialog, "SAVE CHANGES", save, SUCCESS).pack(
            anchor="w", padx=30, pady=20
        )

    def save_db_row(self, original, values):
        table = self.current_db_table
        if table == "attendance":
            if not values.get("student_id"):
                raise ValueError("student_id is required.")
            if not values.get("date"):
                raise ValueError("date is required.")
            if not values.get("time"):
                raise ValueError("time is required.")
            if not values.get("status"):
                raise ValueError("status is required.")
        elif table == "students" and not values.get("name"):
            raise ValueError("Student name is required.")

        with db_connect() as conn:
            with conn.cursor() as cur:
                if original is None:
                    names = []
                    params = []
                    for name, value in values.items():
                        if value == "":
                            continue
                        names.append(name)
                        params.append(value)
                    if not names:
                        raise ValueError("Enter at least one value.")
                    cols = ", ".join(quote_identifier(n) for n in names)
                    placeholders = ", ".join(["%s"] * len(params))
                    cur.execute(
                        f"INSERT INTO {quote_identifier(table)} ({cols}) VALUES ({placeholders})",
                        params,
                    )
                else:
                    pk = "id" if table == "attendance" else "student_id"
                    pk_value = original[pk]
                    assignments = []
                    params = []
                    for name, value in values.items():
                        if name == pk:
                            continue
                        assignments.append(f"{quote_identifier(name)} = %s")
                        params.append(None if value == "" else value)
                    if not assignments:
                        return
                    params.append(pk_value)
                    cur.execute(
                        f"UPDATE {quote_identifier(table)} SET {', '.join(assignments)} WHERE {quote_identifier(pk)} = %s",
                        params,
                    )
            conn.commit()

    def db_delete_row(self):
        row = self.selected_db_row()
        if row is None:
            return
        table = self.current_db_table
        pk = "id" if table == "attendance" else "student_id"
        if not messagebox.askyesno(
            "Delete record",
            f"Delete {table} record {row.get(pk)}?\n\nThis cannot be undone.",
            parent=self.db_window,
        ):
            return
        try:
            with db_connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"DELETE FROM {quote_identifier(table)} WHERE {quote_identifier(pk)} = %s",
                        (row[pk],),
                    )
                conn.commit()

            # A student's FAISS embedding uses the same student_id as PostgreSQL.
            # Remove it only when deleting from the students table; attendance
            # record deletion must never affect face embeddings.
            if table == "students":
                try:
                    vector_store = VectorStore()
                    removed = vector_store.remove_vector(int(row["student_id"]))
                    if int(removed) > 0:
                        print(
                            f"Removed FAISS embedding for student {row['student_id']}."
                        )
                    else:
                        print(
                            f"No FAISS embedding found for student {row['student_id']}."
                        )
                except Exception as vector_exc:
                    self.db_refresh()
                    self.refresh_status()
                    messagebox.showwarning(
                        "FAISS cleanup warning",
                        "The student was deleted from PostgreSQL, but their FAISS "
                        "embedding could not be removed. Do not register the same "
                        "student again until this is corrected. Details: "
                        f"{vector_exc}",
                        parent=self.db_window,
                    )
                    return

            self.db_refresh()
            self.refresh_status()
        except Exception as exc:
            messagebox.showerror("Delete failed", str(exc), parent=self.db_window)

    # ------------------------------------------------------------------
    # Column manager
    # ------------------------------------------------------------------
    def open_column_manager(self):
        table = self.current_db_table
        dialog = tk.Toplevel(self.db_window)
        dialog.title(f"Column Manager • {table}")
        dialog.geometry("720x570")
        dialog.configure(bg=PANEL)
        dialog.transient(self.db_window)
        dialog.grab_set()

        tk.Label(
            dialog, text="COLUMN MANAGER", font=(FONT, 19, "bold"),
            fg=TEXT, bg=PANEL
        ).pack(anchor="w", padx=28, pady=(24, 4))
        tk.Label(
            dialog,
            text="Add, rename, or remove custom PostgreSQL columns. Core application columns are protected.",
            font=(FONT, 9), fg=MUTED, bg=PANEL, wraplength=650, justify="left"
        ).pack(anchor="w", padx=28, pady=(0, 18))

        tree = ttk.Treeview(
            dialog, style="Modern.Treeview", show="headings", height=11,
            columns=("name", "type", "nullable", "default")
        )
        for col, text_ in [
            ("name", "COLUMN"), ("type", "TYPE"),
            ("nullable", "NULLABLE"), ("default", "DEFAULT")
        ]:
            tree.heading(col, text=text_)
            tree.column(col, width=180 if col == "name" else 145)
        tree.pack(fill="both", expand=True, padx=28)

        def refresh_columns():
            tree.delete(*tree.get_children())
            for name, dtype, nullable, default in get_columns(table):
                tree.insert(
                    "", "end",
                    values=(name, dtype, nullable, default or "—")
                )

        refresh_columns()

        buttons = tk.Frame(dialog, bg=PANEL)
        buttons.pack(fill="x", padx=28, pady=18)

        def add_column():
            form = tk.Toplevel(dialog)
            form.title("Add Column")
            form.geometry("430x280")
            form.configure(bg=PANEL)
            form.transient(dialog)
            form.grab_set()

            name_var = tk.StringVar()
            type_var = tk.StringVar(value="TEXT")

            tk.Label(form, text="Column name", fg=MUTED, bg=PANEL,
                     font=(FONT, 9, "bold")).pack(anchor="w", padx=25, pady=(24, 5))
            tk.Entry(form, textvariable=name_var, fg=TEXT, bg=PANEL_3,
                     insertbackground=TEXT, relief="flat").pack(fill="x", padx=25, ipady=9)

            tk.Label(form, text="Data type", fg=MUTED, bg=PANEL,
                     font=(FONT, 9, "bold")).pack(anchor="w", padx=25, pady=(15, 5))
            ttk.Combobox(
                form, textvariable=type_var, values=TYPE_OPTIONS,
                state="readonly", style="Modern.TCombobox"
            ).pack(fill="x", padx=25)

            def do_add():
                name = name_var.get().strip()
                if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
                    messagebox.showerror(
                        "Invalid column", "Use letters, numbers, and underscores; do not start with a number.",
                        parent=form
                    )
                    return
                try:
                    with db_connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                f"ALTER TABLE {quote_identifier(table)} ADD COLUMN {quote_identifier(name)} {type_var.get()}"
                            )
                        conn.commit()
                    form.destroy()
                    refresh_columns()
                    self.db_refresh()
                except Exception as exc:
                    messagebox.showerror("Add column failed", str(exc), parent=form)

            self.make_button(form, "ADD COLUMN", do_add, SUCCESS).pack(
                anchor="w", padx=25, pady=22
            )

        def rename_column():
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("Select a column", "Select a column first.", parent=dialog)
                return
            old = tree.item(selection[0], "values")[0]
            if old in PROTECTED_COLUMNS[table]:
                messagebox.showwarning(
                    "Protected column",
                    f"'{old}' is required by the application and cannot be renamed.",
                    parent=dialog,
                )
                return

            form = tk.Toplevel(dialog)
            form.title("Rename Column")
            form.geometry("430x230")
            form.configure(bg=PANEL)
            form.transient(dialog)
            form.grab_set()
            var = tk.StringVar(value=old)

            tk.Label(form, text=f"Rename '{old}' to", fg=MUTED, bg=PANEL,
                     font=(FONT, 9, "bold")).pack(anchor="w", padx=25, pady=(24, 5))
            tk.Entry(form, textvariable=var, fg=TEXT, bg=PANEL_3,
                     insertbackground=TEXT, relief="flat").pack(fill="x", padx=25, ipady=9)

            def do_rename():
                new = var.get().strip()
                if not new or not new.replace("_", "").isalnum() or new[0].isdigit():
                    messagebox.showerror("Invalid column", "Invalid column name.", parent=form)
                    return
                try:
                    with db_connect() as conn:
                        with conn.cursor() as cur:
                            cur.execute(
                                f"ALTER TABLE {quote_identifier(table)} RENAME COLUMN {quote_identifier(old)} TO {quote_identifier(new)}"
                            )
                        conn.commit()
                    form.destroy()
                    refresh_columns()
                    self.db_refresh()
                except Exception as exc:
                    messagebox.showerror("Rename failed", str(exc), parent=form)

            self.make_button(form, "RENAME COLUMN", do_rename, ACCENT).pack(
                anchor="w", padx=25, pady=22
            )

        def delete_column():
            selection = tree.selection()
            if not selection:
                messagebox.showinfo("Select a column", "Select a column first.", parent=dialog)
                return
            name = tree.item(selection[0], "values")[0]
            if name in PROTECTED_COLUMNS[table]:
                messagebox.showwarning(
                    "Protected column",
                    f"'{name}' is required by the application and cannot be deleted.",
                    parent=dialog,
                )
                return
            if not messagebox.askyesno(
                "Delete column",
                f"Delete column '{name}' from {table}?\n\nAll data in this column will be permanently removed.",
                parent=dialog,
            ):
                return
            try:
                with db_connect() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            f"ALTER TABLE {quote_identifier(table)} DROP COLUMN {quote_identifier(name)}"
                        )
                    conn.commit()
                refresh_columns()
                self.db_refresh()
            except Exception as exc:
                messagebox.showerror("Delete column failed", str(exc), parent=dialog)

        self.make_button(buttons, "＋ ADD", add_column, SUCCESS).pack(side="left", padx=4)
        self.make_outline_button(buttons, "RENAME", rename_column).pack(side="left", padx=4)
        self.make_outline_button(buttons, "DELETE", delete_column).pack(side="left", padx=4)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def refresh_status(self):
        active = []
        for kind, process in list(self.child_processes.items()):
            if process.poll() is None:
                active.append(kind.title())

        if active:
            self.status_label.config(text="Running: " + " • ".join(active), fg=SUCCESS)
            self.system_pill.config(text="●  ACTIVE", fg=SUCCESS)
        else:
            self.status_label.config(text="System ready", fg=SUCCESS)
            self.system_pill.config(text="●  SYSTEM READY", fg=SUCCESS)

        try:
            students = table_count("students")
            attendance, window_start, window_end = current_attendance_count()
            self.students_count.config(text=str(students))
            self.attendance_count.config(text=str(attendance))
            self.attendance_period.config(
                text=(
                    f"{window_start.strftime('%Y-%m-%d')}  •  "
                    f"{window_start.strftime('%H:%M')}–{window_end.strftime('%H:%M')}"
                )
            )
            self.db_link.config(text="PostgreSQL  •  CONNECTED", fg=SUCCESS)
        except Exception:
            self.students_count.config(text="—")
            self.attendance_count.config(text="—")
            if hasattr(self, "attendance_period"):
                self.attendance_period.config(text="12-hour window unavailable")
            self.db_link.config(text="PostgreSQL  •  OFFLINE", fg=DANGER)

        self.root.after(1500, self.refresh_status)

    def on_close(self):
        running = [
            process for process in self.child_processes.values()
            if process.poll() is None
        ]
        if running:
            if not messagebox.askyesno(
                "Active session",
                "A system session is still running. Close the frontend anyway?",
                parent=self.root,
            ):
                return
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    SmartAttendanceFrontend(root)
    root.mainloop()
