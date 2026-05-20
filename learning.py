import datetime
import sqlite3
from typing import Dict, List, Optional, Tuple


# ------------------------------------------------------------
# SM-2 algorithm (tuned for 0–5 grading)
# ------------------------------------------------------------
def sm2_update(
    grade: int,  # 0..5 (0=blackout, 1=wrong, 2=hard, 3=good, 4=easy, 5=perfect)
    repetition: int,
    difficulty: float,
    interval_days: int,
) -> Tuple[int, float, int]:
    """
    Apply SM-2 update rules.
    Returns (new_repetition, new_difficulty, new_interval_days)
    """
    if grade >= 3:  # correct response
        if repetition == 0:
            interval_days = 1
        elif repetition == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * difficulty)
        repetition += 1
    else:  # incorrect response
        repetition = 0
        interval_days = 1

    # Update difficulty factor
    difficulty += 0.1 - (5 - grade) * (0.08 + (5 - grade) * 0.02)
    if difficulty < 1.3:
        difficulty = 1.3

    return repetition, difficulty, interval_days


# ------------------------------------------------------------
# SQLite database wrapper
# ------------------------------------------------------------
class WordLearner:
    def __init__(self, db_path: str = "words.db"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS words (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                word TEXT UNIQUE NOT NULL,
                translation TEXT,
                cefr_level TEXT,
                repetition INTEGER DEFAULT 0,
                difficulty REAL DEFAULT 2.5,
                interval_days INTEGER DEFAULT 1,
                next_review DATE NOT NULL,
                last_review DATE
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_next_review ON words(next_review)"
        )
        self.conn.commit()

    def add_words(
        self,
        words: List[str],
        translations: Optional[Dict[str, str]] = None,
        cefr_map: Optional[Dict[str, str]] = None,
    ) -> int:
        """
        Insert new words into the learning queue (ignores duplicates).
        - translations: { word: translated_text }
        - cefr_map: { word: level } (optional)
        Returns number of newly added words.
        """
        if not words:
            return 0

        cursor = self.conn.cursor()
        now = datetime.date.today()
        added = 0

        for w in words:
            # Skip if already in DB
            cursor.execute("SELECT 1 FROM words WHERE word = ?", (w,))
            if cursor.fetchone():
                continue

            trans = translations.get(w, "") if translations else ""
            level = cefr_map.get(w, "") if cefr_map else ""

            # New word: first review is tomorrow (interval=1)
            next_review = now + datetime.timedelta(days=1)

            cursor.execute(
                """
                INSERT INTO words (word, translation, cefr_level, next_review)
                VALUES (?, ?, ?, ?)
            """,
                (w, trans, level, next_review),
            )
            added += 1

        self.conn.commit()
        return added

    def get_due_words(self, limit: int = 20) -> List[Tuple[str, str, int, float, int]]:
        """
        Return words whose next_review <= today.
        Each tuple: (word, translation, repetition, difficulty, interval_days)
        """
        today = datetime.date.today()
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT word, translation, repetition, difficulty, interval_days
            FROM words
            WHERE next_review <= ?
            ORDER BY next_review
            LIMIT ?
        """,
            (today, limit),
        )
        return cursor.fetchall()

    def update_review(self, word: str, grade: int) -> None:
        """
        Update word's scheduling data after a review.
        grade: 0..5  (see sm2_update)
        """
        cursor = self.conn.cursor()
        cursor.execute(
            """
            SELECT repetition, difficulty, interval_days
            FROM words WHERE word = ?
        """,
            (word,),
        )
        row = cursor.fetchone()
        if not row:
            return

        repetition, difficulty, interval = row
        new_rep, new_diff, new_interval = sm2_update(
            grade, repetition, difficulty, interval
        )

        today = datetime.date.today()
        next_review = today + datetime.timedelta(days=new_interval)

        cursor.execute(
            """
            UPDATE words
            SET repetition = ?, difficulty = ?, interval_days = ?,
                next_review = ?, last_review = ?
            WHERE word = ?
        """,
            (new_rep, new_diff, new_interval, next_review, today, word),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()


# ------------------------------------------------------------
# Simple interactive review session
# ------------------------------------------------------------
def review_session(learner: WordLearner, max_words: int = 10):
    """
    Console‑based review. Shows word + translation, asks for grade (0‑5).
    Stops when no due words or after `max_words` reviews.
    """
    due = learner.get_due_words(limit=max_words)
    if not due:
        print("🎉 No words due for review today! Come back tomorrow.")
        return

    print(f"\n📚 Review session – {len(due)} word(s) due\n")
    for word, trans, rep, diff, interval in due:
        print(f"\nWord: {word}")
        if trans:
            print(f"Meaning: {trans}")
        else:
            print("(no translation stored)")

        while True:
            try:
                grade = int(
                    input(
                        "How well did you know it? (0=blackout, 1=wrong, 2=hard, 3=good, 4=easy, 5=perfect): "
                    )
                )
                if 0 <= grade <= 5:
                    break
                print("Please enter a number between 0 and 5.")
            except ValueError:
                print("Invalid input. Enter a number 0‑5.")

        learner.update_review(word, grade)
        print("✓ Updated.")

    print("\n✅ Session finished. Keep learning!")
