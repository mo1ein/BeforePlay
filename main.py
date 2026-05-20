import csv
import html
import logging
import re
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

import pdfkit
import requests
from deep_translator import GoogleTranslator
from nltk import download as nltk_download
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from pypdf import PdfReader

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def _ensure_nltk_resources() -> None:
    for resource in ["names", "wordnet", "omw-1.4"]:
        try:
            if resource == "names":
                names.words()
            else:
                WordNetLemmatizer().lemmatize("test")
        except LookupError:
            nltk_download(resource, quiet=True)


_ensure_nltk_resources()


class DocumentParser:
    """Factory‑style parser that picks the right method based on file extension."""

    @staticmethod
    # todo: use better way to detect pdf or srt, extention is not a strong way
    def extract_text(file_path: Path, pages: Optional[list[int]] = None) -> str:
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            return DocumentParser._parse_pdf(file_path, pages)
        elif suffix == ".srt":
            return DocumentParser._parse_srt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}. Use .pdf or .srt")

    @staticmethod
    def _parse_pdf(path: Path, pages: Optional[list[int]] = None) -> str:
        reader = PdfReader(path)
        total_pages = len(reader.pages)
        full_text = []

        # If pages is None -> read all pages
        target_pages = pages if pages is not None else range(1, total_pages + 1)

        for page_num in target_pages:
            if page_num < 1 or page_num > total_pages:
                logger.warning(
                    f"Page {page_num} out of range (1-{total_pages}). Skipping."
                )
                continue
            # Convert 1‑based user page to 0‑based index
            page = reader.pages[page_num - 1]
            text = page.extract_text()
            if text:
                full_text.append(text)
        print("\n".join(full_text))
        return "\n".join(full_text)

    @staticmethod
    def _parse_page_spec(spec: str) -> list[int]:
        """Convert e.g. '45', '50,100', '10-15,20' into list of ints."""
        pages = set()
        for part in spec.split(","):
            part = part.strip()
            if "-" in part:
                start, end = part.split("-")
                pages.update(range(int(start), int(end) + 1))
            else:
                pages.add(int(part))
        return sorted(pages)

    @staticmethod
    def _parse_srt(path: Path) -> str:
        timestamp_re = re.compile(
            r"\d{2}:\d{2}:\d{2}[.,]\d{3}\s*-->\s*\d{2}:\d{2}:\d{2}[.,]\d{3}"
        )
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            text = f.read()
        return timestamp_re.sub(" ", text)


class WordProcessor:
    """Convert raw text into sorted word frequency dictionary."""

    # Precompiled regex patterns (class-level, shared)
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
    _POSSESSIVE_RE = re.compile(r"(?i)(?:'s|’s|s'|s’)$")

    def __init__(self, stopwords_path: Path, names_set: Optional[Set[str]] = None):
        self.stopwords = self._load_stopwords(stopwords_path)
        self.names = names_set if names_set is not None else self._load_names()
        self._lemmatizer = WordNetLemmatizer()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_stopwords(path: Path) -> Set[str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return set(f.read().split())
        except FileNotFoundError:
            logger.warning(f"Stopwords file not found: {path}")
            return set()

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_names() -> Set[str]:
        return {name.lower() for name in names.words()}

    def _lemmatize_noun(self, word: str) -> str:
        return self._lemmatizer.lemmatize(word, pos="n")

    def _lemmatize_verb(self, word: str) -> str:
        return self._lemmatizer.lemmatize(word, pos="v")

    def process(self, raw_text: str) -> dict[str, int]:
        # Remove HTML
        text = self._HTML_TAG_RE.sub(" ", raw_text)

        # Tokenise
        raw_words = self._WORD_RE.findall(text)
        counts = Counter()

        for w in raw_words:
            word = w.lower()
            if word in self.stopwords or word in self.names:
                continue
            word = self._POSSESSIVE_RE.sub("", word)
            word = self._lemmatize_noun(word)
            word = self._lemmatize_verb(word)
            if word not in self.stopwords and len(word) > 1:
                counts[word] += 1

        return dict(counts.most_common())


class CEFRHelper:
    """Load CEFR CSV (local or URL) and provide level queries."""

    _LEVEL_RANKS = {"A1": 1, "A2": 2, "B1": 3, "B2": 4, "C1": 5, "C2": 6}

    def __init__(self, source: Path):
        self._mapping = self._load(source)

    @classmethod
    def _load(cls, source: Path) -> dict[str, str]:
        source_str = str(source)
        if source_str.startswith(("http://", "https://")):
            # Convert GitHub blob URL to raw
            raw = source_str.replace("github.com", "raw.githubusercontent.com").replace(
                "/blob/", "/"
            )
            with urllib.request.urlopen(raw) as resp:
                reader = csv.reader(line.decode("utf-8") for line in resp)
                return cls._process_rows(reader)
        else:
            with open(source, "r", encoding="utf-8") as f:
                return cls._process_rows(csv.reader(f))

    @classmethod
    def _process_rows(cls, reader) -> dict[str, str]:
        mapping = {}
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 3:
                continue
            word = row[0].strip().lower()
            level = row[2].strip().upper()
            current_rank = cls._LEVEL_RANKS.get(mapping.get(word, ""), 99)
            new_rank = cls._LEVEL_RANKS.get(level, 99)
            if new_rank < current_rank:
                mapping[word] = level
        return mapping

    def get_level(self, word: str) -> Optional[str]:
        return self._mapping.get(word.lower())

    def filter_words(
        self, word_counts: dict[str, int], target_level: str
    ) -> dict[str, int]:
        target = target_level.upper()
        return {w: c for w, c in word_counts.items() if self.get_level(w) == target}


class WordTranslator:
    """Translate English words to target language using Google Translate."""

    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    @staticmethod
    def translate_org(words: list[str], target_lang: str = "fa") -> dict[str, str]:
        if not words:
            return {}
        logger.info(f"Translating {len(words)} words to '{target_lang}'...")
        try:
            translator = GoogleTranslator(source="en", target=target_lang)
            translated = translator.translate_batch(words)
            return dict(zip(words, translated))
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return {}

    @staticmethod
    def translate(words: list[str], target_lang: str = "fa") -> dict[str, str]:
        """
        Translate a list of English words to the target language using batch requests.
        If batch fails, falls back to per‑word requests (slower, but more robust).
        Returns a dictionary {word: translation}.
        """
        if not words:
            return {}

        logger.info(
            f"Translating {len(words)} words to '{target_lang}' via batch Google Translate..."
        )

        result = {}
        for idx, word in enumerate(words, 1):
            translation = WordTranslator._fetch_single_translation(word, target_lang)
            result[word] = translation if translation else "N/A"
            # Polite delay
            if idx % 10 == 0:
                time.sleep(0.5)
            else:
                time.sleep(0.2)
        return result

    @staticmethod
    def _fetch_single_translation(word: str, target_lang: str) -> Optional[str]:
        lang_name = "fa"
        query = f"translate {word} from english to {lang_name}"
        params = {
            "q": query,
        }
        try:
            response = requests.get(
                "https://www.google.com/search",
                params=params,
                headers=WordTranslator.HEADERS,
                timeout=15,
            )
            response.raise_for_status()
            html = response.text
            print(html)
        except Exception as e:
            logger.error(f"Request failed for '{word}': {e}")
            return None


class Exporter:
    """Static methods for exporting word data to various formats."""

    @staticmethod
    def to_csv(data: list[tuple[str, str, int]], path: Path) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["word", "translation", "repetition"])
            writer.writerows(data)
        logger.info(f"CSV saved to {path}")

    @staticmethod
    def to_html(data: list[tuple[str, str, int]], path: Path) -> None:
        """
        Export the data to a simple HTML file with a styled table.
        """
        # Build table rows with proper HTML escaping
        rows_html = []
        for idx, (word, trans, rep) in enumerate(data, start=1):
            escaped_word = html.escape(word)
            escaped_trans = html.escape(trans)
            rows_html.append(
                # todo: fix this
                f"<tr><td lang='en'>{idx}</td><td lang='en'>{escaped_word}</td><td lang='fa'>{escaped_trans}</td><td lang='en'>{rep}</td></tr>"
                # test
                # f"<tr><td lang='en'>{escaped_word}</td><td lang='fa'>کتاب</td><td lang='en'>{rep}</td></tr>"
            )
        rows_str = "\n".join(rows_html)
        template = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Word Frequency Report</title>
    <style>
        /* English text */
        :lang(en) {
            font-family: 'Ubuntu', sans-serif;
        }
        /* Persian text */
        :lang(fa) {
            font-family: 'Vazirmatn', 'Vazir', sans-serif;
        }
        /* Table styling */
        table {
            border-collapse: collapse;
            width: 65%;
            margin: 20px auto;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 10px 2px 10px 2px;
            text-align: center;
        }
        th {
            color:#ffffff;
            background-color:#0003beb3;
            font-family: 'Ubuntu', sans-serif;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        h2 {
            font-family: 'Ubuntu', sans-serif;
        }
    </style>
</head>
<body>
    <center>
    <h2>Word List</h2>
    <table>
        <thead>
            <tr><th>#</th><th>Word</th><th>Translation</th><th>Repetition</th></tr>
        </thead>
        <tbody>
            __ROWS_PLACEHOLDER__
        </tbody>
    </table>
    </center>
</body>
</html>"""

        full_html = template.replace("__ROWS_PLACEHOLDER__", rows_str)

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(full_html)
            logging.info(f"HTML exported successfully to {path}")
        except Exception as e:
            logging.error(f"Failed to writing HTML: {e}")

    def to_pdf(data: list[tuple[str, str, int]], path: Path) -> None:
        tmp_html = path.with_suffix(".html")
        Exporter.to_html(data, tmp_html)
        try:
            pdfkit.from_file(str(tmp_html), str(path))
            logger.info(f"PDF saved to {path}")
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
        finally:
            if tmp_html.exists():
                tmp_html.unlink()


class WordPipeline:
    """Orchestrate the whole extraction → filter → translate → export process."""

    def __init__(self, stopwords_path: Path, cefr_source: Optional[Path] = None):
        self.processor = WordProcessor(stopwords_path)
        self.cefr = CEFRHelper(cefr_source) if cefr_source else None
        self.translator = WordTranslator()

    def run(
        self,
        document: Path,
        target_level: Optional[str] = None,
        target_lang: str = "fa",
        out_csv: Optional[Path] = None,
        out_html: Optional[Path] = None,
        out_pdf: Optional[Path] = None,
        pages: Optional[list[int]] = None,
    ) -> dict[str, int]:
        """
        Extract words, optionally filter by CEFR level, translate and export.
        Returns final word frequency dict.
        """
        if pages:
            page_list = DocumentParser._parse_page_spec(pages)
            if page_list:
                pages = page_list
            else:
                logger.error("Invalid page specification. Ignoring --pages.")

        print(pages)
        # 1. Extract raw text
        raw_text = DocumentParser.extract_text(document, pages)
        logger.info(f"Extracted {len(raw_text)} characters from {document}")

        # 2. Get word frequencies
        word_counts = self.processor.process(raw_text)
        logger.info(f"Found {len(word_counts)} unique words")

        # 3. Filter by CEFR level if requested
        if target_level and self.cefr:
            word_counts = self.cefr.filter_words(word_counts, target_level)
            logger.info(f"Filtered to {len(word_counts)} words at level {target_level}")

        if not word_counts:
            logger.warning("No words to export.")
            return {}

        # 4. Translate
        # todo: fix if have a stable net
        # translations = self.translator.translate(list(word_counts.keys()), target_lang)
        translations = {}

        # 5. Build sorted export data (word, translation, repetition)
        export_data = [
            (w, translations.get(w, "N/A"), c) for w, c in word_counts.items()
        ]
        export_data.sort(key=lambda x: x[2], reverse=True)

        # 6. Export
        if out_csv:
            Exporter.to_csv(export_data, out_csv)
        if out_html:
            Exporter.to_html(export_data, out_html)
        if out_pdf:
            Exporter.to_pdf(export_data, out_pdf)

        return word_counts


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract words from PDF/SRT, translate, export."
    )
    parser.add_argument(
        "-i", "--input", required=True, help="Input file (.pdf or .srt)"
    )
    parser.add_argument(
        "--pages", help="PDF pages to read, e.g. '45', '50,100', '1-5,10'"
    )
    parser.add_argument(
        "-s", "--stopwords", default="stop_words.txt", help="Stopwords file"
    )
    parser.add_argument("-c", "--cefr", help="CEFR CSV file or URL")
    parser.add_argument("-l", "--level", help="Filter by CEFR level (e.g., B1)")
    parser.add_argument(
        "--lang", default="fa", help="Target language code (default: fa)"
    )
    parser.add_argument("--csv", default="export.csv", help="Output CSV file")
    parser.add_argument("--html", default="export.html", help="Output HTML file")
    parser.add_argument("--pdf", default="export.pdf", help="Output PDF file")

    args = parser.parse_args()
    pipeline = WordPipeline(
        stopwords_path=Path(args.stopwords),
        cefr_source=Path(args.cefr) if args.cefr else None,
    )
    pipeline.run(
        document=Path(args.input),
        target_level=args.level,
        target_lang=args.lang,
        out_csv=Path(args.csv),
        out_html=Path(args.html),
        out_pdf=Path(args.pdf),
        pages=args.pages,
    )


if __name__ == "__main__":
    main()
