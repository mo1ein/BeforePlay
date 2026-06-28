import csv
import asyncio
import json
import html
import logging
import re
import threading
import urllib.request
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Optional, Set

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
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
    _POSSESSIVE_RE = re.compile(r"(?i)(?:'s|'s|s'|s')$")

    _stopword_cache: dict[str, Set[str]] = {}
    _names_cache: Optional[Set[str]] = None

    def __init__(self, stopwords_path: Path, names_set: Optional[Set[str]] = None):
        self.stopwords = self._load_stopwords(stopwords_path)
        self.names = names_set if names_set is not None else self._load_names()
        self._lemmatizer = WordNetLemmatizer()

    @staticmethod
    def _load_stopwords(path: Path) -> Set[str]:
        key = str(path)
        if key not in WordProcessor._stopword_cache:
            try:
                WordProcessor._stopword_cache[key] = set(
                    Path(path).read_text(encoding="utf-8").split()
                )
            except FileNotFoundError:
                logger.warning("Stopwords file not found: %s", path)
                WordProcessor._stopword_cache[key] = set()
        return WordProcessor._stopword_cache[key]

    @staticmethod
    def _load_names() -> Set[str]:
        if WordProcessor._names_cache is None:
            WordProcessor._names_cache = {name.lower() for name in names.words()}
        return WordProcessor._names_cache

    @lru_cache(maxsize=50000)
    def _lemmatize(self, word: str) -> str:
        noun = self._lemmatizer.lemmatize(word, pos="n")
        if noun != word:
            return noun
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
            word = self._lemmatize(word)
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


def _create_progress(label: str, total: int) -> Progress:
    """Create a Rich progress bar with standard layout."""
    return Progress(
        SpinnerColumn(),
        TextColumn(f"[bold blue]{label}[/bold blue] [cyan]{{task.description}}[/cyan]"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("[green]{task.completed}/{task.total} words[/green]"),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


class WordTranslator:
    """Translate English words to target language using Google Translate."""

    _CACHE_PATH = Path(".translation_cache.json")
    _CONCURRENT_LIMIT = 10
    _DELAY_BETWEEN = 0.05

    def __init__(self, target_lang: str = "fa"):
        self.target_lang = target_lang
        self._translator = GoogleTranslator(source="en", target=target_lang)

    @staticmethod
    def _load_cache() -> dict[str, str]:
        if WordTranslator._CACHE_PATH.exists():
            return json.loads(WordTranslator._CACHE_PATH.read_text())
        return {}

    @staticmethod
    def _save_cache(cache: dict[str, str]) -> None:

        tmp = WordTranslator._CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cache, ensure_ascii=False))
        tmp.rename(WordTranslator._CACHE_PATH)

    async def _translate_word(
        self,
        word: str,
        semaphore: asyncio.Semaphore,
        progress: Progress,
        task_id,
    ) -> tuple[str, str]:
        async with semaphore:
            loop = asyncio.get_event_loop()
            try:
                result = await loop.run_in_executor(
                    None, self._translator.translate, word
                )
                await asyncio.sleep(WordTranslator._DELAY_BETWEEN)
                progress.advance(task_id)
                return word, result or "N/A"
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.warning("Failed '%s': %s", word, e)
                progress.advance(task_id)
                return word, "N/A"
            except Exception as e:
                logger.warning("Unexpected error for '%s': %s", word, e)
                progress.advance(task_id)
                return word, "N/A"

    def translate(self, words: list[str]) -> dict[str, str]:
        if not words:
            return {}

        cache = self._load_cache()
        to_translate = [w for w in words if w not in cache]

        logger.info(
            "%d cached, %d new words to translate...", len(cache), len(to_translate)
        )

        if not to_translate:
            return {w: cache[w] for w in words}

        async def run_all():
            semaphore = asyncio.Semaphore(self._CONCURRENT_LIMIT)

            with _create_progress(
                f"→ {self.target_lang}", len(to_translate)
            ) as progress:
                task_id = progress.add_task(
                    f"→ {self.target_lang}", total=len(to_translate)
                )

                tasks = [
                    self._translate_word(w, semaphore, progress, task_id)
                    for w in to_translate
                ]

                results = []
                for i in range(0, len(tasks), 50):
                    chunk = await asyncio.gather(*tasks[i : i + 50])
                    results.extend(chunk)
                    cache.update(dict(chunk))

            return results

        asyncio.run(run_all())
        self._save_cache(cache)
        return {w: cache[w] for w in words}


class Exporter:
    """Static methods for exporting word data to various formats."""

    @staticmethod
    def to_csv(
        data: list[tuple[str, str, int]], path: Path, media_urls: Optional[dict] = None
    ) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if media_urls:
                writer.writerow(
                    ["word", "translation", "repetition", "pronunciation", "image"]
                )
                for word, trans, rep in data:
                    audio, img = media_urls.get(word, (None, None))
                    writer.writerow([word, trans, rep, audio or "", img or ""])
            else:
                writer.writerow(["word", "translation", "repetition"])
                writer.writerows(data)
        logger.info("CSV saved to %s", path)

    @staticmethod
    def to_html(
        data: list[tuple[str, str, int]], path: Path, media_urls: Optional[dict] = None
    ) -> None:
        """
        Export the data to a simple HTML file with a styled table.
        """
        has_media = bool(media_urls)
        # Build table rows with proper HTML escaping
        rows_html = []
        for idx, (word, trans, rep) in enumerate(data, start=1):
            escaped_word = html.escape(word)
            escaped_trans = html.escape(trans)
            if has_media:
                audio_url, img_url = media_urls.get(word, (None, None))
                audio_cell = (
                    f"<button onclick=\"var a=new Audio('{audio_url}');var s=this;"
                    f"s.classList.add('playing');a.onended=function(){{s.classList.remove('playing')}};"
                    f'a.play()" '
                    f"style='background:none;border:none;cursor:pointer;padding:4px;color:#4285f4' title='Play pronunciation'>"
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='20' height='20' viewBox='0 0 24 24' fill='none' "
                    f"stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>"
                    f"<polygon points='11 5 6 9 2 9 2 15 6 15 11 19 11 5'></polygon>"
                    f"<path class='wave1' d='M15.54 8.46a5 5 0 0 1 0 7.07'></path>"
                    f"<path class='wave2' d='M19.07 4.93a10 10 0 0 1 0 14.14'></path></svg></button>"
                    if audio_url
                    else ""
                )
                img_cell = (
                    f"<img src='{img_url}' style='max-width:100px;max-height:100px;border-radius:4px' loading='lazy'>"
                    if img_url
                    else ""
                )
                rows_html.append(
                    f"<tr><td lang='en'>{idx}</td><td lang='en'>{escaped_word}</td>"
                    f"<td lang='fa'>{escaped_trans}</td><td lang='en'>{rep}</td>"
                    f"<td>{audio_cell}</td><td>{img_cell}</td></tr>"
                )
            else:
                rows_html.append(
                    f"<tr><td lang='en'>{idx}</td><td lang='en'>{escaped_word}</td><td lang='fa'>{escaped_trans}</td><td lang='en'>{rep}</td></tr>"
                )
        rows_str = "\n".join(rows_html)

        header_row = (
            "<tr><th>#</th><th>Word</th><th>Translation</th><th>Repetition</th>"
            + ("<th>Pronunciation</th><th>Image</th>" if has_media else "")
            + "</tr>"
        )

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
        img {
            border-radius: 4px;
        }
        .playing .wave1 {
            animation: soundWave 0.6s ease-in-out infinite alternate;
        }
        .playing .wave2 {
            animation: soundWave 0.6s ease-in-out 0.15s infinite alternate;
        }
        @keyframes soundWave {
            from { opacity: 0.3; }
            to { opacity: 1; }
        }
    </style>
</head>
<body>
    <center>
    <h2>Word List</h2>
    <table>
        <thead>
            __HEADER_ROW__
        </thead>
        <tbody>
            __ROWS_PLACEHOLDER__
        </tbody>
    </table>
    </center>
</body>
</html>"""

        full_html = template.replace("__ROWS_PLACEHOLDER__", rows_str).replace(
            "__HEADER_ROW__", header_row
        )

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(full_html)
            logging.info("HTML exported successfully to %s", path)
        except OSError as e:
            logging.error("Failed to write HTML: %s", e)

    def to_pdf(
        data: list[tuple[str, str, int]], path: Path, media_urls: Optional[dict] = None
    ) -> None:
        tmp_html = path.with_suffix(".html")
        Exporter.to_html(data, tmp_html, media_urls=media_urls)
        try:
            pdfkit.from_file(str(tmp_html), str(path))
            logger.info("PDF saved to %s", path)
        except (OSError, pdfkit.PDFKitError) as e:
            logger.error("PDF conversion failed: %s", e)
        finally:
            if tmp_html.exists():
                tmp_html.unlink()


class MediaCache:
    """Disk-based cache for media URLs and downloaded files."""

    _DIR = Path(".media_cache")
    _AUDIO_DIR = _DIR / "audio"
    _IMAGE_DIR = _DIR / "images"
    _INDEX = _DIR / "index.json"

    def __init__(self):
        self._DIR.mkdir(exist_ok=True)
        self._AUDIO_DIR.mkdir(exist_ok=True)
        self._IMAGE_DIR.mkdir(exist_ok=True)
        self._index: dict[str, dict] = {}
        if self._INDEX.exists():
            self._index = json.loads(self._INDEX.read_text())

    def _save_index(self) -> None:
        self._INDEX.write_text(json.dumps(self._index, ensure_ascii=False))

    def get_urls(self, word: str) -> tuple[Optional[str], Optional[str]]:
        entry = self._index.get(word)
        if not entry:
            return None, None
        return entry.get("audio_url"), entry.get("image_url")

    def set_urls(
        self, word: str, audio_url: Optional[str], image_url: Optional[str]
    ) -> None:
        if word not in self._index:
            self._index[word] = {}
        if audio_url:
            self._index[word]["audio_url"] = audio_url
        if image_url:
            self._index[word]["image_url"] = image_url

    def get_data(self, word: str) -> tuple[Optional[bytes], Optional[bytes]]:
        entry = self._index.get(word, {})
        audio_data = None
        image_data = None
        audio_path = self._AUDIO_DIR / f"{word}.mp3"
        image_path = self._IMAGE_DIR / f"{word}.png"
        if audio_path.exists():
            audio_data = audio_path.read_bytes()
        elif entry.get("audio_url"):
            audio_data = self._download(entry["audio_url"], audio_path)
        if image_path.exists():
            image_data = image_path.read_bytes()
        elif entry.get("image_url"):
            image_data = self._download(entry["image_url"], image_path)
        return audio_data, image_data

    @staticmethod
    def _download(url: str, dest: Path) -> Optional[bytes]:
        try:
            resp = requests.get(
                url, timeout=10, headers={"User-Agent": "BeforePlay/1.0"}
            )
            if resp.status_code == 200:
                dest.write_bytes(resp.content)
                return resp.content
        except requests.Timeout:
            logger.debug("Download timed out: %s", url)
        except requests.ConnectionError:
            logger.debug("Download connection error: %s", url)
        return None

    def save(self) -> None:
        self._save_index()


class WiktionaryMedia:
    """Fetch pronunciation audio and image for words (async, concurrent, with fallbacks)."""

    _CONCURRENCY = 10
    _GOOGLE_TTS_URL = (
        "https://translate.google.com/translate_tts?ie=UTF-8&tl=en&client=tw-ob&q={}"
    )
    _HEADERS = {"User-Agent": "BeforePlay/1.0 (vocab tool; contact: github)"}
    _local = threading.local()

    @classmethod
    def _get_session(cls) -> requests.Session:
        if not hasattr(cls._local, "session"):
            cls._local.session = requests.Session()
            cls._local.session.headers.update(cls._HEADERS)
        return cls._local.session

    @classmethod
    def _batch_image_urls(cls, words: list[str]) -> dict[str, Optional[str]]:
        """Fetch image URLs for multiple words via pageimages + Wikipedia fallback."""
        results: dict[str, Optional[str]] = {w: None for w in words}
        session = cls._get_session()
        BATCH = 50

        # 1) Wiktionary pageimages batch
        for i in range(0, len(words), BATCH):
            batch = words[i : i + BATCH]
            try:
                resp = session.get(
                    "https://en.wiktionary.org/w/api.php",
                    params={
                        "action": "query",
                        "titles": "|".join(batch),
                        "prop": "pageimages",
                        "format": "json",
                        "pithumbsize": 300,
                    },
                    timeout=15,
                )
                if resp.status_code == 200:
                    pages = resp.json().get("query", {}).get("pages", {})
                    for p in pages.values():
                        title = p.get("title", "").lower()
                        thumb = p.get("thumbnail", {}).get("source")
                        if thumb:
                            results[title] = thumb
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.debug("Batch image fetch failed: %s", e)

        # 2) Wikipedia REST fallback for words without images
        missing = [w for w in words if not results.get(w)]
        for i in range(0, len(missing), BATCH):
            batch = missing[i : i + BATCH]
            for word in batch:
                try:
                    resp = session.get(
                        f"https://en.wikipedia.org/api/rest_v1/page/summary/{word}",
                        timeout=8,
                    )
                    if resp.status_code == 200:
                        thumb = resp.json().get("thumbnail", {}).get("source")
                        if thumb:
                            results[word] = thumb
                except (requests.Timeout, requests.ConnectionError):
                    pass

        return results

    @classmethod
    def _find_audio_url(cls, word: str, session: requests.Session) -> Optional[str]:
        """Find pronunciation audio URL — Wikimedia first, then Google TTS."""
        try:
            resp = session.get(
                f"https://en.wiktionary.org/api/rest_v1/page/media-list/{word}",
                timeout=8,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                audio_titles = [i["title"] for i in items if i.get("type") == "audio"]
                if audio_titles:
                    resp2 = session.get(
                        "https://commons.wikimedia.org/w/api.php",
                        params={
                            "action": "query",
                            "titles": audio_titles[0],
                            "prop": "imageinfo",
                            "iiprop": "url",
                            "format": "json",
                        },
                        timeout=8,
                    )
                    if resp2.status_code == 200:
                        pages = resp2.json().get("query", {}).get("pages", {})
                        for p in pages.values():
                            url = p.get("imageinfo", [{}])[0].get("url")
                            if url:
                                return url
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.debug("Wikimedia audio failed for '%s': %s", word, e)

        # Google TTS fallback
        try:
            tts_url = cls._GOOGLE_TTS_URL.format(word.replace(" ", "+"))
            resp = session.get(tts_url, timeout=8)
            if resp.status_code == 200:
                return tts_url
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.debug("Google TTS failed for '%s': %s", word, e)

        return None

    @classmethod
    def _find_image_url(cls, word: str, session: requests.Session) -> Optional[str]:
        """Find image URL — pageimages first, then Wikimedia Commons search."""
        try:
            resp = session.get(
                "https://en.wiktionary.org/w/api.php",
                params={
                    "action": "query",
                    "titles": word,
                    "prop": "pageimages",
                    "format": "json",
                    "pithumbsize": 300,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    thumb = p.get("thumbnail", {}).get("source")
                    if thumb:
                        return thumb
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.debug("Pageimages failed for '%s': %s", word, e)

        # Commons search fallback
        try:
            resp = session.get(
                "https://commons.wikimedia.org/w/api.php",
                params={
                    "action": "query",
                    "generator": "search",
                    "gsrsearch": f"{word} illustration",
                    "gsrnamespace": "6",
                    "prop": "imageinfo",
                    "iiprop": "url|thumburl",
                    "iiurlwidth": "300",
                    "format": "json",
                    "gsrlimit": "1",
                },
                timeout=8,
            )
            if resp.status_code == 200:
                pages = resp.json().get("query", {}).get("pages", {})
                for p in pages.values():
                    thumb = p.get("imageinfo", [{}])[0].get("thumburl")
                    if thumb:
                        return thumb
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.debug("Commons search failed for '%s': %s", word, e)

        # Wikipedia REST API fallback
        try:
            resp = session.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{word}",
                timeout=8,
            )
            if resp.status_code == 200:
                thumb = resp.json().get("thumbnail", {}).get("source")
                if thumb:
                    return thumb
        except (requests.Timeout, requests.ConnectionError) as e:
            logger.debug("Wikipedia image failed for '%s': %s", word, e)

        return None

    @classmethod
    def _fetch_word(
        cls, word: str, image_urls: dict[str, Optional[str]], cache: MediaCache
    ) -> tuple[str, Optional[bytes], Optional[bytes]]:
        """Fetch audio+image bytes for one word, using cache."""
        cached_audio, cached_image = cache.get_urls(word)
        if cached_audio or cached_image:
            return word, *cache.get_data(word)

        session = cls._get_session()
        audio_url = cls._find_audio_url(word, session)
        image_url = image_urls.get(word) or cls._find_image_url(word, session)
        cache.set_urls(word, audio_url, image_url)

        audio_data = None
        image_data = None
        if audio_url:
            try:
                resp = session.get(audio_url, timeout=10)
                if resp.status_code == 200:
                    audio_data = resp.content
                    (MediaCache._AUDIO_DIR / f"{word}.mp3").write_bytes(resp.content)
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.debug("Audio download failed for '%s': %s", word, e)
        if image_url:
            try:
                resp = session.get(image_url, timeout=10)
                if resp.status_code == 200:
                    image_data = resp.content
                    (MediaCache._IMAGE_DIR / f"{word}.png").write_bytes(resp.content)
            except (requests.Timeout, requests.ConnectionError) as e:
                logger.debug("Image download failed for '%s': %s", word, e)
        return word, audio_data, image_data

    @classmethod
    async def _fetch_all(
        cls,
        words: list[str],
        progress: Optional[Progress] = None,
        task_id=None,
    ) -> dict[str, tuple[Optional[bytes], Optional[bytes]]]:
        """Fetch audio+image for all words concurrently via thread pool."""
        cache = MediaCache()
        uncached = [
            w for w in words if not cache.get_urls(w)[0] and not cache.get_urls(w)[1]
        ]

        if progress and task_id is not None:
            for _ in range(len(words) - len(uncached)):
                progress.advance(task_id)

        if not uncached:
            return {w: cache.get_data(w) for w in words}

        image_urls = await asyncio.to_thread(cls._batch_image_urls, uncached)

        semaphore = asyncio.Semaphore(cls._CONCURRENCY)

        async def limited_fetch(
            word: str,
        ) -> tuple[str, Optional[bytes], Optional[bytes]]:
            async with semaphore:
                return await asyncio.to_thread(cls._fetch_word, word, image_urls, cache)

        tasks = [limited_fetch(w) for w in uncached]
        results = {}
        for coro in asyncio.as_completed(tasks):
            try:
                word, audio, image = await coro
                results[word] = (audio, image)
            except Exception as e:
                logger.debug("Media fetch failed: %s", e)
            if progress and task_id is not None:
                progress.advance(task_id)

        cache.save()

        # Merge cached + fetched
        for w in words:
            if w not in results:
                results[w] = cache.get_data(w)
        return results

    @classmethod
    async def _fetch_all_urls(
        cls,
        words: list[str],
        progress: Optional[Progress] = None,
        task_id=None,
    ) -> dict[str, tuple[Optional[str], Optional[str]]]:
        """Fetch audio+image URLs for all words concurrently (no download)."""
        cache = MediaCache()
        uncached = [
            w for w in words if not cache.get_urls(w)[0] and not cache.get_urls(w)[1]
        ]

        if progress and task_id is not None:
            for _ in range(len(words) - len(uncached)):
                progress.advance(task_id)

        if not uncached:
            results = {w: cache.get_urls(w) for w in words}
            return results

        image_urls = await asyncio.to_thread(cls._batch_image_urls, uncached)

        semaphore = asyncio.Semaphore(cls._CONCURRENCY)

        async def limited_fetch(word: str) -> tuple[str, Optional[str], Optional[str]]:
            async with semaphore:
                loop = asyncio.get_event_loop()
                session = cls._get_session()
                audio_url = await loop.run_in_executor(
                    None, cls._find_audio_url, word, session
                )
                image_url = image_urls.get(word)
                cache.set_urls(word, audio_url, image_url)
                return word, audio_url, image_url

        tasks = [limited_fetch(w) for w in uncached]
        results = {}
        for coro in asyncio.as_completed(tasks):
            try:
                word, audio_url, image_url = await coro
                results[word] = (audio_url, image_url)
            except Exception as e:
                logger.debug("Media URL fetch failed: %s", e)
            if progress and task_id is not None:
                progress.advance(task_id)

        cache.save()

        for w in words:
            if w not in results:
                results[w] = cache.get_urls(w)
        return results

    @classmethod
    def fetch_all(
        cls, words: list[str], progress: Optional[Progress] = None, task_id=None
    ) -> dict[str, tuple[Optional[bytes], Optional[bytes]]]:
        """Sync wrapper — fetch audio+image for all words."""
        return asyncio.run(cls._fetch_all(words, progress, task_id))

    @classmethod
    def fetch_all_urls(
        cls, words: list[str], progress: Optional[Progress] = None, task_id=None
    ) -> dict[str, tuple[Optional[str], Optional[str]]]:
        """Sync wrapper — fetch audio+image URLs for all words."""
        return asyncio.run(cls._fetch_all_urls(words, progress, task_id))


class AnkiExporter:
    """Export word list as an Anki .apkg file. Requires `genanki`."""

    _DECK_ID = 1607392319
    _MODEL_ID = 1607392312

    @staticmethod
    def _get_model():
        import genanki

        return genanki.Model(
            AnkiExporter._MODEL_ID,
            "BeforePlay Card",
            fields=[
                {"name": "Word"},
                {"name": "Translation"},
                {"name": "Repetition"},
                {"name": "Audio"},
                {"name": "Picture"},
            ],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": "<div style='font-size:24px;text-align:center'>{{Word}}</div>",
                    "afmt": (
                        "{{FrontSide}}<hr id='answer'>"
                        "<div style='font-size:20px;text-align:center'>{{Translation}}</div>"
                        "<div style='text-align:center;margin-top:10px'>{{Audio}}</div>"
                        "<div style='text-align:center;margin-top:10px'>{{Picture}}</div>"
                    ),
                }
            ],
        )

    @staticmethod
    def to_apkg(
        data: list[tuple[str, str, int]], path: Path, with_media: bool = True
    ) -> None:
        import genanki

        model = AnkiExporter._get_model()
        deck = genanki.Deck(AnkiExporter._DECK_ID, "BeforePlay Words")

        media_data: dict[str, tuple[Optional[bytes], Optional[bytes]]] = {}
        if with_media:
            words = [w for w, _, _ in data]
            with _create_progress("Fetching media", len(words)) as progress:
                task_id = progress.add_task("audio + images", total=len(words))
                media_data = WiktionaryMedia.fetch_all(words, progress, task_id)
            count = sum(1 for a, i in media_data.values() if a or i)
            logger.info("Media fetched for %d/%d words", count, len(words))

        media_files = []
        for word, trans, rep in data:
            audio_field = ""
            picture_field = ""

            if with_media and word in media_data:
                audio_data, img_data = media_data[word]
                if audio_data:
                    filename = f"{word}.mp3"
                    audio_field = f"[sound:{filename}]"
                    media_files.append((filename, audio_data))
                if img_data:
                    filename = f"{word}.png"
                    picture_field = f'<img src="{filename}" style="max-width:200px">'
                    media_files.append((filename, img_data))

            note = genanki.Note(
                model=model,
                fields=[word, trans, str(rep), audio_field, picture_field],
            )
            deck.add_note(note)

        package = genanki.Package(deck)

        import tempfile
        import os

        tmpdir = tempfile.mkdtemp()
        temp_paths = []
        for filename, file_data in media_files:
            tmp_path = os.path.join(tmpdir, filename)
            with open(tmp_path, "wb") as f:
                f.write(file_data)
            temp_paths.append(tmp_path)

        package.media_files = temp_paths
        package.write_to_file(str(path))

        for p in temp_paths:
            os.unlink(p)
        os.rmdir(tmpdir)

        if with_media:
            logger.info(
                f"Anki deck saved to {path} ({len(media_files)} media files attached)"
            )
        else:
            logger.info("Anki deck saved to %s", path)


class WordPipeline:
    """Orchestrate the whole extraction → filter → translate → export process."""

    def __init__(self, stopwords_path: Path, cefr_source: Optional[Path] = None):
        self.processor = WordProcessor(stopwords_path)
        self.cefr = CEFRHelper(cefr_source) if cefr_source else None

    def run(
        self,
        document: Path,
        target_level: Optional[str] = None,
        target_lang: str = "fa",
        out_csv: Optional[Path] = None,
        out_html: Optional[Path] = None,
        out_pdf: Optional[Path] = None,
        out_anki: Optional[Path] = None,
        pages: Optional[list[int]] = None,
        with_media: bool = True,
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

        # 1. Extract raw text
        raw_text = DocumentParser.extract_text(document, pages)
        logger.info("Extracted %d characters from %s", len(raw_text), document)

        # 2. Get word frequencies
        word_counts = self.processor.process(raw_text)
        logger.info("Found %d unique words", len(word_counts))

        # 3. Filter by CEFR level if requested
        if target_level and self.cefr:
            word_counts = self.cefr.filter_words(word_counts, target_level)
            logger.info(
                "Filtered to %d words at level %s", len(word_counts), target_level
            )

        if not word_counts:
            logger.warning("No words to export.")
            return {}

        # 4. Translate
        translator = WordTranslator(target_lang)
        translations = translator.translate(list(word_counts.keys()))

        # 5. Build sorted export data (word, translation, repetition)
        export_data = [
            (w, translations.get(w, "N/A"), c) for w, c in word_counts.items()
        ]
        export_data.sort(key=lambda x: x[2], reverse=True)

        # 6. Fetch media URLs for csv/html/pdf
        media_urls = None
        if with_media and (out_csv or out_html or out_pdf):
            with _create_progress("Fetching media", len(word_counts)) as progress:
                task_id = progress.add_task("audio + images", total=len(word_counts))
                media_urls = WiktionaryMedia.fetch_all_urls(
                    list(word_counts.keys()), progress, task_id
                )

        # 7. Export
        if out_csv:
            Exporter.to_csv(export_data, out_csv, media_urls=media_urls)
        if out_html:
            Exporter.to_html(export_data, out_html, media_urls=media_urls)
        if out_pdf:
            Exporter.to_pdf(export_data, out_pdf, media_urls=media_urls)
        if out_anki:
            AnkiExporter.to_apkg(export_data, out_anki, with_media=with_media)

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
    parser.add_argument("--csv", nargs="?", const="export.csv", help="Output CSV file")
    parser.add_argument(
        "--html", nargs="?", const="export.html", help="Output HTML file"
    )
    parser.add_argument("--pdf", nargs="?", const="export.pdf", help="Output PDF file")
    parser.add_argument(
        "--anki", nargs="?", const="export.apkg", help="Output Anki deck (.apkg)"
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Skip downloading pronunciation/images for Anki cards",
    )

    args = parser.parse_args()
    logger.debug("Args: %s", args)
    pipeline = WordPipeline(
        stopwords_path=Path(args.stopwords),
        cefr_source=Path(args.cefr) if args.cefr else None,
    )
    pipeline.run(
        document=Path(args.input),
        target_level=args.level,
        target_lang=args.lang,
        out_csv=Path(args.csv) if args.csv else None,
        out_html=Path(args.html) if args.html else None,
        out_pdf=Path(args.pdf) if args.pdf else None,
        out_anki=Path(args.anki) if args.anki else None,
        pages=args.pages,
        with_media=not args.no_media,
    )


if __name__ == "__main__":
    main()
