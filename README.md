# Word Extraction

Extract words from **PDFs** and **SRT** subtitle files, normalize and filter by CEFR level, translate them and export the results to **CSV**, **HTML**, or **PDF**.

## Installation

1. Clone the repository and navigate to the project folder.

```bash
git clone ...
git clone https://github.com/mo1ein/lexisexy.git
cd lexisexy
```

### uv
```bash
uv venv
source .venv/bin/activate  
uv pip install -r requirements.txt
uv run main.py
```

## Usage
Basic command:

```bash
python main.py --input document.pdf --csv output.csv
```

Examples

- Extract all words from a PDF, no translation, only CSV:

```bash
python word_pipeline.py -i report.pdf --csv words.csv
```

- Extract only B1 words from an SRT file, translate to German, and export to PDF:

```bash
python word_pipeline.py -i movie.srt -c cefr.csv -l B1 --lang de --pdf b1_words.pdf
```

- Use a CEFR CSV from a GitHub raw URL:

```bash
python word_pipeline.py -i book.pdf -c https://raw.githubusercontent.com/.../cefr.csv -l A2 --html a2_words.html
```

Stopwords File
Provide a plain text file with one stopword per line (or space-separated). Example:

```text
the
and
to
of
a
```

If the file is missing, the script will warn you and continue with an empty stopword list.
