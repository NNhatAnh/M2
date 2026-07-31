from pathlib import Path
import pandas as pd


class Report:
    def __init__(self):
        self.lines = []

    def title(self, text):
        self.lines.append("=" * 60)
        self.lines.append(text)
        self.lines.append("=" * 60)
        self.lines.append("")

    def add(self, key, value):
        self.lines.append(f"{key:<30}: {value}")

    def blank(self):
        self.lines.append("")

    def save(self, path):
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            for line in self.lines:
                f.write(str(line) + "\n")


def save_summary(summary, path):
    df = pd.DataFrame(summary)
    df.to_csv(path, index=False)
