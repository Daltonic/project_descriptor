import os
import re
import json
from pathlib import Path
from typing import List, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


class TextExtractor:
    def __init__(
        self,
        base_url="http://localhost:11434",
        model="phi4-mini:3.8b",
        temperature=0.3,
        timeout=60,
    ):
        self.chat_model = ChatOllama(
            base_url=base_url, model=model, temperature=temperature, timeout=timeout
        )

    def extract_json_from_markdown(self, response):
        """
        Extracts JSON content from a markdown-formatted response.
        Looks for content within ```json ... ``` blocks.
        """
        # Look for content between ```json and ```
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1)
        # Fallback: try to find any JSON-like array
        match = re.search(r"\[\s*(.*?)\s*\]", response, re.DOTALL)
        if match:
            return match.group(1)
        return None  # Return None if no JSON block is found

    def get_file_content(self, file_path: Path) -> str:
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

    def split_content_into_blocks(self, content: str) -> List[str]:
        """
        Split the content into blocks based on date patterns like "MonthDay time".
        This helps process large files in chunks.
        """
        # Pattern for dates: e.g., Mar4 21:12, Nov5 20:31, May3 19:55
        date_pattern = r"^[A-Za-z]{3}\d{1,2}\s+\d{1,2}:\d{2}$"
        blocks = []
        current_block = []
        lines = content.splitlines()
        for line in lines:
            stripped = line.strip()
            if re.match(date_pattern, stripped):
                if current_block:
                    blocks.append("\n".join(current_block))
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append("\n".join(current_block))
        return [block.strip() for block in blocks if block.strip()]

    def infer_year_for_block(self, block: str, current_year: int) -> str:
        """
        Use LLM to infer year for a single block if needed.
        """
        # Quick check if year is already mentioned
        if re.search(r"\b(20)\d{2}\b", block):
            return block  # Year clues present, no need for LLM

        prompt_template = ChatPromptTemplate.from_template(
            """Infer the year for this note block based on context clues. If no clues, use {current_year}.

Block: {block}

Respond with the full block text, but replace the date line with one that includes the inferred year in YYYY-MM-DD HH:MM format.

Example input date: "Mar4 21:12"
Example output: "2020-03-04 21:12" followed by the rest.

Respond **only** with the modified block text, no JSON or extra text."""
        )
        try:
            llm_chain = prompt_template | self.chat_model | StrOutputParser()
            modified_block = llm_chain.invoke(
                {"block": block, "current_year": current_year}
            ).strip()
            return modified_block
        except Exception:
            # Fallback: append current year
            date_match = re.search(
                r"^([A-Za-z]{3})(\d{1,2})\s+(\d{1,2}:\d{2})", block, re.MULTILINE
            )
            if date_match:
                month_str, day_str, time_str = date_match.groups()
                month_map = {
                    "Jan": "01",
                    "Feb": "02",
                    "Mar": "03",
                    "Apr": "04",
                    "May": "05",
                    "Jun": "06",
                    "Jul": "07",
                    "Aug": "08",
                    "Sep": "09",
                    "Oct": "10",
                    "Nov": "11",
                    "Dec": "12",
                }
                month_num = month_map.get(month_str[:3], "01")
                year = str(current_year)
                new_date_line = f"{year}-{month_num}-{day_str.zfill(2)} {time_str}"
                return re.sub(
                    r"^[A-Za-z]{3}\d{1,2}\s+\d{1,2}:\d{2}",
                    new_date_line,
                    block,
                    flags=re.MULTILINE,
                )
            return block

    def create_extraction_prompt(
        self, block: str, current_date: str = "2025-10-21"
    ) -> str:
        prompt_data = f"""
TEXT EXTRACTION REQUEST FOR BLOCK

Current date for reference: {current_date}

BLOCK CONTENT:
{chr(10)}{block}

Parse this single block into a JSON array (likely 1 object) with:
- "created_date": ISO format string (YYYY-MM-DD HH:MM) from the date line.
- "title": First meaningful line after date as title.
- "description": Full body, preserve \n for breaks.

Ignore folders like "-Work (date)". Only extract notes.

Respond **only** with valid JSON array, e.g.:
[
  {{
    "created_date": "2025-03-04 21:12",
    "title": "Welcome to use Notebook!",
    "description": "Notebook is...\\n\\nWhether..."
  }}
]
"""
        return prompt_data

    def extract_notes_from_block(
        self, block: str, current_date: str = "2025-10-21"
    ) -> List[Dict]:
        # First, infer year if needed
        current_year = int(current_date[:4])
        block_with_year = self.infer_year_for_block(block, current_year)

        prompt_template = ChatPromptTemplate.from_template("{extraction_prompt}")
        try:
            llm_chain = prompt_template | self.chat_model | StrOutputParser()
            response = llm_chain.invoke(
                {
                    "extraction_prompt": self.create_extraction_prompt(
                        block_with_year, current_date
                    )
                }
            )

            # Extract JSON content from the response
            response_content = self.extract_json_from_markdown(response)
            if response_content:
                notes = json.loads(response_content)
                if isinstance(notes, list):
                    return notes
                else:
                    raise ValueError("LLM did not return a list of notes.")
            else:
                raise ValueError(f"No valid JSON found in LLM response: {response}")
        except json.JSONDecodeError as e:
            raise ValueError(
                f"JSON Parsing Error: {e}\nFailed Response: '{response_content}'"
            )
        except Exception as e:
            raise ValueError(f"Error extracting notes from block: {str(e)}")

    def extract_notes(
        self, content: str, current_date: str = "2025-10-21"
    ) -> List[Dict]:
        blocks = self.split_content_into_blocks(content)
        all_notes = []
        for i, block in enumerate(blocks):
            print(f"Processing block {i+1}/{len(blocks)}...")
            try:
                block_notes = self.extract_notes_from_block(block, current_date)
                all_notes.extend(block_notes)
            except Exception as e:
                print(f"Warning: Skipping block {i+1} due to error: {e}")
                continue
        return all_notes

    def save_to_json(self, notes: List[Dict], output_file: str):
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(notes, f, indent=2, ensure_ascii=False)
        print(f"Extracted notes saved to: {output_file}")

    def process_txt_file(
        self,
        txt_path: str,
        output_file: str = None,
        current_date: str = "2025-10-21",
    ) -> List[Dict]:
        print(f"Processing txt file: {txt_path}")
        file_path = Path(txt_path).resolve()
        if not file_path.exists() or not file_path.is_file():
            raise FileNotFoundError(f"TXT file does not exist: {txt_path}")

        content = self.get_file_content(file_path)
        print(f"Read {len(content)} characters from file.")

        notes = self.extract_notes(content, current_date)
        print(f"Extracted {len(notes)} notes.")

        if output_file:
            self.save_to_json(notes, output_file)

        return notes


def main():
    extractor = TextExtractor(timeout=120)
    txt_path = input("Enter the path to the txt file: ").strip()

    if not os.path.isfile(txt_path):
        print(f"Invalid path: {txt_path}")
        return

    try:
        output_file = input(
            "Enter output JSON file path (default: notes.json): "
        ).strip()
        if not output_file:
            output_file = "notes.json"

        results = extractor.process_txt_file(
            txt_path=txt_path,
            output_file=output_file,
            current_date="2025-10-21",
        )

        # Print a preview
        print("\n" + "=" * 50)
        print("EXTRACTION COMPLETE")
        print("=" * 50)
        for note in results[:3]:  # Preview first 3
            print(f"Date: {note['created_date']}")
            print(f"Title: {note['title']}")
            print(f"Description preview: {note['description'][:100]}...")
            print("-" * 30)

        if len(results) > 3:
            print(f"... and {len(results) - 3} more notes.")

    except Exception as e:
        print(f"Error processing txt file: {e}")


if __name__ == "__main__":
    main()
