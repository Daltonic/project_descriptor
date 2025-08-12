import os
import re
import json
from pathlib import Path
from typing import List, Dict, Set
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


class ProjectAnalyzer:
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
        self.default_ignore = {
            "__pycache__",
            ".git",
            ".gitignore",
            ".DS_Store",
            "node_modules",
            ".env",
            ".venv",
            "venv",
            "env",
            ".pytest_cache",
            ".mypy_cache",
            "dist",
            "build",
            ".idea",
            ".vscode",
            "*.pyc",
            "*.pyo",
            "*.pyd",
            ".coverage",
            "htmlcov",
            ".tox",
            ".cache",
            "eggs",
            "*.egg-info",
            "logs",
            "*.log",
            ".npm",
            ".yarn",
            "package-lock.json",
            "yarn.lock",
        }
        self.code_extensions = {
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".cs",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".swift",
            ".kt",
            ".scala",
            ".html",
            ".css",
            ".scss",
            ".sass",
            ".less",
            ".vue",
            ".svelte",
            ".sql",
            ".sh",
            ".bat",
            ".ps1",
            ".yml",
            ".yaml",
            ".json",
            ".xml",
            ".md",
            ".rst",
            ".txt",
            ".dockerfile",
            ".config",
            ".conf",
        }

    def should_ignore(self, path: Path, ignore_patterns: Set[str]) -> bool:
        name = path.name.lower()
        if name in ignore_patterns:
            return True
        for pattern in ignore_patterns:
            if pattern.startswith("*") and name.endswith(pattern[1:]):
                return True
            if pattern.endswith("*") and name.startswith(pattern[:-1]):
                return True
        return False

    def extract_json_from_markdown(self, response):
        """
        Extracts JSON content from a markdown-formatted response.
        Looks for content within ```json ... ``` blocks.
        """
        # Look for content between ```json and ```
        match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if match:
            return match.group(1)
        return None  # Return None if no JSON block is found

    def get_file_info(self, file_path: Path) -> Dict:
        try:
            stat = file_path.stat()
            file_info = {
                "name": file_path.name,
                "path": str(file_path),
                "size": stat.st_size,  # Fixed the typo here
                "extension": file_path.suffix.lower(),
                "is_code": file_path.suffix.lower() in self.code_extensions,
                "content_preview": None,
            }
            if file_info["is_code"] and stat.st_size < 50000:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        file_info["content_preview"] = (
                            content[:500] + "..." if len(content) > 500 else content
                        )
                        file_info["line_count"] = len(content.splitlines())
                except Exception:
                    pass
            return file_info
        except Exception as e:
            return {"name": file_path.name, "path": str(file_path), "error": str(e)}

    def analyze_directory(
        self, root_path: str, ignore_folders: List[str] = None
    ) -> Dict:
        root = Path(root_path).resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(
                f"Project path does not exist or is not a directory: {root_path}"
            )

        ignore_patterns = self.default_ignore.copy()
        if ignore_folders:
            ignore_patterns.update(ignore_folders)

        project_info = {
            "project_name": root.name,
            "root_path": str(root),
            "structure": {},
            "files": [],
            "directories": [],
            "statistics": {
                "total_files": 0,
                "total_directories": 0,
                "code_files": 0,
                "total_size": 0,
                "file_types": {},
                "languages": set(),
            },
        }

        def scan_directory(current_path: Path, relative_path: str = ""):
            items = []
            try:
                for item in sorted(current_path.iterdir()):
                    if self.should_ignore(item, ignore_patterns):
                        continue

                    item_relative = (
                        os.path.join(relative_path, item.name)
                        if relative_path
                        else item.name
                    )

                    if item.is_dir():
                        project_info["statistics"]["total_directories"] += 1
                        project_info["directories"].append(item_relative)
                        sub_items = scan_directory(item, item_relative)
                        items.append(
                            {
                                "name": item.name,
                                "type": "directory",
                                "path": item_relative,
                                "items": sub_items,
                            }
                        )
                    else:
                        project_info["statistics"]["total_files"] += 1
                        if project_info["statistics"]["total_files"] > 500:
                            raise ValueError(
                                "Project contains more than 500 files. Aborting analysis."
                            )

                        file_info = self.get_file_info(item)
                        file_info["relative_path"] = item_relative
                        project_info["files"].append(file_info)
                        project_info["statistics"]["total_size"] += file_info.get(
                            "size", 0
                        )

                        ext = file_info.get("extension", "")
                        if ext:
                            project_info["statistics"]["file_types"][ext] = (
                                project_info["statistics"]["file_types"].get(ext, 0) + 1
                            )
                        if file_info.get("is_code"):
                            project_info["statistics"]["code_files"] += 1
                            lang_map = {
                                ".py": "Python",
                                ".js": "JavaScript",
                                ".ts": "TypeScript",
                                ".tsx": "TypeScript",
                                ".jsx": "JavaScript",
                                ".java": "Java",
                                ".cpp": "C++",
                                ".c": "C",
                                ".cs": "C#",
                                ".php": "PHP",
                                ".rb": "Ruby",
                                ".go": "Go",
                                ".rs": "Rust",
                                ".swift": "Swift",
                                ".kt": "Kotlin",
                                ".scala": "Scala",
                                ".html": "HTML",
                                ".css": "CSS",
                                ".scss": "SCSS",
                                ".sql": "SQL",
                            }
                            if ext in lang_map:
                                project_info["statistics"]["languages"].add(
                                    lang_map[ext]
                                )

                        items.append(
                            {
                                "name": item.name,
                                "type": "file",
                                "path": item_relative,
                                "size": file_info.get("size", 0),
                                "extension": ext,
                            }
                        )
            except PermissionError:
                pass
            return items

        project_info["structure"] = scan_directory(root)
        project_info["statistics"]["languages"] = list(
            project_info["statistics"]["languages"]
        )
        return project_info

    def generate_tree_view(
        self, structure: List[Dict], prefix: str = "", is_last: bool = True
    ) -> str:
        tree = ""
        for i, item in enumerate(structure):
            is_last_item = i == len(structure) - 1
            current_prefix = "└─ " if is_last_item else "├─ "
            tree += f"{prefix}{current_prefix}{item['name']}\n"
            if item["type"] == "directory" and "items" in item:
                extension = "   " if is_last_item else "│  "
                tree += self.generate_tree_view(
                    item["items"], prefix + extension, is_last_item
                )
        return tree

    def create_summary_prompt(self, project_info: Dict) -> str:
        file_summary = []
        for file_info in project_info["files"][:50]:
            if file_info.get("is_code") and file_info.get("content_preview"):
                file_summary.append(
                    f"File: {file_info['relative_path']}\n"
                    f"Language: {file_info.get('extension', 'unknown')}\n"
                    f"Preview: {file_info['content_preview'][:200]}...\n"
                )
        stats = project_info["statistics"]
        prompt_data = f"""
PROJECT ANALYSIS REQUEST

Project Name: {project_info['project_name']}
Root Path: {project_info['root_path']}

STATISTICS:
- Total Files: {stats['total_files']}
- Code Files: {stats['code_files']}
- Total Directories: {stats['total_directories']}
- Total Size: {stats['total_size']} bytes
- Languages Used: {', '.join(stats['languages'])}
- File Types: {dict(list(stats['file_types'].items())[:10])}

PROJECT STRUCTURE:
{self.generate_tree_view(project_info['structure'])}

KEY FILES CONTENT PREVIEW:
{chr(10).join(file_summary[:20])}

Please provide a comprehensive analysis of this project including:
1. Project type and purpose (based on structure and files)
2. Technology stack and architecture
3. Main components and their likely purposes
4. Code quality observations
5. Potential areas for improvement
6. Overall assessment and recommendations
"""
        return prompt_data

    def generate_summary(self, project_info: Dict) -> str:
        prompt_template = ChatPromptTemplate.from_template(
            """You are an expert software architect providing a detailed analysis of this project. Write as if you are confidently describing what this project IS and DOES.

{project_data}

Provide your analysis using confident, definitive language. Avoid tentative phrases like "appears to be", "seems to", "likely", "probably", "suggests". Instead use direct statements about what the project IS, CONTAINS, and ACCOMPLISHES.

Examples of confident language:
- "This project IS a web application that..."
- "The architecture USES a microservices pattern..."
- "The codebase IMPLEMENTS advanced algorithms..."
- "This system PROVIDES real-time data processing..."

Cover all the requested points with specific, insightful observations using definitive language."""
        )
        prompt_data = self.create_summary_prompt(project_info)
        try:
            llm_chain = prompt_template | self.chat_model | StrOutputParser()
            summary = llm_chain.invoke({"project_data": prompt_data})
            return summary.strip()
        except Exception as e:
            return f"Error generating LLM summary: {str(e)}\n\nFallback: Basic project analysis shows {project_info['statistics']['total_files']} files across {len(project_info['statistics']['languages'])} programming languages."

    def generate_natural_description(self, project_info: Dict) -> str:
        """Generate a natural, user-friendly description of what the project is about"""
        file_summary = []
        for file_info in project_info["files"][:20]:  # Limit to top 20 files
            if file_info.get("is_code") and file_info.get("content_preview"):
                file_summary.append(
                    f"File: {file_info['relative_path']}\n"
                    f"Extension: {file_info.get('extension', 'unknown')}\n"
                    f"Preview: {file_info['content_preview'][:300]}...\n"
                )
        
        stats = project_info["statistics"]
        
        # Get main file types for context
        main_files = []
        for file_info in project_info["files"][:10]:
            if file_info.get("is_code"):
                main_files.append(f"{file_info['name']}")
        
        prompt_template = ChatPromptTemplate.from_template(
            """You are writing a project description as if you are the project owner introducing YOUR project to others.

Write a brief, confident description (2-3 paragraphs) that explains what this project IS and DOES. Make it:
- Written in first person or direct statements (avoid "appears to be", "seems like", "likely", "probably")
- Confident and definitive about what the project does
- Natural and conversational tone
- Use appropriate emojis where they add value (but don't overdo it)
- Focus on the project's purpose and functionality

PROJECT DETAILS:
Project Name: {project_name}
Languages: {languages}
Total Files: {total_files}
Main Files: {main_files}
File Types: {file_types}

KEY FILES PREVIEW:
{file_preview}

Write as if you're the creator saying "This project is..." or "I built this to..." or simply state what it does directly. 

Examples of what TO do:
- "This is a cryptocurrency trading bot built with Python..."
- "I created this web application to help users manage..."
- "This project provides a complete solution for..."

Examples of what NOT to do:
- "This appears to be a trading bot..."
- "The project seems to focus on..."
- "It likely handles..."
- "Based on the structure, this looks like..."

Be definitive, confident, and direct about what the project IS and DOES."""
        )
        
        try:
            llm_chain = prompt_template | self.chat_model | StrOutputParser()
            description = llm_chain.invoke({
                "project_name": project_info['project_name'],
                "languages": ', '.join(stats['languages']) if stats['languages'] else 'Unknown',
                "total_files": stats['total_files'],
                "main_files": ', '.join([f"`{f}`" for f in main_files[:5]]) if main_files else 'No main files identified',
                "file_types": ', '.join([f"`{ext}`({count})" for ext, count in list(stats['file_types'].items())[:5]]),
                "file_preview": '\n'.join(file_summary[:10])
            })
            return description.strip()
        except Exception as e:
            # Fallback natural description
            languages = ', '.join(stats['languages']) if stats['languages'] else 'mixed technologies'
            project_title = project_info['project_name'].replace('_', ' ').replace('-', ' ').title()
            return f"""🚀 **{project_title}** is a software project built with {languages}. 

This project contains {stats['total_files']} files organized across {stats['total_directories']} directories, with {stats['code_files']} code files making up the core functionality. The project follows standard project organization patterns and is ready for development.

Based on the file structure and technology stack, this is a well-organized codebase that's designed for collaboration and active development! 💻✨"""

    def generate_descriptions(self, project_info: Dict) -> Dict:
        descriptions = {"directories": {}, "files": {}}
        files_by_dir = {}
        for file_info in project_info["files"]:
            dir_path = os.path.dirname(file_info["relative_path"]) or "root"
            if dir_path not in files_by_dir:
                files_by_dir[dir_path] = []
            files_by_dir[dir_path].append(file_info)

        for dir_path in list(files_by_dir.keys()) + [
            d for d in project_info["directories"] if d not in files_by_dir
        ]:
            print(f"Generating descriptions for directory: {dir_path}")
            file_list = [f["name"] for f in files_by_dir.get(dir_path, [])]
            # Batch files if there are more than 10
            batch_size = 10
            file_batches = [
                file_list[i : i + batch_size]
                for i in range(0, len(file_list), batch_size)
            ]

            dir_description = ""
            file_descriptions = {}

            for batch in file_batches:
                prompt_template = ChatPromptTemplate.from_template(
                    """
                    For the directory "{directory}", which contains the files: {file_list}, provide:
                    1. A brief description of what the directory IS and DOES (1-2 sentences) - use confident, definitive language
                    2. A brief description for each file explaining what it IS and DOES (1-2 sentences) - use confident, definitive language

                    Use confident language like "contains", "provides", "handles", "implements" instead of "appears to", "seems to", "likely".

                    Respond **only** with a valid JSON object in this format:
                    {{
                        "directory_description": "Description of what this directory is and does",
                        "files": {{
                            "file1.py": "Description of what this file is and does",
                            "file2.py": "Description of what this file is and does"
                        }}
                    }}
                    """
                )
                try:
                    llm_chain = prompt_template | self.chat_model | StrOutputParser()
                    response = llm_chain.invoke(
                        {
                            "directory": dir_path,
                            "file_list": ", ".join(batch) if batch else "No files",
                        }
                    )
                    # Log the raw response for debugging
                    print(f"Raw LLM Response for {dir_path}: '{response}'")

                    # Extract JSON content from the response
                    response_content = self.extract_json_from_markdown(response)
                    if response_content:
                        print(f"Extracted JSON for {dir_path}: '{response_content}'")
                        # Parse the extracted JSON
                        dir_data = json.loads(response_content)
                        if not dir_description:
                            dir_description = dir_data.get(
                                "directory_description", "Description unavailable"
                            )
                        for filename, desc in dir_data.get("files", {}).items():
                            file_path = os.path.join(dir_path, filename)
                            file_descriptions[file_path] = desc
                    else:
                        print(f"No JSON found in response for {dir_path}: '{response}'")
                        dir_description = "Description unavailable due to missing JSON"
                        for filename in batch:
                            file_path = os.path.join(dir_path, filename)
                            file_descriptions[file_path] = (
                                "Description unavailable due to missing JSON"
                            )
                except json.JSONDecodeError as e:
                    print(f"JSON Parsing Error for {dir_path}: {e}")
                    print(f"Failed Response: '{response_content}'")
                    dir_description = (
                        "Description unavailable due to JSON parsing error"
                    )
                    for filename in batch:
                        file_path = os.path.join(dir_path, filename)
                        file_descriptions[file_path] = (
                            "Description unavailable due to JSON parsing error"
                        )
                except Exception as e:
                    print(f"Unexpected Error for {dir_path}: {e}")
                    dir_description = "Description unavailable due to unexpected error"
                    for filename in batch:
                        file_path = os.path.join(dir_path, filename)
                        file_descriptions[file_path] = (
                            "Description unavailable due to unexpected error"
                        )

            descriptions["directories"][dir_path] = dir_description
            descriptions["files"].update(file_descriptions)

        return descriptions

    def get_file_icon(self, extension: str) -> str:
        icons = {
            ".py": "🐍",
            ".js": "📜",
            ".ts": "📘",
            ".tsx": "⚛️",
            ".jsx": "⚛️",
            ".html": "🌐",
            ".css": "🎨",
            ".scss": "🎨",
            ".json": "📋",
            ".md": "📖",
            ".txt": "📄",
            ".yml": "⚙️",
            ".yaml": "⚙️",
            ".dockerfile": "🐳",
            ".sql": "🗄️",
            ".sh": "⚡",
            ".bat": "⚡",
            ".xml": "📄",
            ".csv": "📊",
            ".log": "📝",
            ".env": "🔐",
            "": "📄",
        }
        return icons.get(extension.lower(), "📄")

    def analyze_project(
        self,
        root_path: str,
        ignore_folders: List[str] = None,
        output_file: str = None,
        include_tree: bool = True,
        generate_readme: bool = True,
    ) -> Dict:
        print(f"Analyzing project: {root_path}")
        project_info = self.analyze_directory(root_path, ignore_folders)
        print(
            f"Found {project_info['statistics']['total_files']} files in {project_info['statistics']['total_directories']} directories"
        )
        
        print("Generating natural project description...")
        natural_description = self.generate_natural_description(project_info)
        
        print("Generating LLM summary...")
        llm_summary = self.generate_summary(project_info)
        
        descriptions = {}
        if generate_readme:
            print("Generating detailed descriptions...")
            descriptions = self.generate_descriptions(project_info)
        
        report = {
            "project_info": project_info,
            "natural_description": natural_description,
            "llm_summary": llm_summary,
            "descriptions": descriptions,
            "tree_view": (
                self.generate_tree_view(project_info["structure"])
                if include_tree
                else None
            ),
        }
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"Analysis saved to: {output_file}")
        return report


def main():
    analyzer = ProjectAnalyzer(timeout=120)
    project_path = input("Enter the path to the project folder: ").strip()

    if not os.path.isdir(project_path):
        print(f"Invalid path: {project_path}")
        return

    try:
        results = analyzer.analyze_project(
            root_path=project_path,
            ignore_folders=[
                "logs",
                "test_data",
                "temp",
                "cache",
                "node_modules",
                "data",
                ".next",
                "public",
            ],
            output_file="project_analysis.json",
            include_tree=True,
            generate_readme=True,
        )

        # Generate the README content with natural description first
        project_name = results['project_info']['project_name']
        natural_description = results['natural_description']
        
        # Quick stats with emojis
        stats = results['project_info']['statistics']
        quick_stats = f"""## 📊 Quick Stats

- 📁 **{stats['total_files']} files** across {stats['total_directories']} directories
- 💻 **{stats['code_files']} code files** in {len(stats['languages'])} programming languages
- 🚀 **Languages:** {', '.join(stats['languages']) if stats['languages'] else 'Mixed'}
- 📦 **Size:** {stats['total_size']:,} bytes

---"""

        # Construct the console output string
        console_output = f"""
{'=' * 50}
PROJECT ANALYSIS COMPLETE
{'=' * 50}

Project: {project_name}
Files: {stats['total_files']}
Languages: {', '.join(stats['languages'])}

{'-' * 50}
NATURAL DESCRIPTION:
{'-' * 50}
{natural_description}

{'-' * 50}
LLM SUMMARY:
{'-' * 50}
{results['llm_summary']}

{'-' * 50}
PROJECT STRUCTURE:
{'-' * 50}
{results['tree_view']}
"""

        # Print to console
        print(console_output)

        # Generate the flat list of files and directories
        descriptions = results["descriptions"]
        all_paths = list(descriptions["directories"].keys()) + list(
            descriptions["files"].keys()
        )
        all_paths.sort()
        flat_list = "## 📋 All Files and Directories\n\n"
        for path in all_paths:
            if path in descriptions["directories"]:
                desc = descriptions["directories"][path]
                flat_list += f"- 📁 **`{path}/`** - {desc}\n"
            else:
                desc = descriptions["files"][path]
                ext = os.path.splitext(path)[1]
                icon = analyzer.get_file_icon(ext)
                flat_list += f"- {icon} **`{path}`** - {desc}\n"

        # Create the complete README content with natural description first
        project_title = project_name.replace('_', ' ').replace('-', ' ').title()
        readme_content = f"""# {project_title}

{natural_description}

{quick_stats}

## 🔍 Detailed Analysis

{results['llm_summary']}

## 🌳 Project Structure

```
{results['tree_view']}
```

{flat_list}

---
*This README was automatically generated by ProjectAnalyzer* ✨
"""

        readme_path = os.path.join(project_path, "README.md")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(readme_content)
        print(f"README.md generated at: {readme_path}")

    except Exception as e:
        print(f"Error analyzing project: {e}")


if __name__ == "__main__":
    main()

# /Users/darlingtongospel/Sites/ai_trader_bot_course