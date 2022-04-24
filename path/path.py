from pathlib import Path

# Project_Root: Guildline_Generation Directory
def get_project_root() -> Path:
    return str(Path(__file__).parent.parent)