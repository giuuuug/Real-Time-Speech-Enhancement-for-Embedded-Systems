import os
from pathlib import Path
from typing import List, Optional




def validate_dir(dir_path: str) -> Path:
    """Verifica che la directory esista e restituisce un oggetto Path."""
    path = Path(dir_path)
    if not path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
    return path


def ensure_dir_not_empty(dir_path: str):
    """Solleva un errore se la directory è vuota."""
    path = validate_dir(dir_path)
    if not any(path.iterdir()):
        raise ValueError(f"Directory is empty: {dir_path}")


def count_files_in_dir(dir_path: str) -> int:
    path = validate_dir(dir_path)
    return sum(1 for p in path.iterdir() if p.is_file())


def get_all_files(dir_path: str) -> List[str]:
    """Restituisce la lista ordinata di tutti i file (percorsi assoluti)."""
    path = validate_dir(dir_path)
    # Filtra solo i file e converte in stringa
    files = [str(p) for p in path.iterdir() if p.is_file()]
    files.sort()
    return files


def get_file_by_index_in_dir(dir_path: str, index: int) -> str:
    """
    FIXME - ATTENZIONE: Questa funzione è inefficiente se chiamata ripetutamente (es. in un loop).
    Legge e ordina la directory ogni volta.
    """
    files = get_all_files(dir_path)
    try:
        return files[index]
    except IndexError:
        raise IndexError(
            f"Index {index} out of range for directory {dir_path} with {len(files)} files."
        )


def get_latest_file_in_dir(dir_path: str) -> Optional[str]:
    path = validate_dir(dir_path)
    files = [p for p in path.iterdir() if p.is_file()]

    if not files:
        return None

    # Trova il file più recente in base al tempo di modifica
    latest_file = max(files, key=os.path.getmtime)
    return str(latest_file)

