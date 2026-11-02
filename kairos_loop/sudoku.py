"""
sudoku.py — данные для эксперимента Этапа 0 (CIDA-Loop).

Два источника:
    1. Sudoku-Extreme (тот же набор, что в HRM/TRM) — для сравнения с опубликованными числами.
    2. Локальный генератор головоломок с единственным решением — работает без интернета.

Кодировка: 81 клетка построчно, 0 = пусто, 1..9 = цифра.
Аугментации сохраняют корректность судоку: перестановка цифр, полос, строк внутри полосы,
стопок, столбцов внутри стопки и транспонирование.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple

import numpy as np

FULL_MASK = 0x3FE  # биты 1..9


def _box(r: int, c: int) -> int:
    return (r // 3) * 3 + c // 3


def count_solutions(grid, limit: int = 2) -> int:
    """Число решений (не больше limit). Бэктрекинг с битовыми масками и MRV."""
    g = [int(v) for v in grid]
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
    for i, v in enumerate(g):
        if v:
            b = 1 << v
            r, c = divmod(i, 9)
            k = _box(r, c)
            if rows[r] & b or cols[c] & b or boxes[k] & b:
                return 0
            rows[r] |= b
            cols[c] |= b
            boxes[k] |= b
    empties = [i for i in range(81) if g[i] == 0]
    count = 0

    def solve() -> bool:
        nonlocal count
        best, best_mask, best_n = -1, 0, 10
        for i in empties:
            if g[i] == 0:
                r, c = divmod(i, 9)
                m = ~(rows[r] | cols[c] | boxes[_box(r, c)]) & FULL_MASK
                n = bin(m).count("1")
                if n < best_n:
                    best, best_mask, best_n = i, m, n
                    if n <= 1:
                        break
        if best == -1:
            count += 1
            return count >= limit
        if best_n == 0:
            return False
        r, c = divmod(best, 9)
        k = _box(r, c)
        m = best_mask
        while m:
            b = m & -m
            m ^= b
            g[best] = b.bit_length() - 1
            rows[r] |= b
            cols[c] |= b
            boxes[k] |= b
            if solve():
                return True
            g[best] = 0
            rows[r] ^= b
            cols[c] ^= b
            boxes[k] ^= b
        return False

    solve()
    return count


def random_solution(rng: np.random.Generator) -> np.ndarray:
    """Случайная заполненная сетка (бэктрекинг со случайным порядком цифр)."""
    g = [0] * 81
    rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9

    def fill(i: int) -> bool:
        if i == 81:
            return True
        r, c = divmod(i, 9)
        k = _box(r, c)
        for v in rng.permutation(9) + 1:
            b = 1 << int(v)
            if rows[r] & b or cols[c] & b or boxes[k] & b:
                continue
            g[i] = int(v)
            rows[r] |= b
            cols[c] |= b
            boxes[k] |= b
            if fill(i + 1):
                return True
            rows[r] ^= b
            cols[c] ^= b
            boxes[k] ^= b
        g[i] = 0
        return False

    fill(0)
    return np.array(g, dtype=np.int64)


def make_puzzle(rng: np.random.Generator, min_clues: int = 22) -> Tuple[np.ndarray, np.ndarray]:
    """Убирает клетки, пока решение остаётся единственным (или пока не дошли до min_clues)."""
    solution = random_solution(rng)
    puzzle = solution.copy()
    clues = 81
    for i in rng.permutation(81):
        if clues <= min_clues:
            break
        v = puzzle[i]
        puzzle[i] = 0
        if count_solutions(puzzle, 2) != 1:
            puzzle[i] = v
        else:
            clues -= 1
    return puzzle, solution


def generate_dataset(n: int, seed: int = 0, min_clues: int = 22) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    pairs = [make_puzzle(rng, min_clues) for _ in range(n)]
    return np.stack([p for p, _ in pairs]), np.stack([s for _, s in pairs])


def load_sudoku_extreme(split: str, n: Optional[int] = None, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Sudoku-Extreme с Hugging Face (sapientinc/sudoku-extreme, файлы train.csv / test.csv)."""
    from huggingface_hub import hf_hub_download

    print(f"[data] downloading sudoku-extreme/{split}.csv ...", flush=True)
    path = hf_hub_download("sapientinc/sudoku-extreme", f"{split}.csv", repo_type="dataset")
    return read_sudoku_csv(path, n, seed)


def _grid_bytes_to_array(cells: bytes) -> np.ndarray:
    arr = np.frombuffer(cells.strip(b'"'), dtype=np.uint8).astype(np.int64) - ord("0")
    arr[(arr < 0) | (arr > 9)] = 0  # '.' и прочие символы пустых клеток → 0
    return arr


def read_sudoku_csv(path: str, n: Optional[int] = None, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Читает CSV (source, question, answer, rating). Сначала выбирает n строк, потом разбирает только их:
    в train.csv миллионы головоломок, полный разбор занимал бы десятки минут и гигабайты памяти.
    """
    with open(path, "rb") as f:
        header = f.readline().decode().strip().split(",")
        lines = f.read().splitlines()
    lines = [line for line in lines if line]
    q_col = header.index("question") if "question" in header else 1
    a_col = header.index("answer") if "answer" in header else 2
    if n is not None and n < len(lines):
        idx = np.sort(np.random.default_rng(seed).choice(len(lines), size=n, replace=False))
    else:
        idx = np.arange(len(lines))
    puzzles, solutions = [], []
    for i in idx:
        row = lines[i].split(b",")
        puzzles.append(_grid_bytes_to_array(row[q_col]))
        solutions.append(_grid_bytes_to_array(row[a_col]))
    print(f"[data] {path}: {len(lines)} rows, using {len(idx)}", flush=True)
    return np.stack(puzzles), np.stack(solutions)


def is_valid_solution(solution: np.ndarray) -> bool:
    g = np.asarray(solution).reshape(9, 9)
    target = set(range(1, 10))
    for i in range(9):
        if set(g[i]) != target or set(g[:, i]) != target:
            return False
        r, c = divmod(i, 3)
        if set(g[r * 3:r * 3 + 3, c * 3:c * 3 + 3].ravel()) != target:
            return False
    return True


def _random_cell_permutation(rng: np.random.Generator) -> np.ndarray:
    rows = np.concatenate([b * 3 + rng.permutation(3) for b in rng.permutation(3)])
    cols = np.concatenate([s * 3 + rng.permutation(3) for s in rng.permutation(3)])
    idx = rows[:, None] * 9 + cols[None, :]
    if rng.random() < 0.5:
        idx = idx.T
    return idx.reshape(81)


def augment(puzzles: np.ndarray, solutions: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Случайная симметрия судоку для каждого примера батча."""
    b = len(puzzles)
    idx = np.stack([_random_cell_permutation(rng) for _ in range(b)])
    digit_map = np.stack([np.concatenate([[0], rng.permutation(9) + 1]) for _ in range(b)])
    p = np.take_along_axis(puzzles, idx, 1)
    s = np.take_along_axis(solutions, idx, 1)
    return np.take_along_axis(digit_map, p, 1), np.take_along_axis(digit_map, s, 1)


def batches(
    puzzles: np.ndarray,
    solutions: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    use_augment: bool = True,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Бесконечный поток батчей из базового набора (с аугментацией на лету)."""
    while True:
        idx = rng.integers(0, len(puzzles), size=batch_size)
        p, s = puzzles[idx], solutions[idx]
        if use_augment:
            p, s = augment(p, s, rng)
        yield p, s
