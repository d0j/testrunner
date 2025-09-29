#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ЗАЧЕМ:
- Быстро прогонять задачи с кейсами (вход/эталон) без правок вашей программы.
- Видеть понятный отчёт: шапка-таблица, для FAIL — вход (INPUT), реальный STDOUT, EXPECTED и дружелюбный DIFF.

КАК РАБОТАЕТ (high level):
1) Рядом с runner лежит ваша программа (по умолчанию: test.py) и папка с тестами ИЛИ ZIP с тестами.
2) В тестах пары файлов:
     N         — «вход»: Питон-код, который надо выполнить в namespace вашей программы
     N.clue    — «ожидаемый stdout»
3) Мы не трогаем вашу программу: создаём «harness» — небольшой хук, который:
     - импортирует (runpy.run_path) вашу программу в отдельный namespace,
     - читает stdin (содержимое файла N) и выполняет его через exec в том же namespace.
   Т.е. всё, как будто вы написали тестовый код прямо внизу вашего файла — только без правок.
4) Собираем stdout/stderr, сравниваем stdout с эталоном:
     по умолчанию сравнение «ПО СТРОКАМ» после normalizing CRLF→LF (newline-mode=lf).
     Это даёт «exact, но кроссплатформенный»: игнорируем только различие в типе перевода строк и финальном \n.
5) Печатаем сводку (таблица), затем детальные блоки только для FAIL.
   Для FAIL показываем:
     — ---INPUT--- (сырой текст входа, копипастопригодный; максимум N строк)
     — ---STDOUT--- (реальный вывод, опционально с «невидимыми» символами)
     — ---EXPECTED--- (эталон, опционально с «невидимыми»)
     — ---DIFF--- (первая несовпадающая строка и «каретка»)

КЛЮЧЕВЫЕ ПРИЁМЫ:
- Ничего не меняем в вашей программе. Все «танцы» — в harness.
- STDOUT/EXPECTED: можно подсвечивать «невидимые» (space→·, tab→→, конец строки отмечаем ⏎).
- INPUT теперь выводится «как есть», без нумерации и подсветки — чтобы легко копировать в отладку.

ПАРАМЕТРЫ:
  --prog PATH            путь к программе (по умолчанию: test.py)
  --cases PATH           путь к папке с тестами ИЛИ zip; если не задан, runner сам найдёт лучшее рядом
  --timeout SECONDS      таймаут каждого кейса (по умолчанию: 3.0)
  --lines N              максимум показываемых строк в STDOUT/EXPECTED/DIFF (по умолчанию: 10)
  --no-invisibles        не показывать «невидимые» символы в STDOUT/EXPECTED (· → ⏎)
  --allow-nonzero-exit   считать rc!=0 допустимым (по умолчанию: False)
  --newline-mode MODE    'lf' (CRLF→LF и сравнение ПО СТРОКАМ) или 'keep' (строгий побайтовый матч); по умолчанию: lf
  --show-input WHEN      'fail'|'always'|'never' — печать секции INPUT (по умолчанию: fail)
  --input-lines N        сколько строк входа печатать в INPUT (по умолчанию: 12)

СТРУКТУРА ТЕСТОВ:
  <dir>/
    1
    1.clue
    2
    2.clue
    ...

ПРИМЕЧАНИЕ:
- Если рядом с runner нет папки с тестами, но есть ZIP — runner распакует его во временную папку и использует её.
- В отчёте видны пути «case → clue», чтобы всегда можно было быстро открыть исходники кейсов.
"""

from __future__ import annotations

import argparse
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import List, Tuple, Optional


# ==========================
# Настройки по умолчанию
# ==========================
DEFAULT_PROG = "test.py"
DEFAULT_CASES = None            # если None — авто-поиск рядом с runner
TIMEOUT_SEC = 3.0
MAX_SHOW_LINES = 33
SHOW_INVISIBLES = True          # показывать · → ⏎ в STDOUT/EXPECTED
ALLOW_NONZERO_EXIT = False      # rc != 0 — это не ОК (если не переопределить флагом)
ENCODING = "utf-8"
DEFAULT_NEWLINE_MODE = "lf"     # 'lf' (CRLF→LF + сравнение ПО СТРОКАМ) | 'keep' (строгий байт-в-байт)
DEFAULT_SHOW_INPUT = "fail"     # 'fail'|'always'|'never'
DEFAULT_INPUT_LINES = 12


# ==========================
# Утилиты форматирования/детекта
# ==========================
def to_lf(s: str) -> str:
    """Нормализация перевода строк к LF (CRLF→LF)."""
    return s.replace("\r\n", "\n").replace("\r", "\n")

def read_text(p: Path, encoding: str = ENCODING) -> str:
    """Чтение текста (удаляем BOM на всякий случай)."""
    s = p.read_text(encoding=encoding, errors="strict")
    return s.lstrip("\ufeff")

def fmt_ms(sec: float) -> str:
    return f"{sec * 1e3:.1f} ms"

def clip_lines(s: str, limit: int) -> Tuple[str, int]:
    """Обрезаем вывод по строкам (не более limit), возвращаем (текст, сколько_пропустили)."""
    lines = s.splitlines()
    if len(lines) <= limit:
        return s, 0
    shown = "\n".join(lines[:limit])
    skipped = len(lines) - limit
    return shown, skipped

def visualize_invisibles_block(s: str) -> str:
    """
    STDOUT/EXPECTED: показываем «невидимые» символы (для наглядности, НЕ для копирования).
      space -> ·
      tab   -> →
      конец строки отмечаем ⏎
    """
    lines = to_lf(s).split("\n")
    out = []
    for line in lines:
        if line == "":
            out.append("⟂ <blank>⏎")
        else:
            out.append(line.replace(" ", "·").replace("\t", "→") + "⏎")
    return "\n".join(out) if out else ""

def clip_input_plain_keepends(s: str, limit: int) -> Tuple[str, int]:
    """
    INPUT: готовим «сырой», копипастопригодный блок без нумерации, без подсветки.
    Важно: сохраняем исходные переводы строк — splitlines(keepends=True).
    """
    lines = s.splitlines(keepends=True)
    if len(lines) <= limit:
        return s, 0
    shown = "".join(lines[:limit])
    skipped = len(lines) - limit
    return shown, skipped


# ==========================
# Поиск/подготовка кейсов
# ==========================
def discover_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """
    Ищем пары (input_file, clue_file):
      - input: любой файл, у которого есть сосед <same_name>.clue
      - исключаем сами .clue из входов
      - сортируем «по-человечески»: если имя — число, то по int; иначе — лексикографически
    """
    files = [p for p in root.iterdir() if p.is_file()]
    inputs = [p for p in files if not p.name.endswith(".clue")]

    def key(p: Path):
        stem = p.name
        try:
            return (0, int(stem))
        except ValueError:
            return (1, stem)

    inputs.sort(key=key)
    pairs: List[Tuple[Path, Path]] = []
    for inp in inputs:
        clue = inp.with_suffix(inp.suffix + ".clue") if inp.suffix else inp.parent / (inp.name + ".clue")
        if clue.exists():
            pairs.append((inp, clue))
    return pairs

def autodetect_cases_dir(base: Path) -> Optional[Path]:
    """
    Ищем «лучшую» папку с тестами среди подпапок:
    — сначала ./test/,
    — иначе выбираем директорию с максимальным числом корректных пар.
    """
    test_dir = base / "test"
    if test_dir.is_dir() and discover_pairs(test_dir):
        return test_dir
    best, best_cnt = None, 0
    for p in base.iterdir():
        if p.is_dir():
            cnt = len(discover_pairs(p))
            if cnt > best_cnt:
                best, best_cnt = p, cnt
    return best

def maybe_extract_zip(zip_path: Path) -> Optional[Path]:
    """Распаковать ZIP во временную папку и вернуть путь к ней (или None при ошибке)."""
    if not zip_path.exists() or not zip_path.is_file():
        return None
    try:
        tmpdir = tempfile.TemporaryDirectory()
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(tmpdir.name)
        print(f"[info] распакован архив: {zip_path} -> {tmpdir.name}")
        # удерживаем времянку до конца процесса
        maybe_extract_zip._keep = getattr(maybe_extract_zip, "_keep", [])  # type: ignore[attr-defined]
        maybe_extract_zip._keep.append(tmpdir)  # type: ignore[attr-defined]
        return Path(tmpdir.name)
    except Exception as e:
        print(f"[warn] не удалось распаковать {zip_path}: {e}", file=sys.stderr)
        return None

def autodetect_cases_dir_or_zip(base: Path) -> Optional[Path]:
    """
    Ищем тесты:
      1) папки (см. autodetect_cases_dir),
      2) если не нашли — пробуем ZIP-файлы рядом: распакуем и выберем тот, где больше всего пар.
    """
    d = autodetect_cases_dir(base)
    if d and discover_pairs(d):
        print(f"[info] авто-выбрана папка с тестами: {d}")
        return d

    zips = [p for p in base.iterdir() if p.is_file() and p.suffix.lower() == ".zip"]
    best_dir, best_cnt = None, 0
    for z in zips:
        extracted = maybe_extract_zip(z)
        if extracted:
            cnt = len(discover_pairs(extracted))
            if cnt > best_cnt:
                best_dir, best_cnt = extracted, cnt
    if best_dir and best_cnt > 0:
        print(f"[info] авто-выбрана папка с тестами: {best_dir}")
        return best_dir
    return None


# ==========================
# Сравнение и DIFF
# ==========================
def contents_equal(expected_raw: str, got_raw: str, newline_mode: str) -> bool:
    """
    Критерий «OK»:
      - 'lf'  : сравнение ПО СТРОКАМ после CRLF→LF (игнорируем только стиль EOL и финальный \n)
      - 'keep': строгий побайтовый матч
    """
    if newline_mode == "lf":
        el = to_lf(expected_raw).splitlines()
        gl = to_lf(got_raw).splitlines()
        return el == gl
    else:
        return expected_raw == got_raw

def first_diff_line(exp: List[str], got: List[str]) -> Optional[int]:
    """Первая строка, где различие; None — если одинаково и по длине тоже."""
    m = min(len(exp), len(got))
    for i in range(m):
        if exp[i] != got[i]:
            return i
    if len(exp) != len(got):
        return m
    return None

def first_diff_col(a: str, b: str) -> Optional[int]:
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return m
    return None

def human_diff(expected_raw: str, got_raw: str, newline_mode: str) -> Tuple[str, Optional[str]]:
    """
    Человеко-понятный DIFF:
      — если построчно совпало, но байты нет → почти всегда «разный финальный перевод строки»
      — иначе показываем первую различающуюся строку и «каретку».
    Возвращает (diff_text, optional_hint).
    """
    exp_lf = to_lf(expected_raw)
    got_lf = to_lf(got_raw)
    el = exp_lf.splitlines()
    gl = got_lf.splitlines()

    if el == gl and expected_raw != got_raw:
        e_end = expected_raw.endswith(("\n", "\r\n"))
        g_end = got_raw.endswith(("\n", "\r\n"))
        if e_end != g_end:
            return "(lines identical, but trailing newline differs: one has final EOL, the other doesn't)", "разный финальный перевод строки"
        if newline_mode == "keep":
            return "(lines identical, but EOL style differs: CRLF vs LF)", "разные переводы строк (CRLF vs LF)"
        return "(no visible difference — check binary/EOL details)", None

    if not el and not gl:
        return "(both empty)", None

    i = first_diff_line(el, gl)
    if i is None:
        return "(no difference found — contents equal?)", None

    e_line = el[i] if i < len(el) else ""
    g_line = gl[i] if i < len(gl) else ""
    c = first_diff_col(e_line, g_line)

    hint = None
    if e_line.rstrip() == g_line.rstrip() and e_line != g_line:
        hint = "вывод отличается только хвостовыми пробелами"
    elif e_line.lower() == g_line.lower() and e_line != g_line:
        hint = "отличается только регистр"

    lo = max(0, i - 1)
    hi = min(max(len(el), len(gl)), i + 1)
    numw = len(str(hi))
    lines = [f"(first mismatch at line {i+1})"]
    for k in range(lo, hi+1):
        e = el[k] if k < len(el) else ""
        g = gl[k] if k < len(gl) else ""
        mark = ">>" if k == i else "  "
        if k == i:
            lines.append(f"{mark}{str(k+1).rjust(numw)}: {g}    !=    {e}")
            if c is not None:
                caret = " " * (len(str(k+1)) + 2 + 2 + c) + "^ here"
                lines.append("  " + caret)
        else:
            lines.append(f"{mark}{str(k+1).rjust(numw)}: {g}")

    return "\n".join(lines), hint


# ==========================
# Harness (хук-раннер)
# ==========================
HARNESS_SRC = r"""
import sys, os, runpy
if len(sys.argv) < 2:
    sys.exit("HARNESS: missing target path")
target = sys.argv[1]
# позволяем относительные импорты из папки программы
sys.path.insert(0, os.path.dirname(os.path.abspath(target)))
# исполняем программу и получаем её namespace
ns = runpy.run_path(target, run_name="__tested__")
# читаем stdin целиком — это Питон-код входа кейса — и исполняем в том же namespace
code = sys.stdin.read()
exec(compile(code, "<stdin>", "exec"), ns, ns)
""".lstrip()

def ensure_harness(tmpdir: Path) -> Path:
    hp = tmpdir / "_harness_runner.py"
    hp.write_text(HARNESS_SRC, encoding=ENCODING)
    return hp


# ==========================
# Запуск одного кейса
# ==========================
def run_case(prog: Path, stdin_text: str, timeout: float, harness_path: Path):
    """Запускаем harness с вашей программой, подаём stdin, возвращаем (rc, stdout, stderr, dt)."""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = ENCODING
    env.setdefault("PYTHONUTF8", "1")
    cmd = [sys.executable, str(harness_path), str(prog)]
    t0 = time.perf_counter()
    try:
        cp = subprocess.run(
            cmd,
            input=stdin_text,
            text=True,
            encoding=ENCODING,
            capture_output=True,
            timeout=timeout,
            env=env,
        )
        dt = time.perf_counter() - t0
        return cp.returncode, (cp.stdout or ""), (cp.stderr or ""), dt
    except subprocess.TimeoutExpired as e:
        dt = time.perf_counter() - t0
        out = e.stdout if isinstance(e.stdout, str) else ""
        err = (e.stderr if isinstance(e.stderr, str) else "") + f"\nTIMEOUT after {timeout}s"
        return 124, out, err, dt


# ==========================
# Main
# ==========================
def main() -> int:
    ap = argparse.ArgumentParser(description="Testrunner (human-friendly, exact by lines, with INPUT/DIFF)")
    ap.add_argument("--prog", type=Path, default=DEFAULT_PROG, help="путь к программе (по умолчанию: test.py)")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="путь к папке с тестами ИЛИ zip; по умолчанию — авто-поиск рядом")
    ap.add_argument("--timeout", type=float, default=TIMEOUT_SEC, help="таймаут кейса, сек")
    ap.add_argument("--lines", type=int, default=MAX_SHOW_LINES, help="максимум строк для STDOUT/EXPECTED/DIFF")
    ap.add_argument("--no-invisibles", action="store_true", help="не показывать невидимые символы в STDOUT/EXPECTED (· → ⏎)")
    ap.add_argument("--allow-nonzero-exit", action="store_true", default=ALLOW_NONZERO_EXIT, help="считать rc!=0 допустимым")
    ap.add_argument("--newline-mode", choices=("lf", "keep"), default=DEFAULT_NEWLINE_MODE,
                    help="сравнение: 'lf' (нормализация+ПО СТРОКАМ) или 'keep' (побайтово)")
    ap.add_argument("--show-input", choices=("fail", "always", "never"), default=DEFAULT_SHOW_INPUT,
                    help="показывать секцию ---INPUT--- (по умолчанию: fail)")
    ap.add_argument("--input-lines", type=int, default=DEFAULT_INPUT_LINES,
                    help="сколько строк входа печатать в секции INPUT")
    args = ap.parse_args()

    prog = args.prog
    if not prog.exists():
        print(f"[error] программа не найдена: {prog}", file=sys.stderr)
        return 2

    # autodetect cases / zip, если не указали явно
    if args.cases is None:
        base = Path(__file__).resolve().parent
        cases_dir = autodetect_cases_dir_or_zip(base)
        if not cases_dir:
            print(f"[error] не нашёл папку/zip с тестами рядом с {base}. Укажи --cases.", file=sys.stderr)
            return 2
    else:
        if args.cases.is_file() and args.cases.suffix.lower() == ".zip":
            extracted = maybe_extract_zip(args.cases)
            if not extracted:
                print(f"[error] не удалось распаковать zip: {args.cases}", file=sys.stderr)
                return 2
            cases_dir = extracted
        else:
            cases_dir = args.cases
            if not cases_dir.exists():
                print(f"[error] путь не найден: {cases_dir}", file=sys.stderr)
                return 2
        print(f"[info] папка/zip с тестами: {cases_dir}")

    pairs = discover_pairs(cases_dir)
    if not pairs:
        print(f"[error] не найдено пар вход+ответ в {cases_dir}", file=sys.stderr)
        return 2

    print(f"Program: {prog.name}")
    print(f"Cases:   {cases_dir}  ({len(pairs)} пар)\n")

    # harness
    tmpdir = tempfile.TemporaryDirectory()
    harness = ensure_harness(Path(tmpdir.name))

    # Выполняем все кейсы, собираем результаты
    results = []
    total_t = 0.0
    for idx, (inp, clue) in enumerate(pairs, 1):
        stdin_text = read_text(inp)
        expected_raw = read_text(clue)
        rc, stdout_raw, stderr, dt = run_case(prog, stdin_text, args.timeout, harness)
        total_t += dt
        ok = contents_equal(expected_raw, stdout_raw, args.newline_mode) and (args.allow_nonzero_exit or rc == 0)
        results.append({
            "idx": idx,
            "inp": inp,
            "clue": clue,
            "rc": rc,
            "dt": dt,
            "ok": ok,
            "stdin_text": stdin_text,
            "stdout_raw": stdout_raw,
            "expected_raw": expected_raw,
            "stderr": stderr,
        })

    # ---------- Сводная таблица (выровненная) ----------
    name_w = max(4, max(len(r["inp"].name) for r in results))
    res_w  = len("RESULT")
    time_w = 10
    rc_w   = max(2, len("RC"))
    mode_w = len("exact")

    print(f" #  {'NAME'.ljust(name_w)}  {'RESULT'.ljust(res_w)}  {'TIME'.rjust(time_w)}  {'RC'.rjust(rc_w)}  {'MODE'.ljust(mode_w)}")
    print(f"--  {'-'*name_w}  {'-'*res_w}  {'-'*time_w}  {'-'*rc_w}  {'-'*mode_w}")

    for r in results:
        status = "OK" if r["ok"] else "FAIL"
        time_s = fmt_ms(r["dt"]).rjust(time_w)
        print(f"{str(r['idx']).rjust(2)}  {r['inp'].name.ljust(name_w)}  {status.ljust(res_w)}  {time_s}  {str(r['rc']).rjust(rc_w)}  {'exact'}")

    print()

    # ---------- Детальные блоки ----------
    ok_count = sum(1 for r in results if r["ok"])
    for r in results:
        show_input = (args.show_input == "always") or (args.show_input == "fail" and not r["ok"])
        if r["ok"] and not show_input:
            continue  # для OK ничего не печатаем, если не просили INPUT

        if not r["ok"]:
            print("-" * 80)
            print(f"[FAIL] #{r['idx']:g}  ({fmt_ms(r['dt'])}, rc={r['rc']})  mode=exact | newline={'LF' if args.newline_mode=='lf' else 'keep'} | lines={args.lines}")
            print(f"case: {r['inp'].name} -> {r['clue'].name}")
            print(f"path: {r['inp']}  ->  {r['clue']}")
            print("-" * 80)

        # ---INPUT--- (сырой, копипастопригодный)
        if show_input:
            print("---INPUT---")
            preview, skipped = clip_input_plain_keepends(r["stdin_text"], args.input_lines)
            print(preview, end="")  # сохраняем исходные переводы строк, не добавляем лишний \n
            if skipped > 0:
                print(f"\n... (+{skipped} lines skipped)")
            print()

        if r["ok"]:
            continue  # для OK после INPUT выходим

        # ---STDOUT---
        print("---STDOUT---")
        shown_out = visualize_invisibles_block(r["stdout_raw"]) if not args.no_invisibles else r["stdout_raw"]
        shown_out, skip_out = clip_lines(shown_out, args.lines)
        print(shown_out or "⟂ <empty>")
        if skip_out:
            print(f"... (+{skip_out} lines skipped)")
        print()

        # ---EXPECTED---
        print("---EXPECTED---")
        shown_exp = visualize_invisibles_block(r["expected_raw"]) if not args.no_invisibles else r["expected_raw"]
        shown_exp, skip_exp = clip_lines(shown_exp, args.lines)
        print(shown_exp or "⟂ <empty>")
        if skip_exp:
            print(f"... (+{skip_exp} lines skipped)")
        print()

        # ---DIFF---
        diff_text, hint = human_diff(r["expected_raw"], r["stdout_raw"], args.newline_mode)
        print(f"---DIFF---  {diff_text}")
        if hint:
            print(f"hint: {hint}.")
        print("=" * 80)
        print()

    tmpdir.cleanup()

    # Итоги
    print("Summary")
    print(f"Total: {len(results)}  |  OK: {ok_count}  |  FAIL: {len(results)-ok_count}  |  Duration: {total_t:.3f}s")
    return 0 if ok_count == len(results) and len(results) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
