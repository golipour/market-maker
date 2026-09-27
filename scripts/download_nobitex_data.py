#!/usr/bin/env python3
import csv
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

CHUNKS = {
    "aa": 2097152,
    "ab": 2097152,
    "ac": 2097152,
    "ad": 2097152,
    "ae": 2097152,
    "af": 2097152,
    "ag": 2097152,
    "ah": 2097152,
    "ai": 2097152,
    "aj": 1466160,
}

CHUNK_DIR = Path("/tmp/nob_chunks")
CHUNK_DIR.mkdir(parents=True, exist_ok=True)
REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = REPO_ROOT / "research" / "model_lab" / "nobitex_USDT_IRT_10000ms.csv"
OUTPUT_PARQUET = REPO_ROOT / "research" / "model_lab" / "nobitex_USDT_IRT_10000ms.parquet"
OUTPUT_XLSX = REPO_ROOT / "research" / "model_lab" / "nobitex_USDT_IRT_10000ms.xlsx"
OUTPUT_XZ = CHUNK_DIR / "nobitex_USDT_IRT_10000ms.csv.xz"


def download_one(name: str, expected_size: int) -> None:
    dest = CHUNK_DIR / f"xz_chunk_{name}"
    if dest.exists() and dest.stat().st_size == expected_size:
        print(f"[{name}] Already complete ({expected_size} bytes)", flush=True)
        return

    temp_dest = CHUNK_DIR / f"xz_chunk_{name}.tmp"
    env = dict(os.environ)
    env["KUBECONFIG"] = os.path.expanduser("~/.kube/odysseus-dev")

    for attempt in range(1, 25):
        current_size = temp_dest.stat().st_size if temp_dest.exists() else 0
        if current_size == expected_size:
            temp_dest.rename(dest)
            print(f"[{name}] Successfully completed {current_size} bytes", flush=True)
            return
        if current_size > expected_size:
            temp_dest.unlink()
            current_size = 0

        offset_1based = current_size + 1
        print(
            f"[{name}] Attempt {attempt}: resuming from byte {offset_1based} (need {expected_size - current_size} more bytes)...",
            flush=True,
        )

        cmd = [
            "kubectl",
            "-n",
            "market-maker",
            "exec",
            "postgres-748468bc4d-zb4hr",
            "-c",
            "postgres",
            "--",
            "tail",
            "-c",
            f"+{offset_1based}",
            f"/tmp/xz_chunk_{name}",
        ]
        with open(temp_dest, "ab") as f:
            res = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, env=env)

        new_size = temp_dest.stat().st_size if temp_dest.exists() else 0
        if new_size == expected_size:
            temp_dest.rename(dest)
            print(f"[{name}] Successfully finished {new_size} bytes", flush=True)
            return
        else:
            print(
                f"[{name}] Partial progress: now at {new_size}/{expected_size} bytes. Retrying in 2s...",
                flush=True,
            )
            time.sleep(2)

    raise RuntimeError(f"Failed to download chunk {name} after {attempt} attempts")


def convert_formats() -> None:
    print("Converting to Parquet...", flush=True)
    try:
        import pyarrow.csv as pv
        import pyarrow.parquet as pq

        table = pv.read_csv(OUTPUT_CSV)
        pq.write_table(table, OUTPUT_PARQUET, compression="snappy")
        print(f"Created {OUTPUT_PARQUET} ({OUTPUT_PARQUET.stat().st_size} bytes)", flush=True)
    except Exception as e:
        print(f"Parquet conversion failed: {e}", flush=True)

    print("Converting to Excel (.xlsx)...", flush=True)
    try:
        import openpyxl

        wb = openpyxl.Workbook(write_only=True)
        ws = wb.create_sheet(title="nobitex_USDT_IRT_10s")

        with open(OUTPUT_CSV, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader)
            ws.append(header)
            count = 0
            for row in reader:
                converted_row = []
                for val in row:
                    if val == "" or val.lower() == "null":
                        converted_row.append(None)
                    elif val == "t" or val.lower() == "true":
                        converted_row.append(True)
                    elif val == "f" or val.lower() == "false":
                        converted_row.append(False)
                    else:
                        try:
                            converted_row.append(int(val))
                        except ValueError:
                            try:
                                converted_row.append(float(val))
                            except ValueError:
                                converted_row.append(val)
                ws.append(converted_row)
                count += 1
                if count % 100000 == 0:
                    print(f"Excel export: {count} rows written...", flush=True)

        wb.save(OUTPUT_XLSX)
        print(f"Created {OUTPUT_XLSX} ({OUTPUT_XLSX.stat().st_size} bytes)", flush=True)
    except Exception as e:
        print(f"Excel conversion failed: {e}", flush=True)


def main() -> None:
    for name, size in sorted(CHUNKS.items()):
        download_one(name, size)

    print("All chunks validated! Concatenating to .xz...", flush=True)
    with open(OUTPUT_XZ, "wb") as outfile:
        for name in sorted(CHUNKS.keys()):
            chunk_file = CHUNK_DIR / f"xz_chunk_{name}"
            with open(chunk_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)

    print(f"Created {OUTPUT_XZ} ({OUTPUT_XZ.stat().st_size} bytes)", flush=True)

    print(f"Decompressing to {OUTPUT_CSV}...", flush=True)
    subprocess.run(["xz", "-d", "-k", "-f", str(OUTPUT_XZ)], check=True)
    decompressed_tmp = CHUNK_DIR / "nobitex_USDT_IRT_10000ms.csv"
    shutil.move(decompressed_tmp, OUTPUT_CSV)

    print(f"Created {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size} bytes)", flush=True)

    line_count = int(subprocess.check_output(["wc", "-l", str(OUTPUT_CSV)]).split()[0])
    print(f"Total lines in CSV: {line_count}", flush=True)

    convert_formats()
    print("All downloads and conversions complete!", flush=True)


if __name__ == "__main__":
    main()
