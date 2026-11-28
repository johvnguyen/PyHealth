# Author(s): Nishanth Manjunath, Esther Hwang, Hekili Jordan
# NetID(s): inseoh2, nm89, hekilij2
"""Ablation of the MIMIC-III ICD-9 category task with a Transformers encoder.

The task ``MIMIC3ICD9CategoryClassification`` turns MIMIC-III discharge
summaries into a 19-label multilabel problem (Gao et al., MLHC 2022). This
script trains PyHealth's ``TransformersModel`` on it under several task
configurations and compares them on a held-out set of patients.

Ablation (not part of the paper, which uses one text setting):

* **How much of the note is read.** Only the first 64, 128 or 256 words of the
  discharge summary are kept. If the diagnoses are mostly stated early in the
  note, a short prefix is enough, and the classifier is cheaper to train.
* **Which notes are used.** Discharge summaries only, or discharge summaries
  plus physician notes. The notes of an admission are joined in time order and
  then cut to the first N words.

Each configuration is run with each seed. The seed changes the patient split
and the model initialisation, so the table reports the mean and the standard
deviation over seeds. The metrics are micro-averaged, so they stay defined
when a small split lacks some of the 19 categories.

Usage with MIMIC-III (needs credentialed access)::

    python examples/mimic3_icd9_category_classification_transformers.py \\
        --root /path/to/mimic-iii/1.4 --epochs 5 --seeds 0 1 2

Usage without any real data, to check that the pipeline runs. The notes are
synthetic, so the numbers say nothing about the real task::

    python examples/mimic3_icd9_category_classification_transformers.py \\
        --synthetic --epochs 3

The default encoder is ``emilyalsentzer/Bio_ClinicalBERT`` and is downloaded
from Hugging Face on first use. Use a GPU for the full MIMIC-III data.

The encoder reads at most ``--max-length`` tokens (512 by default), which is
roughly 350 words of clinical text. Word limits above that make no difference,
because the extra words are cut by the encoder, not by the task.

Results on synthetic notes (``--synthetic --epochs 3 --seeds 0 1 2``, Colab T4
GPU, ``Bio_ClinicalBERT``, 80 patients and 122 admissions). The notes are
synthetic, so this only shows that the pipeline trains end to end. It says
nothing about the real task::

    | configuration | pr_auc_micro | roc_auc_micro | f1_micro |
    |---|---|---|---|
    | discharge_64_words | 0.181 ± 0.020 | 0.755 ± 0.012 | 0.221 ± 0.015 |
    | discharge_128_words | 0.222 ± 0.064 | 0.779 ± 0.050 | 0.225 ± 0.038 |
    | discharge_256_words | 0.201 ± 0.035 | 0.796 ± 0.047 | 0.276 ± 0.042 |
    | discharge_physician_256_words | 0.187 ± 0.037 | 0.780 ± 0.047 | 0.264 ± 0.029 |

Mean ± standard deviation over 3 seeds. The test split has only about 18
admissions, and the differences between settings are no larger than the spread
across seeds, so no setting is better than another here. Real MIMIC-III notes
have not been run.
"""

import argparse
import csv
import gzip
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import TransformersModel
from pyhealth.tasks import MIMIC3ICD9CategoryClassification
from pyhealth.trainer import Trainer

# Name -> keyword arguments of the task. Each entry is one ablation setting.
ABLATION: dict[str, dict[str, Any]] = {
    "discharge_64_words": {"max_words": 64},
    "discharge_128_words": {"max_words": 128},
    "discharge_256_words": {"max_words": 256},
    "discharge_physician_256_words": {
        "note_categories": ("Discharge summary", "Physician"),
        "max_words": 256,
    },
}
METRICS = ["pr_auc_micro", "roc_auc_micro", "f1_micro"]
MONITOR = "pr_auc_micro"

# Synthetic data: ICD-9 code and the phrase that signals it in a note.
SYNTHETIC_CONDITIONS = {
    "4280": "heart failure",
    "486": "pneumonia",
    "25000": "diabetes",
    "5849": "kidney failure",
    "311": "depression",
    "1629": "lung cancer",
}
FILLER_WORDS = [
    "patient",
    "admitted",
    "stable",
    "discharged",
    "follow",
    "plan",
    "medication",
    "history",
    "exam",
    "vitals",
    "normal",
    "labs",
    "reviewed",
    "family",
    "social",
    "allergies",
    "denies",
    "pain",
]


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch.

    Args:
        seed: The seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_gz_csv(path: Path, header: list[str], rows: list[list[Any]]) -> None:
    """Write a gzipped CSV file.

    Args:
        path: The output path.
        header: The column names.
        rows: The rows.
    """
    with gzip.open(path, "wt", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def make_note(phrases: list[str], num_words: int, rng: random.Random) -> str:
    """Build a synthetic note with the given phrases at random positions.

    Args:
        phrases: Phrases to hide in the note, one per diagnosis.
        num_words: Number of filler words around the phrases.
        rng: Random generator.

    Returns:
        The note text.
    """
    words = [rng.choice(FILLER_WORDS) for _ in range(num_words)]
    for phrase in phrases:
        words.insert(rng.randrange(len(words) + 1), phrase)
    return " ".join(words)


def write_synthetic_mimic3(root: Path, num_patients: int, seed: int) -> None:
    """Write a small fake MIMIC-III folder with synthetic discharge summaries.

    Each admission gets one or two of the conditions in
    ``SYNTHETIC_CONDITIONS``, and a discharge summary that mentions them at a
    random position, so a longer prefix of the note is more likely to contain
    the signal. It also gets a physician note that mentions nothing.

    Args:
        root: An existing empty folder.
        num_patients: Number of patients, with one or two admissions each.
        seed: Random seed.
    """
    rng = random.Random(seed)
    patients, admissions, icustays, diagnoses, notes = [], [], [], [], []
    hadm_id, row_id = 100, 1
    for subject_id in range(1, num_patients + 1):
        patients.append([subject_id, "F", "2050-01-01", "", "", "", 0])
        for visit in range(rng.randint(1, 2)):
            hadm_id += 1
            day = f"21{visit:02d}-01"
            admissions.append(
                [subject_id, hadm_id, f"{day}-01 10:00:00", f"{day}-05 10:00:00"]
                + [""] * 14
            )
            icustays.append(
                [subject_id, hadm_id, hadm_id, "MICU", "metavision", "MICU"]
                + [f"{day}-01 11:00:00", f"{day}-02 11:00:00"]
            )
            codes = rng.sample(sorted(SYNTHETIC_CONDITIONS), rng.randint(1, 2))
            for seq_num, code in enumerate(codes, start=1):
                diagnoses.append([row_id, subject_id, hadm_id, seq_num, code])
                row_id += 1
            phrases = [SYNTHETIC_CONDITIONS[code] for code in codes]
            for category, text in (
                ("Physician", make_note([], 150, rng)),
                ("Discharge summary", make_note(phrases, 300, rng)),
            ):
                notes.append(
                    [row_id, subject_id, hadm_id, f"{day}-05", f"{day}-05 09:00:00"]
                    + ["", category, "Report", "", "", text]
                )
                row_id += 1

    admission_columns = [
        "subject_id", "hadm_id", "admittime", "dischtime", "admission_type",
        "admission_location", "insurance", "language", "religion",
        "marital_status", "ethnicity", "edregtime", "edouttime", "deathtime",
        "discharge_location", "diagnosis", "hospital_expire_flag",
        "has_chartevents_data",
    ]  # fmt: skip
    write_gz_csv(
        root / "PATIENTS.csv.gz",
        ["subject_id", "gender", "dob", "dod", "dod_hosp", "dod_ssn", "expire_flag"],
        patients,
    )
    write_gz_csv(root / "ADMISSIONS.csv.gz", admission_columns, admissions)
    write_gz_csv(
        root / "ICUSTAYS.csv.gz",
        ["subject_id", "hadm_id", "icustay_id", "first_careunit", "dbsource",
         "last_careunit", "intime", "outtime"],
        icustays,
    )  # fmt: skip
    write_gz_csv(
        root / "DIAGNOSES_ICD.csv.gz",
        ["row_id", "subject_id", "hadm_id", "seq_num", "icd9_code"],
        diagnoses,
    )
    write_gz_csv(
        root / "NOTEEVENTS.csv.gz",
        ["row_id", "subject_id", "hadm_id", "chartdate", "charttime", "storetime",
         "category", "description", "cgid", "iserror", "text"],
        notes,
    )  # fmt: skip


def run_once(
    dataset: MIMIC3Dataset,
    task_kwargs: dict[str, Any],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    """Train and test one model for one task configuration and one seed.

    Args:
        dataset: The MIMIC-III dataset, with the notes and diagnoses tables.
        task_kwargs: Keyword arguments of ``MIMIC3ICD9CategoryClassification``.
        args: The command-line arguments.
        seed: Seed for the patient split and the model initialisation.

    Returns:
        The test-set metrics, keyed by metric name.
    """
    set_seed(seed)
    task = MIMIC3ICD9CategoryClassification(**task_kwargs)
    samples = dataset.set_task(task)
    train_set, val_set, test_set = split_by_patient(
        samples, [0.7, 0.15, 0.15], seed=seed
    )
    loaders = [
        get_dataloader(subset, batch_size=args.batch_size, shuffle=shuffle)
        for subset, shuffle in ((train_set, True), (val_set, False), (test_set, False))
    ]
    model = TransformersModel(
        dataset=samples, model_name=args.model_name, max_length=args.max_length
    )
    trainer = Trainer(
        model=model,
        metrics=METRICS,
        device="cuda" if torch.cuda.is_available() else "cpu",
        enable_logging=False,
    )
    trainer.train(
        train_dataloader=loaders[0],
        val_dataloader=loaders[1],
        epochs=args.epochs,
        optimizer_params={"lr": args.lr},
        monitor=MONITOR,
    )
    return trainer.evaluate(loaders[2])


def print_table(results: dict[str, list[dict[str, float]]]) -> None:
    """Print the mean and standard deviation of each metric per configuration.

    Args:
        results: For each configuration name, the test metrics of every seed.
    """
    print(f"\n| configuration | {' | '.join(METRICS)} |")
    print(f"|---|{'---|' * len(METRICS)}")
    for name, runs in results.items():
        cells = []
        for metric in METRICS:
            values = np.array([run[metric] for run in runs])
            cells.append(f"{values.mean():.3f} ± {values.std():.3f}")
        print(f"| {name} | {' | '.join(cells)} |")
    num_seeds = len(next(iter(results.values())))
    print(f"\nMean ± standard deviation over {num_seeds} seed(s).")


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", help="Path to the MIMIC-III folder.")
    parser.add_argument(
        "--synthetic", action="store_true", help="Use synthetic notes, no real data."
    )
    parser.add_argument("--dev", action="store_true", help="Load a small dev subset.")
    parser.add_argument("--model-name", default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--num-synthetic-patients", type=int, default=80)
    args = parser.parse_args()
    if not args.synthetic and not args.root:
        parser.error("give --root, or use --synthetic")
    return args


def main() -> None:
    """Run the ablation and print the results table."""
    args = parse_args()
    with tempfile.TemporaryDirectory() as scratch:
        root = args.root
        if args.synthetic:
            root = Path(scratch) / "mimic3"
            root.mkdir()
            write_synthetic_mimic3(root, args.num_synthetic_patients, seed=0)
        dataset = MIMIC3Dataset(
            root=str(root),
            tables=["diagnoses_icd", "noteevents"],
            cache_dir=scratch,
            dev=args.dev,
        )
        results: dict[str, list[dict[str, float]]] = {}
        for name, task_kwargs in ABLATION.items():
            print(f"\n=== {name} ===")
            results[name] = [
                run_once(dataset, task_kwargs, args, seed) for seed in args.seeds
            ]
        print_table(results)


if __name__ == "__main__":
    main()