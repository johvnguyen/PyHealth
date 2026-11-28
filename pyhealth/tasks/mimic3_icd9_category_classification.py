# Author(s): Nishanth Manjunath, Esther Hwang, Hekili Jordan
# NetID(s): inseoh2, nm89, hekilij2	
# Paper: Classifying Unstructured Clinical Notes via Automatic Weak
#     Supervision (Gao et al., MLHC 2022)
# Paper link: https://proceedings.mlr.press/v182/gao22a.html
# Description: Multilabel classification of MIMIC-III discharge summaries into
#     the 19 top-level ICD-9 diagnosis categories used by the paper (which
#     follows FasTag, Venkataraman et al., 2020). This is the data side of
#     the paper's clinical experiment, which the authors' public repository
#     does not include.

"""MIMIC-III discharge summaries to the 19 top-level ICD-9 categories.

This module is the data side of the clinical experiment in "Classifying
Unstructured Clinical Notes via Automatic Weak Supervision" (Gao et al.,
MLHC 2022). The paper labels MIMIC-III discharge summaries with the 19
top-level ICD-9 diagnosis categories, following FasTag (Venkataraman et al.,
2020), and trains a text classifier without human-labelled notes. The
authors' public code covers only the four non-clinical benchmarks, so this
task supplies the MIMIC-III side in PyHealth.

For each hospital admission that has a discharge summary and at least one
diagnosis in one of the 19 categories, the task yields one sample::

    {
        "patient_id": "1",
        "visit_id": "100",                  # the admission (hadm_id)
        "text": "heart failure and ...",    # the discharge summary
        "categories": [                      # multilabel target, 19 outputs
            "Endocrine, nutritional and metabolic",
            "Circulatory system",
        ],
    }

Contents:
    ICD9_CATEGORIES: The 19 category names, in the order of the paper's Table 4.
    icd9_to_category: Maps one ICD-9 code to its category.
    MIMIC3ICD9CategoryClassification: The task, for ``MIMIC3Dataset``.

Typical use, with a MIMIC-III folder you have access to::

    from pyhealth.datasets import MIMIC3Dataset
    from pyhealth.tasks import MIMIC3ICD9CategoryClassification

    dataset = MIMIC3Dataset(
        root="/path/to/mimic-iii/1.4",
        tables=["diagnoses_icd", "noteevents"],
    )
    samples = dataset.set_task(MIMIC3ICD9CategoryClassification())

Codes in the ICD-9 symptoms range (780-799) belong to none of the 19
categories and give no label. The category ranges are the standard ICD-9
chapters, not taken from the paper, which lists only the category names.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

from pyhealth.processors import MultiLabelProcessor, TextProcessor

from ..data import Patient
from .base_task import BaseTask

# The 19 categories, in the order of the paper's Table 4. Each numeric range is
# an inclusive range of three-digit ICD-9 codes. The last two categories are
# the E (external causes) and V (supplementary) code families.
ICD9_CATEGORY_RANGES = (
    ("Infectious & parasitic", 1, 139),
    ("Neoplasms", 140, 239),
    ("Endocrine, nutritional and metabolic", 240, 279),
    ("Blood & blood-forming organs", 280, 289),
    ("Mental disorders", 290, 319),
    ("Nervous system", 320, 359),
    ("Sense organs", 360, 389),
    ("Circulatory system", 390, 459),
    ("Respiratory system", 460, 519),
    ("Digestive system", 520, 579),
    ("Genitourinary system", 580, 629),
    ("Pregnancy & childbirth complications", 630, 679),
    ("Skin & subcutaneous tissue", 680, 709),
    ("Musculoskeletal system & connective tissue", 710, 739),
    ("Congenital anomalies", 740, 759),
    ("Perinatal period conditions", 760, 779),
    ("Injury and poisoning", 800, 999),
)
EXTERNAL_CAUSES = "External causes of injury"
SUPPLEMENTARY = "Supplementary"

ICD9_CATEGORIES = tuple(name for name, _, _ in ICD9_CATEGORY_RANGES) + (
    EXTERNAL_CAUSES,
    SUPPLEMENTARY,
)


def icd9_to_category(code: Any) -> str | None:
    """Map an ICD-9 code to one of the paper's 19 top-level categories.

    MIMIC-III stores ICD-9 codes without the dot (for example ``"4019"`` for
    401.9). Codes in the symptoms and ill-defined range (780-799) belong to
    none of the 19 categories and return ``None``, as do empty or malformed
    codes.

    Args:
        code: The ICD-9 code, such as ``"4019"``, ``"V1052"`` or ``"E8788"``.

    Returns:
        The category name, or ``None`` if the code has no category.

    Examples:
        >>> icd9_to_category("4019")
        'Circulatory system'
        >>> icd9_to_category("V1052")
        'Supplementary'
        >>> icd9_to_category("78650") is None
        True
    """
    if code is None:
        return None
    code = str(code).strip().upper()
    if not code:
        return None
    if code[0] == "V":
        return SUPPLEMENTARY
    if code[0] == "E":
        return EXTERNAL_CAUSES
    prefix = code[:3]
    if not prefix.isdigit():
        return None
    value = int(prefix)
    for name, low, high in ICD9_CATEGORY_RANGES:
        if low <= value <= high:
            return name
    return None


class MIMIC3ICD9CategoryClassification(BaseTask):
    """Classify MIMIC-III discharge summaries into top-level ICD-9 categories.

    Works with :class:`~pyhealth.datasets.MIMIC3Dataset` loaded with the
    ``diagnoses_icd`` and ``noteevents`` tables.

    Each admission with a matching note and at least one categorised diagnosis
    becomes one sample: the note text is the input and the set of the paper's
    19 ICD-9 categories present in the admission is the multilabel target.

    The defaults follow the clinical experiment in the paper: discharge
    summaries only, labelled from diagnosis codes. The arguments change which
    text is used, for ablations.

    Args:
        note_categories: Note ``category`` values to keep (case-insensitive).
            Notes from the same admission are joined with a space.
        min_words: Skip admissions whose text has fewer words than this.
        max_words: Keep only the first ``max_words`` words of the text. ``None``
            keeps all of it.

    Attributes:
        task_name: ``"mimic3_icd9_category_classification"``.
        input_schema: ``{"text": TextProcessor}``. The input is the note text,
            as one string.
        output_schema: ``{"categories": MultiLabelProcessor}``. The labels are
            fixed to :data:`ICD9_CATEGORIES`, so the target is always a
            19-dimensional multi-hot vector, in the order of the paper's
            Table 4.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import MIMIC3ICD9CategoryClassification
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "noteevents"],
        ... )  # doctest: +SKIP
        >>> task = MIMIC3ICD9CategoryClassification(max_words=1000)
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "mimic3_icd9_category_classification"
    input_schema: ClassVar[dict[str, Any]] = {"text": TextProcessor}
    # Fixing the labels keeps the target 19-dimensional, in the order of the
    # paper's Table 4, even on a small subset that lacks some categories.
    output_schema: ClassVar[dict[str, Any]] = {
        "categories": (MultiLabelProcessor, {"labels": list(ICD9_CATEGORIES)})
    }

    def __init__(
        self,
        note_categories: Sequence[str] = ("Discharge summary",),
        min_words: int = 1,
        max_words: int | None = None,
    ) -> None:
        """Initialize the task configuration.

        Args:
            note_categories: Note ``category`` values to keep.
            min_words: Minimum number of words a sample's text must have.
            max_words: Maximum number of words kept, or ``None`` for no cap.

        Raises:
            ValueError: If ``note_categories`` is empty, ``min_words`` is
                negative, or ``max_words`` is not positive.

        Examples:
            >>> task = MIMIC3ICD9CategoryClassification(min_words=50)
            >>> task.min_words
            50
        """
        super().__init__()
        if not note_categories:
            raise ValueError("note_categories must not be empty.")
        if min_words < 0:
            raise ValueError("min_words must be non-negative.")
        if max_words is not None and max_words <= 0:
            raise ValueError("max_words must be positive or None.")
        self.note_categories = tuple(c.strip().lower() for c in note_categories)
        self.min_words = min_words
        self.max_words = max_words

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Build one sample per admission from a patient's events.

        Args:
            patient: A patient with ``admissions``, ``diagnoses_icd`` and
                ``noteevents`` events.

        Returns:
            A list of samples with the keys ``patient_id``, ``visit_id``,
            ``text`` and ``categories``. Admissions with no matching note, too
            few words, or no categorised code are skipped.

        Examples:
            >>> task = MIMIC3ICD9CategoryClassification()
            >>> callable(task)
            True
        """
        samples: list[dict[str, Any]] = []
        for admission in patient.get_events(event_type="admissions"):
            hadm_id = admission.hadm_id
            text = self._admission_text(patient, hadm_id)
            categories = self._admission_categories(patient, hadm_id)
            if not text or not categories:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": hadm_id,
                    "text": text,
                    "categories": categories,
                }
            )
        return samples

    def _admission_text(self, patient: Patient, hadm_id: str) -> str:
        """Join the admission's selected notes and apply the word limits.

        Args:
            patient: The patient the admission belongs to.
            hadm_id: The admission id.

        Returns:
            The text, or an empty string if the admission has no selected note
            or fewer than ``min_words`` words.
        """
        notes = patient.get_events(
            event_type="noteevents", filters=[("hadm_id", "==", hadm_id)]
        )
        texts = []
        for note in notes:
            category = str(getattr(note, "category", "")).strip().lower()
            text = getattr(note, "text", None)
            if category in self.note_categories and text is not None:
                texts.append(str(text).strip())
        words = " ".join(texts).split()
        if len(words) < max(self.min_words, 1):
            return ""
        if self.max_words is not None:
            words = words[: self.max_words]
        return " ".join(words)

    def _admission_categories(self, patient: Patient, hadm_id: str) -> list[str]:
        """Return the admission's ICD-9 categories, in the paper's order.

        Args:
            patient: The patient the admission belongs to.
            hadm_id: The admission id.

        Returns:
            The category names present among the admission's diagnoses, ordered
            as in :data:`ICD9_CATEGORIES`. Empty if no code has a category.
        """
        diagnoses = patient.get_events(
            event_type="diagnoses_icd", filters=[("hadm_id", "==", hadm_id)]
        )
        present = {icd9_to_category(event.icd9_code) for event in diagnoses}
        return [name for name in ICD9_CATEGORIES if name in present]
