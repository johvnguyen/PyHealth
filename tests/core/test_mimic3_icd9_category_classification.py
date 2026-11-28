"""Tests for the MIMIC-III ICD-9 category task, on hand-built fake patients.

No MIMIC data is read: patients are plain objects that answer ``get_events``
the way :class:`pyhealth.data.Patient` does.
"""

import unittest
from types import SimpleNamespace

from pyhealth.datasets import create_sample_dataset
from pyhealth.tasks import (
    ICD9_CATEGORIES,
    MIMIC3ICD9CategoryClassification,
    icd9_to_category,
)


class FakePatient:
    """Events by type, filtered on attributes like ``Patient.get_events``."""

    def __init__(self, events, patient_id="p1"):
        self.patient_id = patient_id
        self.events = events

    def get_events(self, event_type=None, filters=None, **kwargs):
        events = list(self.events.get(event_type, []))
        for attr, _, value in filters or []:
            events = [e for e in events if getattr(e, attr) == value]
        return events


def admission(hadm_id):
    return SimpleNamespace(hadm_id=hadm_id)


def note(hadm_id, text, category="Discharge summary"):
    return SimpleNamespace(hadm_id=hadm_id, text=text, category=category)


def diagnosis(hadm_id, code):
    return SimpleNamespace(hadm_id=hadm_id, icd9_code=code)


def make_patient(admissions, notes, diagnoses, patient_id="p1"):
    return FakePatient(
        {
            "admissions": admissions,
            "noteevents": notes,
            "diagnoses_icd": diagnoses,
        },
        patient_id=patient_id,
    )


class TestIcd9ToCategory(unittest.TestCase):
    def test_there_are_19_categories(self):
        self.assertEqual(len(ICD9_CATEGORIES), 19)
        self.assertEqual(len(set(ICD9_CATEGORIES)), 19)

    def test_common_codes(self):
        self.assertEqual(icd9_to_category("4019"), "Circulatory system")
        self.assertEqual(icd9_to_category("25000"), ICD9_CATEGORIES[2])
        self.assertEqual(icd9_to_category("486"), "Respiratory system")
        self.assertEqual(icd9_to_category("V1052"), "Supplementary")
        self.assertEqual(icd9_to_category("E8788"), "External causes of injury")

    def test_range_boundaries(self):
        self.assertEqual(icd9_to_category("001"), "Infectious & parasitic")
        self.assertEqual(icd9_to_category("139"), "Infectious & parasitic")
        self.assertEqual(icd9_to_category("140"), "Neoplasms")
        self.assertEqual(icd9_to_category("359"), "Nervous system")
        self.assertEqual(icd9_to_category("360"), "Sense organs")
        self.assertEqual(icd9_to_category("800"), "Injury and poisoning")
        self.assertEqual(icd9_to_category("999"), "Injury and poisoning")

    def test_symptom_codes_have_no_category(self):
        self.assertIsNone(icd9_to_category("780"))
        self.assertIsNone(icd9_to_category("78650"))
        self.assertIsNone(icd9_to_category("799"))

    def test_bad_input_returns_none(self):
        for bad in (None, "", "   ", "nan", "ABC", "0"):
            self.assertIsNone(icd9_to_category(bad), msg=repr(bad))

    def test_case_and_whitespace_are_ignored(self):
        self.assertEqual(icd9_to_category(" v1052 "), "Supplementary")
        self.assertEqual(icd9_to_category(" 4019"), "Circulatory system")


class TestTaskSamples(unittest.TestCase):
    def test_one_sample_per_admission_with_ordered_labels(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "chest pain and diabetes")],
            [diagnosis("1", "V1052"), diagnosis("1", "4019"), diagnosis("1", "25000")],
        )
        samples = MIMIC3ICD9CategoryClassification()(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["visit_id"], "1")
        self.assertEqual(sample["text"], "chest pain and diabetes")
        # labels follow the paper's category order, whatever the code order
        self.assertEqual(
            sample["categories"],
            [
                "Endocrine, nutritional and metabolic",
                "Circulatory system",
                "Supplementary",
            ],
        )

    def test_repeated_codes_give_one_label(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "text")],
            [diagnosis("1", "4019"), diagnosis("1", "4280")],
        )
        samples = MIMIC3ICD9CategoryClassification()(patient)
        self.assertEqual(samples[0]["categories"], ["Circulatory system"])

    def test_only_discharge_summaries_by_default(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "nursing text", category="Nursing")],
            [diagnosis("1", "4019")],
        )
        self.assertEqual(MIMIC3ICD9CategoryClassification()(patient), [])

    def test_note_categories_option_is_case_insensitive(self):
        patient = make_patient(
            [admission("1")],
            [
                note("1", "discharge text"),
                note("1", "nursing text", category="Nursing"),
            ],
            [diagnosis("1", "4019")],
        )
        task = MIMIC3ICD9CategoryClassification(
            note_categories=("discharge summary", "NURSING")
        )
        self.assertEqual(task(patient)[0]["text"], "discharge text nursing text")

    def test_notes_of_the_same_admission_are_joined(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "first part"), note("1", "addendum")],
            [diagnosis("1", "4019")],
        )
        samples = MIMIC3ICD9CategoryClassification()(patient)
        self.assertEqual(samples[0]["text"], "first part addendum")

    def test_each_admission_uses_only_its_own_notes_and_codes(self):
        patient = make_patient(
            [admission("1"), admission("2")],
            [note("1", "heart"), note("2", "lungs")],
            [diagnosis("1", "4019"), diagnosis("2", "486")],
        )
        samples = MIMIC3ICD9CategoryClassification()(patient)
        self.assertEqual(len(samples), 2)
        by_visit = {s["visit_id"]: s for s in samples}
        self.assertEqual(by_visit["1"]["categories"], ["Circulatory system"])
        self.assertEqual(by_visit["2"]["categories"], ["Respiratory system"])
        self.assertEqual(by_visit["2"]["text"], "lungs")

    def test_admission_without_a_note_is_skipped(self):
        patient = make_patient([admission("1")], [], [diagnosis("1", "4019")])
        self.assertEqual(MIMIC3ICD9CategoryClassification()(patient), [])

    def test_admission_without_a_categorised_code_is_skipped(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "text")],
            [diagnosis("1", "78650"), diagnosis("1", None)],
        )
        self.assertEqual(MIMIC3ICD9CategoryClassification()(patient), [])

    def test_blank_or_missing_note_text_is_skipped(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "   "), note("1", None)],
            [diagnosis("1", "4019")],
        )
        self.assertEqual(MIMIC3ICD9CategoryClassification()(patient), [])

    def test_max_words_keeps_the_start_of_the_text(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "one two three four five")],
            [diagnosis("1", "4019")],
        )
        samples = MIMIC3ICD9CategoryClassification(max_words=3)(patient)
        self.assertEqual(samples[0]["text"], "one two three")

    def test_min_words_skips_short_notes(self):
        patient = make_patient(
            [admission("1")],
            [note("1", "one two")],
            [diagnosis("1", "4019")],
        )
        self.assertEqual(MIMIC3ICD9CategoryClassification(min_words=3)(patient), [])
        self.assertEqual(len(MIMIC3ICD9CategoryClassification(min_words=2)(patient)), 1)

    def test_patient_without_admissions_gives_no_samples(self):
        patient = make_patient([], [], [])
        self.assertEqual(MIMIC3ICD9CategoryClassification()(patient), [])


class TestTaskConfiguration(unittest.TestCase):
    def test_invalid_arguments_raise(self):
        with self.assertRaises(ValueError):
            MIMIC3ICD9CategoryClassification(note_categories=())
        with self.assertRaises(ValueError):
            MIMIC3ICD9CategoryClassification(min_words=-1)
        with self.assertRaises(ValueError):
            MIMIC3ICD9CategoryClassification(max_words=0)

    def test_schemas(self):
        task = MIMIC3ICD9CategoryClassification()
        self.assertEqual(task.task_name, "mimic3_icd9_category_classification")
        self.assertEqual(list(task.input_schema), ["text"])
        self.assertEqual(list(task.output_schema), ["categories"])

    def test_samples_load_into_a_sample_dataset(self):
        patients = [
            make_patient(
                [admission("1")],
                [note("1", "heart failure")],
                [diagnosis("1", "4280")],
                patient_id="p1",
            ),
            make_patient(
                [admission("2")],
                [note("2", "pneumonia and diabetes")],
                [diagnosis("2", "486"), diagnosis("2", "25000")],
                patient_id="p2",
            ),
        ]
        task = MIMIC3ICD9CategoryClassification()
        samples = [s for p in patients for s in task(p)]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="test_icd9_categories",
        )
        self.assertEqual(len(dataset), 2)
        self.assertIn("text", dataset[0])
        # the target is always 19-dimensional, even though these two
        # samples only use three of the categories
        self.assertEqual(len(dataset[0]["categories"]), 19)
        self.assertEqual(len(dataset[1]["categories"]), 19)
        self.assertEqual(float(dataset[0]["categories"].sum()), 1.0)
        self.assertEqual(float(dataset[1]["categories"].sum()), 2.0)


if __name__ == "__main__":
    unittest.main()
