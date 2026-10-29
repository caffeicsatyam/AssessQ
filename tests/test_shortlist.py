import unittest

from shortlist import apply_shortlist_edit, parse_shortlist_edit


def assessment(name):
    return {"metadata": {"assessment_name": name, "test_types": [], "languages": []}, "document": ""}


class ShortlistEditTests(unittest.TestCase):
    def test_remove_and_add_command_uses_catalog_terms(self):
        drops, adds = parse_shortlist_edit("can you remove the typing assessment and add data entry assessment")
        self.assertEqual(drops, ["typing"])
        self.assertEqual(adds, ["data entry"])


    def test_direct_edit_removes_typing_without_reintroducing_it(self):
        current = [assessment("Typing (New)"), assessment("Java (New)")]
        retrieved = [assessment("Data Entry (New)"), assessment("Typing (New)")]
        updated = apply_shortlist_edit(current, ["typing"], ["data entry"], retrieved, limit=7)
        self.assertEqual(
            [item["metadata"]["assessment_name"] for item in updated],
            ["Data Entry (New)", "Java (New)"],
        )


    def test_addition_is_retained_when_shortlist_is_at_its_limit(self):
        current = [assessment("First"), assessment("Second")]
        updated = apply_shortlist_edit(current, [], ["data entry"], [assessment("Data Entry (New)")], limit=2)
        self.assertEqual(
            [item["metadata"]["assessment_name"] for item in updated],
            ["Data Entry (New)", "First"],
        )

    def test_existing_requested_addition_does_not_reorder_or_expand_the_list(self):
        current = [assessment("Typing (New)"), assessment("Data Entry (New)"), assessment("Java (New)")]
        retrieved = [assessment("Data Entry Ten Key Split Screen")]
        updated = apply_shortlist_edit(current, ["typing"], ["data entry"], retrieved, limit=7)
        self.assertEqual(
            [item["metadata"]["assessment_name"] for item in updated],
            ["Data Entry (New)", "Java (New)"],
        )
