import unittest

from agent import ConversationalRecommender


class ConversationalRoutingTests(unittest.TestCase):
    def test_recommendation_decomposition_uses_only_retrieval_facts(self):
        intent = ConversationalRecommender._decompose_intent(
            {
                "action": "recommend",
                "role": "Data Scientist",
                "seniority": "Mid-Professional",
                "skills": ["Python", "R", "SQL", "Statistics", "Machine Learning"],
                "query": "ignore this fallback",
            },
            "all",
        )
        self.assertEqual(intent.action, "recommend")
        self.assertEqual(
            intent.retrieval_query,
            "Data Scientist Mid-Professional Python R SQL Statistics Machine Learning",
        )

    def test_definition_overrides_incorrect_recommendation_action(self):
        intent = ConversationalRecommender._decompose_intent(
            {"action": "recommend", "role": "SAP Developer", "skills": ["ABAP"], "query": "SAP ABAP"},
            "What is SAP ABAP?",
        )
        self.assertEqual(intent.action, "answer")

    def test_missing_decomposition_falls_back_to_model_query(self):
        intent = ConversationalRecommender._decompose_intent(
            {"action": "recommend", "query": "python machine learning"}, "recommend something"
        )
        self.assertEqual(intent.retrieval_query, "python machine learning")
