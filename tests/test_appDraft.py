"""
Tests for appDraft.py — Movie Mood Recommender.

Unit tests run without an API key.
Integration tests (marked @pytest.mark.integration) call the real HF Inference
API and require HF_TOKEN to be set.
"""

import json
import os
import sys
import pytest
from dotenv import load_dotenv

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Load .env so integration tests can pick up the token locally
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from appDraft import (
    build_messages,
    parse_recommendation,
    format_recommendation,
    SYSTEM_PROMPT,
    FEW_SHOT_EXAMPLES,
)


# ---------------------------------------------------------------------------
# Unit tests (no API key needed)
# ---------------------------------------------------------------------------

class TestBuildMessages:
    """Verify the prompt assembly logic."""

    SAMPLE_ANSWERS = {
        "mood": "adventurous",
        "genres": ["Action", "Sci-Fi"],
        "pace": "fast",
        "spectacle_vs_story": "spectacle",
        "familiar_vs_new": "new",
        "open_ended": "I loved Interstellar.",
    }

    def test_returns_list(self):
        msgs = build_messages(self.SAMPLE_ANSWERS)
        assert isinstance(msgs, list)

    def test_first_message_is_system(self):
        msgs = build_messages(self.SAMPLE_ANSWERS)
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == SYSTEM_PROMPT

    def test_few_shot_pairs_present(self):
        msgs = build_messages(self.SAMPLE_ANSWERS)
        # 1 system + 2 user/assistant pairs + 1 final user = 6 messages
        assert len(msgs) == 1 + len(FEW_SHOT_EXAMPLES) * 2 + 1

    def test_last_message_is_user(self):
        msgs = build_messages(self.SAMPLE_ANSWERS)
        assert msgs[-1]["role"] == "user"

    def test_user_answers_embedded_in_last_message(self):
        msgs = build_messages(self.SAMPLE_ANSWERS)
        last = msgs[-1]["content"]
        assert "adventurous" in last
        assert "Interstellar" in last

    def test_few_shot_assistant_messages_are_json(self):
        msgs = build_messages(self.SAMPLE_ANSWERS)
        for msg in msgs:
            if msg["role"] == "assistant":
                data = json.loads(msg["content"])
                assert "recommended_movie" in data


class TestParseRecommendation:
    """Verify JSON parsing of model output."""

    VALID_JSON = json.dumps({
        "why": "Tense and clever.",
        "user_mentioned_movies": ["Skinamarink"],
        "recommended_movie": "A Quiet Place",
        "year": 2018,
    })

    def test_valid_json_returns_dict(self):
        result = parse_recommendation(self.VALID_JSON)
        assert isinstance(result, dict)
        assert result["recommended_movie"] == "A Quiet Place"
        assert result["year"] == 2018

    def test_invalid_json_returns_none(self):
        assert parse_recommendation("not json at all") is None

    def test_empty_string_returns_none(self):
        assert parse_recommendation("") is None

    def test_none_input_returns_none(self):
        assert parse_recommendation(None) is None


class TestFormatRecommendation:
    """Verify markdown formatting of a parsed recommendation."""

    FULL_DATA = {
        "recommended_movie": "A Quiet Place",
        "year": 2018,
        "why": "Tense and clever.",
        "user_mentioned_movies": ["Skinamarink"],
    }

    def test_contains_title_and_year(self):
        out = format_recommendation(self.FULL_DATA)
        assert "A Quiet Place" in out
        assert "2018" in out

    def test_contains_why(self):
        out = format_recommendation(self.FULL_DATA)
        assert "Tense and clever." in out

    def test_contains_mentioned_movies(self):
        out = format_recommendation(self.FULL_DATA)
        assert "Skinamarink" in out

    def test_missing_optional_fields(self):
        data = {"recommended_movie": "Alien"}
        out = format_recommendation(data)
        assert "Alien" in out

    def test_empty_mentioned_list(self):
        data = {**self.FULL_DATA, "user_mentioned_movies": []}
        out = format_recommendation(data)
        assert "Based on your mention of" not in out


# ---------------------------------------------------------------------------
# Integration tests (require HF_TOKEN)
# ---------------------------------------------------------------------------

def _hf_token_available() -> bool:
    return bool(os.getenv("HF_TOKEN"))


@pytest.mark.integration
@pytest.mark.skipif(not _hf_token_available(), reason="HF_TOKEN not set")
class TestCallLLM:
    """Call the real HF Inference API via call_llm and validate the raw output."""

    def test_returns_valid_json(self):
        from appDraft import call_llm

        user_answers = {
            "mood": "happy",
            "genres": ["Comedy"],
            "pace": "medium",
            "spectacle_vs_story": "balanced",
            "familiar_vs_new": "familiar",
            "open_ended": "I enjoyed The Grand Budapest Hotel.",
        }
        messages = build_messages(user_answers)
        token = os.getenv("HF_TOKEN")

        raw = call_llm(messages, token)
        data = json.loads(raw)

        assert "recommended_movie" in data, f"Missing 'recommended_movie' in: {raw}"
        assert "year" in data, f"Missing 'year' in: {raw}"
        assert "why" in data, f"Missing 'why' in: {raw}"
        assert isinstance(data["year"], int), f"'year' should be int, got: {type(data['year'])}"
        assert data["recommended_movie"] != "The Grand Budapest Hotel", (
            "Model should not recommend the same movie the user mentioned"
        )


@pytest.mark.integration
@pytest.mark.skipif(not _hf_token_available(), reason="HF_TOKEN not set")
class TestRecommendMovie:
    """End-to-end test through recommend_movie."""

    def _call_recommend(self):
        from appDraft import recommend_movie
        return recommend_movie(
            mood="scary",
            genres=["Horror", "Thriller"],
            pace="fast",
            spectacle_vs_story="story",
            familiar_vs_new="new",
            open_ended="I like tense movies with clever twists.",
        )

    def test_returns_non_empty_string(self):
        result = self._call_recommend()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_movie_title(self):
        result = self._call_recommend()
        assert ("##" in result) or ("Model response" in result)

    def test_output_has_year(self):
        import re
        result = self._call_recommend()
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", result))
        is_fallback = "Model response" in result
        assert has_year or is_fallback
