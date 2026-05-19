"""
Characterization tests for translate.SUPPORTED_LANGUAGES and
get_supported_languages.

Locks in the public surface so the restructure can't accidentally
drop or rename a language code.
"""
from app.integrations.elevenlabs import SUPPORTED_LANGUAGES, get_supported_languages


# A minimal canonical set we want the API to keep advertising.
_EXPECTED_CORE = {
    "en", "es", "fr", "de", "it", "pt", "pl", "hi", "ja", "ko",
    "zh", "ar", "ru", "tr", "nl", "sv", "id", "vi", "th",
}


def test_supported_languages_includes_core_codes():
    missing = _EXPECTED_CORE - set(SUPPORTED_LANGUAGES.keys())
    assert not missing, f"SUPPORTED_LANGUAGES dropped: {missing}"


def test_supported_languages_values_are_human_names():
    assert SUPPORTED_LANGUAGES["en"] == "English"
    assert SUPPORTED_LANGUAGES["es"] == "Spanish"
    assert SUPPORTED_LANGUAGES["zh"] == "Chinese"


def test_get_supported_languages_returns_a_copy():
    result = get_supported_languages()
    assert result == SUPPORTED_LANGUAGES
    # Mutating the copy must not poison the global.
    result["xx"] = "TestLang"
    assert "xx" not in SUPPORTED_LANGUAGES


def test_supported_languages_has_no_empty_values():
    for code, name in SUPPORTED_LANGUAGES.items():
        assert code, "empty language code"
        assert isinstance(name, str) and name.strip(), f"empty name for {code}"
