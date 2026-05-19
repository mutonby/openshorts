# `openshorts/prompts/`

Externalized Gemini prompt templates. Convention: one `.md` file per
prompt, loaded by name via the package-level loader (planned). Supports
`{{var}}` placeholder substitution at load time. Keeping prompts as files
(rather than string literals buried inside handlers) lets non-engineers
iterate on them and makes prompt-history diffs readable.

Editing-domain prompts that need heavy `f"..."` interpolation may stay in
`openshorts/editing/prompts.py` instead.
