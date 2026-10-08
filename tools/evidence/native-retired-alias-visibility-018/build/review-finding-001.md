# Attempt 1 review finding retained before correction

Attempt 1 keyed retirement only by `instance.identity.w == 22` and then required every matching instance to have organ semantic 51010. Stable IDs are semantic-scoped, so this rejects valid unrelated instances. The actual step-0 pack contains these identities:

- Bone: `(51004, 22, 137, 22)`
- Muscle surface: `(51005, 22, UINT_MAX, 22)`
- Retired organ alias: `(51010, 22, 20, 22)`

The corrected predicate must match `(semantic == 51010, stable ID == 22)` and leave the bone and muscle masks unchanged. Attempt 1 source/build/test evidence remains intact at `native-retired-alias-visibility-018/` and build directory `numi-human-retired-alias-visibility-build-018/`.

Attempt 1 references: source delta SHA256 `3b3620b2d09622c16827b079cd4a473161751e2cd1059598d950b06016909bca`; build pins SHA256 `c6ec30cdd4884111a5a37331ab03cee3936075caa5885836d6c468f663e4dcbb`; viewer SHA256 `b3c5cb47a4d58d234e52f5e012a31c827c12804f6dd41d3ebdd60cd7badaeae2`; CPU regression passed before the review finding. The native binary was not run.
