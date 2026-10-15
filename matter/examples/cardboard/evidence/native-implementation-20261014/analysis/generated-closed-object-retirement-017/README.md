# Closed generated build-cache retirement 017

This receipt records only closed generated CMake `.o` and `.d` intermediates under `CMakeFiles`, plus `.air` intermediate compiler outputs, in completed build directories 139 and 141. The full pre-unlink inventory was captured in the tool session store and transcript before any unlink, then revalidated against path, SHA-256, size, allocated blocks, device, and inode immediately before removal. The persisted `before.json` materializes that exact inventory.

No source, CMake metadata, compiler database, `.metallib`, `.dylib`, executable, scientific input, or run output was removed. Before and after checks revalidated each source-build receipt, all recorded final artifacts and metadata, and complete-build `lsof` scans. Both builds had no matching process and no open files at the deletion check.

Removed 616 files: 68,277,907 logical bytes and 69,550,080 allocated bytes. Available space rose from 121,716,736 to 189,186,048 bytes (+67,469,312). The exact file-level evidence and final artifact hashes are in `before.json`; the after-state checks are in `after.json`.

The pre-unlink inventory digest is `43faae528f67e7751e14fa5927502ec15a75f695632d53c5588af8eb6560f6ad`. The session-store compressed transfer digest was `2e8e40ee1b1a0f0edef1d4c1fa627baa1819aadba7d85553f8cf3384b0e1c016`.
