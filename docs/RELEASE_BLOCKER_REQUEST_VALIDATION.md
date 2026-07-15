# Release blocker: authenticated upload request validation

Live beta returned HTTP 403 for authenticated `POST /api/vectorize` when the browser retained a valid session and request cookie but did not attach the PPC-2 request header.

The fix preserves fail-closed CSRF behavior:

- exact same-origin browser requests may mirror the request cookie into the expected header;
- the existing identity middleware still verifies the token against the active SQLite session;
- cross-origin, missing-origin, missing-cookie, expired-session, and explicitly invalid-header requests remain rejected;
- `/api/vectorize` is not made anonymous and no security check is removed.
