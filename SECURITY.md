# Security policy

This is a small demonstration service and its fixture data is synthetic. Do not
send secrets, credentials, payment details, or personal information to it.

## Reporting a vulnerability

Please do not open a public issue for a suspected security vulnerability. Once this
repository is published, use its GitHub **Security** tab and **Report a
vulnerability** form. If private reporting is not enabled yet, contact the
repository maintainer through their GitHub profile before sharing details. Include
reproduction steps, impact, and a suggested fix. There is no production support
SLA for this demo.

The service does not include authentication, authorization, rate limiting, or
TLS termination. Put it behind an appropriately secured gateway before any
non-local use.
