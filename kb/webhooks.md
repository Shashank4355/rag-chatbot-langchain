---
title: Webhooks
updated: 2026-10-05
---
# Webhooks

Webhooks send an HTTPS POST to your URL when something happens in a workspace. They are available on Team, Business and Enterprise.

## Events

file.created, file.updated, file.deleted, file.shared, user.added and user.removed.

## Verifying a webhook

Each request has an X-Kitebox-Signature header. It is an HMAC-SHA256 of the raw request body, using your webhook secret as the key, encoded in hex. Compare it with your own HMAC before trusting the request.

## Retries

Your endpoint must reply with a 2xx status within 10 seconds. If it doesn't, Kitebox retries up to 8 times with growing gaps, over about 24 hours. After that the delivery is dropped. A webhook that fails every delivery for 3 days in a row is turned off and the admin gets an email.
