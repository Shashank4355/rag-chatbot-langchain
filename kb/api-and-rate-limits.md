---
title: API and rate limits
updated: 2026-10-05
---
# API and rate limits

## Versions

The current API is v2 at api.kitebox.example/v2. API v1 is deprecated and stops working on 31 March 2027. New integrations must use v2.

## Authentication

Create a personal access token under Settings > Developer > Tokens and send it as a Bearer token. Tokens can be limited to read-only. Admins on Business and Enterprise can create workspace tokens that aren't tied to one person.

## Rate limits

Limits are per token, per minute.

| Plan | API v2 | API v1 |
|---|---|---|
| Free | 60 requests | 30 requests |
| Team | 120 requests | 60 requests |
| Business | 600 requests | 60 requests |
| Enterprise | 1,200 requests | 60 requests |

When you go over the limit the API returns HTTP 429 with a Retry-After header in seconds. Wait that long before retrying.

## Uploads over the API

Files up to 150 MB can be uploaded in one request. Larger files must use the chunked upload endpoint with chunks between 5 MB and 100 MB.
