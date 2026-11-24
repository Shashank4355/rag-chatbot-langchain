---
title: Desktop sync error codes
updated: 2026-08-30
---
# Desktop sync error codes

The desktop app runs on Windows 10 and 11 and on macOS 13 or later. There is no Linux desktop app; on Linux, use the web app or the API.

## KB-101: path too long

Windows can't handle file paths longer than 260 characters. Shorten folder or file names, or move the Kitebox folder closer to the drive root, for example C:\Kitebox.

## KB-204: file in use

Another program has the file open and locked. Close the program and the file syncs within a minute.

## KB-309: storage full

The workspace has used all of its storage. Delete files, empty Trash, or ask an admin to add users or upgrade the plan.

## KB-412: clock out of sync

Your computer's clock is more than 5 minutes different from the real time, so Kitebox rejects its requests. Turn on automatic time setting in your system settings and restart the app.

## KB-500: server error

Something failed on Kitebox's side. The app retries on its own. If the error lasts more than an hour, check status.kitebox.example and contact support with the log file from Help > Export logs.
