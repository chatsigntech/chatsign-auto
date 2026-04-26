"""Whitelist regex patterns shared by Phase 3 publish-related modules.

These are defense-in-depth against shell injection in the SCP remote
spec (`user@host:/path`). The remote shell on the target evaluates the
path component, so even though we use `subprocess.run` (no local shell),
arbitrary characters in host/user/path can be exploited.
"""
import re

NAME_OK = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")   # human-friendly server label
HOST_OK = re.compile(r"^[A-Za-z0-9.-]+$")          # IPv4 / hostname (no IPv6 brackets)
USER_OK = re.compile(r"^[A-Za-z0-9_.-]+$")         # POSIX-friendly username
PATH_OK = re.compile(r"^/[A-Za-z0-9/_.-]+$")       # absolute path only
