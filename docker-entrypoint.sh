#!/bin/bash
# Default CMD for the image (Sprint 4 / Tekniska). Not used by local dev
# workflows — `docker compose run` / `make shell` always pass an explicit
# command, which replaces this entirely. This only runs when the container
# is started with no override, i.e. as a persistent service — a RunPod Pod.
#
# RunPod injects the account's registered SSH public key via $PUBLIC_KEY at
# pod boot. Write it to authorized_keys here rather than baking any key into
# the image itself.
set -e

if [ -n "$PUBLIC_KEY" ]; then
    echo "$PUBLIC_KEY" > /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
fi

exec /usr/sbin/sshd -D
