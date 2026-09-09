#!/usr/bin/env bash
# One-command push helper.
#
# Prerequisite (choose one):
#   A) Create an empty repo named cdi-agent-collective on github.com
#      (no README, no license, no .gitignore), then run this script.
#   B) Provide a PAT with repo scope and let the script create the repo:
#        GITHUB_TOKEN=ghp_xxx ./scripts/push_to_github.sh
#
# SSH is routed through port 443 (see ~/.ssh/config) because port 22 is
# unreachable from this machine.

set -euo pipefail

USER_NAME="liuiopshaw"
REPO_NAME="cdi-agent-collective"
REMOTE="git@github.com:${USER_NAME}/${REPO_NAME}.git"

cd "$(dirname "$0")/.."

if [ -n "${GITHUB_TOKEN:-}" ]; then
  echo "Creating repository ${USER_NAME}/${REPO_NAME} via API..."
  curl -sf -X POST \
    -H "Authorization: token ${GITHUB_TOKEN}" \
    -H "Accept: application/vnd.github+json" \
    https://api.github.com/user/repos \
    -d "{\"name\": \"${REPO_NAME}\", \"private\": false,
         \"description\": \"Mechanism-certification multi-agent framework for CDI electrode research\"}" \
    > /dev/null || echo "Repo may already exist; continuing."
fi

if git remote | grep -q "^origin$"; then
  git remote set-url origin "${REMOTE}"
else
  git remote add origin "${REMOTE}"
fi

echo "Pushing to ${REMOTE} ..."
git push -u origin main
echo "Done: https://github.com/${USER_NAME}/${REPO_NAME}"
