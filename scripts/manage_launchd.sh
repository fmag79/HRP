#!/usr/bin/env bash
#
# Manage HRP launchd jobs: install, uninstall, status, reload.
#
# Usage:
#   scripts/manage_launchd.sh install    # symlink plists to ~/Library/LaunchAgents/ and load
#   scripts/manage_launchd.sh uninstall  # unload and remove symlinks
#   scripts/manage_launchd.sh status     # show loaded HRP jobs
#   scripts/manage_launchd.sh reload     # uninstall + install
#

set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PLIST_DIR="${REPO_DIR}/launchd"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
OLD_DAEMON_LABEL="com.hrp.scheduler"

install() {
    echo "Installing HRP launchd jobs..."

    # Ensure target directory exists
    mkdir -p "${LAUNCH_AGENTS_DIR}"

    # Ensure log directory exists
    mkdir -p "${HOME}/hrp-data/logs"

    # Unload old daemon if present
    if launchctl list 2>/dev/null | grep -q "${OLD_DAEMON_LABEL}"; then
        echo "Unloading old daemon: ${OLD_DAEMON_LABEL}"
        launchctl bootout "gui/$(id -u)/${OLD_DAEMON_LABEL}" 2>/dev/null || \
            launchctl unload "${LAUNCH_AGENTS_DIR}/${OLD_DAEMON_LABEL}.plist" 2>/dev/null || true
    fi
    # Remove old daemon plist symlink/file
    rm -f "${LAUNCH_AGENTS_DIR}/${OLD_DAEMON_LABEL}.plist"

    # Install each plist
    local count=0
    for plist in "${PLIST_DIR}"/com.hrp.*.plist; do
        [ -f "$plist" ] || continue
        local basename
        basename="$(basename "$plist")"
        local target="${LAUNCH_AGENTS_DIR}/${basename}"

        # Remove existing symlink/file
        rm -f "${target}"

        # Create symlink
        ln -s "$plist" "${target}"

        # Load the job
        launchctl bootstrap "gui/$(id -u)" "${target}" 2>/dev/null || \
            launchctl load "${target}" 2>/dev/null || true

        echo "  Installed: ${basename}"
        count=$((count + 1))
    done

    echo "Installed ${count} jobs."
    echo ""
    echo "Verify with: scripts/manage_launchd.sh status"
}

uninstall() {
    echo "Uninstalling HRP launchd jobs..."

    local count=0
    for plist in "${LAUNCH_AGENTS_DIR}"/com.hrp.*.plist; do
        [ -f "$plist" ] || [ -L "$plist" ] || continue
        local basename
        basename="$(basename "$plist")"
        local label="${basename%.plist}"

        # Unload the job
        launchctl bootout "gui/$(id -u)/${label}" 2>/dev/null || \
            launchctl unload "$plist" 2>/dev/null || true

        # Remove symlink/file
        rm -f "$plist"

        echo "  Removed: ${basename}"
        count=$((count + 1))
    done

    echo "Removed ${count} jobs."
}

status() {
    echo "HRP launchd jobs:"
    echo ""

    local found=0
    while IFS= read -r line; do
        if echo "$line" | grep -q "com.hrp\."; then
            echo "  $line"
            found=$((found + 1))
        fi
    done < <(launchctl list 2>/dev/null)

    if [ "$found" -eq 0 ]; then
        echo "  No HRP jobs loaded."
        echo ""
        echo "  Install with: scripts/manage_launchd.sh install"
    else
        echo ""
        echo "  ${found} jobs loaded."
    fi
}

reload() {
    echo "Reloading HRP launchd jobs..."
    echo ""
    uninstall
    echo ""
    install
}

# Main
case "${1:-}" in
    install)
        install
        ;;
    uninstall)
        uninstall
        ;;
    status)
        status
        ;;
    reload)
        reload
        ;;
    *)
        echo "Usage: $0 {install|uninstall|status|reload}"
        echo ""
        echo "Commands:"
        echo "  install    Symlink plists to ~/Library/LaunchAgents/ and load all jobs"
        echo "  uninstall  Unload all jobs and remove symlinks"
        echo "  status     Show which HRP jobs are loaded"
        echo "  reload     Uninstall + install (picks up plist changes)"
        exit 1
        ;;
esac
