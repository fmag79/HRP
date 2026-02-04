"""CLI for managing dashboard users."""

from __future__ import annotations

import argparse
import getpass
import sys

from hrp.dashboard.auth import (
    AuthConfig,
    add_user,
    load_users,
    remove_user,
)


def cmd_add_user(args: argparse.Namespace) -> None:
    """Add a new user."""
    config = AuthConfig.from_env()
    password = args.password or getpass.getpass("Password: ")
    add_user(config.users_file, args.username, args.email, args.name, password)
    print(f"Added user: {args.username}")


def cmd_remove_user(args: argparse.Namespace) -> None:
    """Remove a user."""
    config = AuthConfig.from_env()
    remove_user(config.users_file, args.username)
    print(f"Removed user: {args.username}")


def cmd_list_users(args: argparse.Namespace) -> None:
    """List all users."""
    config = AuthConfig.from_env()
    users = load_users(config.users_file)
    usernames = users["credentials"]["usernames"]
    if usernames:
        print("Configured users:")
        for username, info in usernames.items():
            print(f"  {username}: {info.get('name', '')} <{info.get('email', '')}>")
    else:
        print("No users configured")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="HRP Dashboard User Management")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # add-user command
    add_parser = subparsers.add_parser("add-user", help="Add a new user")
    add_parser.add_argument("--username", required=True, help="Unique username")
    add_parser.add_argument("--email", required=True, help="User email")
    add_parser.add_argument("--name", required=True, help="Display name")
    add_parser.add_argument(
        "--password", help="Password (prompted if not provided)"
    )
    add_parser.set_defaults(func=cmd_add_user)

    # remove-user command
    rm_parser = subparsers.add_parser("remove-user", help="Remove a user")
    rm_parser.add_argument("--username", required=True, help="Username to remove")
    rm_parser.set_defaults(func=cmd_remove_user)

    # list-users command
    list_parser = subparsers.add_parser("list-users", help="List all users")
    list_parser.set_defaults(func=cmd_list_users)

    args = parser.parse_args()
    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
