# Permission Modes

Permission modes control whether OctoSlave asks before modifying files or running commands.

---

## Modes

### `autonomous` (default)

Works without asking. Edits files, runs commands freely.

Best for: trusted workflows, automation, rapid prototyping.

### `controlled`

Asks before every file write, file edit, and bash command.

Best for: sensitive codebases, production systems, learning what the agent does.

### `supervised`

Asks before file edits only. Bash commands are auto-allowed.

Best for: situations where you want oversight on file changes but trust shell commands.

---

## Tools that trigger a permission prompt

| Tool | `autonomous` | `controlled` | `supervised` |
|------|-------------|-------------|-------------|
| `read_file` | auto | auto | auto |
| `glob`, `grep`, `list_dir` | auto | auto | auto |
| `web_search`, `web_fetch` | auto | auto | auto |
| `write_file` | auto | **asks** | **asks** |
| `edit_file` | auto | **asks** | **asks** |
| `bash` | auto | **asks** | auto |

---

## Usage

### At startup

```bash
octoslave --permission-mode controlled
octoslave run "fix the bug" --permission-mode supervised
```

### Mid-session

```bash
/permission                  # show current mode
/permission controlled       # switch to controlled
/permission autonomous       # switch back
/permission supervised       # supervised mode
```

### Set as default in config

```bash
octoslave config
# prompts for permission mode during setup

# or non-interactively:
octoslave config --permission-mode controlled
```

Config is saved to `~/.octoslave/config.json`.

---

## Permission prompt

When a tool requires permission, OctoSlave displays:

```
┌──────── Controlled Mode ────────┐
│  ⚠ Permission Required          │
│                                 │
│  ✏️  edit_file                  │
│                                 │
│  OctoSlave wants to:            │
│  edit file: src/main.py         │
│                                 │
│  Working directory: /my/project │
└─────────────────────────────────┘

Allow? (y)/n
```

Enter `y` / `yes` / `ok` to allow, `n` / `no` to deny.
