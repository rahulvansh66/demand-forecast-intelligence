Yes, but the experience is different.

For the workflow you described — **select a few lines → tell Cursor/Claude to modify `xyz.md` → agent edits the file directly** — the best option is still **local Markdown files**.

## Best option: Obsidian vault as local Markdown

Obsidian is the best fit among those three because an Obsidian vault is just a normal local folder full of `.md` files. You can open that folder in Cursor/VS Code, select lines, and ask the agent to edit the file directly. Obsidian itself also describes vault content as durable Markdown files that work offline and outside Obsidian. ([Obsidian][1])

So this works well:

```text
~/ObsidianVaults/project-notes/
  implementation-notes.md
  future-reminders.md
  old-guides/
    old-auth-flow.md
```

Then you can open that folder in Cursor, or add only selected parts as a workspace folder.

## Notion / Google Docs: possible, but less direct

With **Notion** and **Google Docs**, the files are not normal local `.md` files by default. Agents generally cannot just edit a selected block in a cloud doc the same way they edit a local file unless you use an integration, connector, MCP server, browser automation, or export/sync layer.

Claude Code supports MCP for connecting to external tools and data sources, and Notion has MCP-related docs/integration options. That can make editing possible, but it is more setup, more permissions, and usually less predictable than editing a local Markdown file. ([Claude Code][2])

So:

| Tool               |        Direct “edit selected lines in `.md`” workflow? | Good for your use case? |
| ------------------ | -----------------------------------------------------: | ----------------------- |
| Obsidian           |                  Yes, because notes are local Markdown | Best                    |
| Notion             | Not directly, unless using integration/MCP/sync/export | Okay, but friction      |
| Google Docs        |                         Not directly as Markdown files | Less ideal              |
| Plain local folder |                                                    Yes | Also excellent          |

## My recommended setup

Use **Obsidian**, but keep the vault **outside your code repo**:

```text
~/code/my-project/                 # actual repo
  src/
  docs/
  CLAUDE.md
  .cursorignore

~/notes/my-project/                # Obsidian vault / private project notes
  current-notes/
  future-reminders.md
  personal-decisions.md
  old-guides/
```

Then, when you want agent help editing notes, open the notes vault separately in Cursor/Claude.

When you do **not** want agents touching notes while working on code, only open:

```text
~/code/my-project/
```

When you want agents to edit notes, open:

```text
~/notes/my-project/
```

This gives you manual control.

## Even better: split notes into “agent-editable” and “private”

Inside your Obsidian vault:

```text
~/notes/my-project/
  agent-editable/
    implementation-summary.md
    migration-plan.md

  private/
    personal-reminders.md
    rough-thoughts.md
    sensitive-notes.md

  archive/
    old-implementation-guide.md
```

Then ignore/deny the private parts:

```gitignore
# .cursorignore
private/
archive/
```

```gitignore
# .cursorindexingignore
private/
archive/
```

For Claude, use permission deny rules for the same folders.

## Best practical pattern

I would use this:

```text
~/code/project-name/               # repo, agent can work here
~/notes/project-name/              # Obsidian vault, human notes
~/notes/project-name/agent-shared/ # only docs you intentionally let agents edit/read
~/notes/project-name/private/      # never open to agents
```

When you need the agent to update a doc, move or copy that specific doc into `agent-shared/`, or open only that file/folder in Cursor.

That gives you the convenience of local `.md` editing without making your whole private note system visible to the agent.

[1]: https://obsidian.md/help/import/notion?utm_source=chatgpt.com "Import from Notion - Obsidian Help"
[2]: https://code.claude.com/docs/en/mcp?utm_source=chatgpt.com "Connect Claude Code to tools via MCP - Claude Code Docs"
