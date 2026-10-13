---
name: mkdocs-docs
description: Use when a repo needs a documentation site built from Markdown files with a left-panel navigation and cross-page links, when adding, moving, or linking a docs page in a repo that has a mkdocs.yml, when converting README content into docs/, or when `mkdocs build --strict` fails on a link, anchor, or nav entry.
---

# MkDocs Material docs

## Overview

Docs are plain Markdown under `docs/`. `mkdocs.yml` turns them into a static
site: the `nav` list is the left panel, relative `.md` links are the
cross-references, and `mkdocs build --strict` fails on any broken link or
anchor. READMEs stay short entry points that link into `docs/`.

**Core rule: a docs change is done when `uv run mkdocs build --strict` passes
with zero warnings.** CI runs the same command. The red "MkDocs 2.0" banner
from the Material team is a notice, not a warning. It never fails the build.

## Quick reference

| Task | Do |
|------|----|
| Preview | `uv run mkdocs serve -a 127.0.0.1:8001` (port 8000 is often taken by the app) |
| Check | `uv run mkdocs build --strict` from the repo root |
| Add a page | Create `docs/<section>/<name>.md`, add it to `nav` in `mkdocs.yml`, link it from one related page |
| Link a page | `[Text](../other-page.md)`, relative to the current file. Never `/docs/...` or `.html` |
| Link a heading | `[Text](page.md#heading-slug)`. Slug = heading lower-cased, spaces to `-`, punctuation dropped |
| Link a repo file | GitHub URL (`<repo_url>/blob/main/<path>`). A relative link to a non-Markdown file or a directory is a warning |
| Diagram | A ```` ```mermaid ```` fence. Rendered in the browser; `--strict` does not check mermaid syntax |
| Callout | `!!! note "Title"` with the body indented four spaces. Also `tip`, `warning` |

## Set up a new site

1. `uv init --bare` if the repo has no root `pyproject.toml`. Then
   `uv add --dev mkdocs-material`. Use `--dev`, not a named group, so plain
   `uv run mkdocs` works.
2. Copy `templates/mkdocs.yml` from this skill folder to the repo root. Set
   `site_name`, `site_description`, `repo_url`. No remote yet? Delete
   `repo_url` and `edit_uri`, or the header link is dead.
3. Copy `templates/docs.yml` to `.github/workflows/docs.yml`. Its
   `uv run --locked` needs the `uv.lock` from step 1, so commit the lock file.
4. Add `site/` to `.gitignore`.
5. Create `docs/index.md` and one page per topic. Default `nav` order:
   Overview, Getting started, Architecture, reference pages, one page per
   component, Development.
6. Run the strict build.

## Convert READMEs into docs

- Move prose, diagrams, and reference tables into `docs/` unchanged. Split a
  long README by its headings into pages.
- Content repeated across READMEs (a protocol table, a diagram) becomes one
  page. The other pages link to it.
- Each README keeps: title, one paragraph, How to run, How to test, and a link
  to its docs page. How to run stays at the top.
- Rewrite README links such as `../README.md` or `sub_dir/` as docs links.

## Common mistakes

- Page created but not added to `nav`: it builds but nobody can reach it.
  The strict build only logs INFO for this, so check `nav` yourself.
- Dependency added in a `docs` group: every command then needs
  `--group docs`. Use `uv add --dev`.
- `site/` committed. It is build output.
- Serving on `:8000` while the app holds that port. Pass `-a 127.0.0.1:8001`.
- Links written as `/docs/page.md` or `page.html`. Only relative `.md`
  links work on GitHub and in the site.
- Detailed content left in a README after the page exists in `docs/`. The two
  copies drift. Replace it with a link.
