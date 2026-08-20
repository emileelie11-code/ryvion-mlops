# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues on `emileastih1/ryvion-mlops`. Use the `gh` CLI
for all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'` with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` / `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

Infer the repo from `git remote -v` — `gh` does this automatically when run inside a clone. No
`--repo` flag is needed for work in this checkout.

## When a skill says "publish to the issue tracker"

Create a GitHub issue.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.

## Note on forks

Students fork this repo and submit coursework as pull requests. `gh` resolves the repo from
whichever remote the checkout points at, so a command run inside a student's fork targets that
fork's issues, not this upstream. Pass `--repo emileastih1/ryvion-mlops` explicitly when you mean
upstream from inside a fork.
