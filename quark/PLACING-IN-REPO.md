# Adding Quark to cnms-living-fom

Quark lives as its own self-contained app inside the existing repo. The backend,
db, migrations and docs folders stay untouched.

## Steps

1. Unzip `quark.zip` so you have a folder named `quark/`.
2. Move that folder into the root of your local clone of
   `zackwoodel-debug/cnms-living-fom`, alongside `backend/`, `db/`, `docs/`.
3. From the repo root:

   ```bash
   git add quark
   git commit -m "Add Quark, the CNMS research chat interface"
   git push
   ```

## Running it

```bash
cd quark
bun install        # or: npm install
bun run dev        # serves on http://localhost:8080
```

Quark talks to your local Ollama instance. In a separate terminal:

```bash
OLLAMA_ORIGINS=http://localhost:8080 ollama serve
ollama pull llama3.1:8b
```

Everything else — configuration, design notes, current limitations — is in
`quark/README.md`.

## Note on the top-level README

The repo root README describes the backend. Consider adding one line to it
pointing at this folder, for example:

> `quark/` — Quark, the chat interface for the CNMS Living FOM work. See
> [quark/README.md](quark/README.md).
