# cppco

`cppco` renders this C++ project as a literate-programming webpage.

It scans `src`, `include`, and `lib`, honors `.gitignore`, preserves all original code as code blocks, and renders documentation comments with `markdown-it` and KaTeX formulas. The development server runs on Vite, so source edits trigger browser reloads through Vite's websocket client.

Documentation comments use the format:

```cpp
/* @doc
## Heading
Markdown text here.
*/
```

Run it from the repository root:

```sh
npm --prefix scripts/cppco install
npm --prefix scripts/cppco start
```

Then open <http://localhost:5173>.

Each source file is served on its own `/file/...` page, with project tree navigation on the left.

Build a static GitHub Pages-ready site:

```sh
npm --prefix scripts/cppco run build
```

The static output is written to `scripts/cppco/dist`. Pass a different output directory with:

```sh
node scripts/cppco/server.mjs --build path/to/dist
```

Optional flags:

```sh
node scripts/cppco/server.mjs --root . --port 8080
```
