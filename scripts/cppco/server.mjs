import { createRequire } from "node:module";
import { mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import path from "node:path";
import process from "node:process";
import MarkdownIt from "markdown-it";
import markdownItKatex from "markdown-it-katex";
import { createServer as createViteServer } from "vite";

const require = createRequire(import.meta.url);
const args = parseArgs(process.argv.slice(2));
const projectRoot = path.resolve(args.root ?? path.join(import.meta.dirname, "..", ".."));
const port = Number(args.port ?? process.env.PORT ?? 5173);
const sourceRoots = ["src", "include", "lib"];
const sourceExts = new Set([".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"]);
const markdown = new MarkdownIt({
  html: false,
  linkify: true,
  typographer: false,
  langPrefix: "language-"
}).use(markdownItKatex, { throwOnError: false });

if (args.build) {
  await buildStaticSite(projectRoot, path.resolve(projectRoot, args.out ?? "scripts/cppco/dist"));
} else {
  await startServer();
}

async function startServer() {
  const vite = await createViteServer({
    root: projectRoot,
    appType: "custom",
    server: {
      port
    }
  });

  watchSourceChanges(vite, projectRoot);
  vite.middlewares.use(handleRequest);

  await vite.listen();
  vite.printUrls();
  console.log(`cppco serving ${projectRoot}`);
}

async function handleRequest(req, res) {
  try {
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    if (url.pathname === "/style.css") {
      send(res, 200, "text/css; charset=utf-8", stylesheet());
      return;
    }
    if (url.pathname === "/client.js") {
      send(res, 200, "text/javascript; charset=utf-8", clientScript());
      return;
    }
    if (url.pathname === "/highlight.min.js") {
      await sendHighlightAsset(res, "highlight.min.js", "text/javascript; charset=utf-8", "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/highlight.min.js");
      return;
    }
    if (url.pathname === "/highlight.css") {
      await sendHighlightAsset(res, "styles/github.min.css", "text/css; charset=utf-8", "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/styles/github.min.css");
      return;
    }
    if (url.pathname === "/katex.min.css") {
      await sendPackageAsset(res, "katex", "dist/katex.min.css", "text/css; charset=utf-8", "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css");
      return;
    }
    if (url.pathname.startsWith("/fonts/")) {
      await sendKatexFont(res, url.pathname.slice("/fonts/".length));
      return;
    }

    const files = await collectSourceFiles(projectRoot);
    if (url.pathname === "/") {
      const firstFile = files[0]?.relative;
      if (!firstFile) {
        send(res, 200, "text/html; charset=utf-8", await renderEmptyPage(liveRenderOptions()));
        return;
      }
      redirect(res, fileHref(firstFile));
      return;
    }

    if (!url.pathname.startsWith("/file/")) {
      send(res, 404, "text/plain; charset=utf-8", "Not found\n");
      return;
    }

    const activePath = decodeURIComponent(url.pathname.slice("/file/".length));
    const activeFile = files.find((file) => file.relative === activePath);
    if (!activeFile) {
      send(res, 404, "text/plain; charset=utf-8", "Source file not found\n");
      return;
    }

    const html = await renderPage(files, activeFile, liveRenderOptions());
    send(res, 200, "text/html; charset=utf-8", html);
  } catch (error) {
    console.error(error);
    send(res, 500, "text/plain; charset=utf-8", `${error.stack ?? error}\n`);
  }
}

function parseArgs(argv) {
  const parsed = {};
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--root") {
      parsed.root = argv[++i];
    } else if (arg === "--port") {
      parsed.port = argv[++i];
    } else if (arg === "--build") {
      parsed.build = true;
      if (argv[i + 1] && !argv[i + 1].startsWith("-")) {
        parsed.out = argv[++i];
      }
    } else if (arg === "--out") {
      parsed.out = argv[++i];
    } else if (arg === "--help" || arg === "-h") {
      console.log("Usage: node scripts/cppco/server.mjs [--root PATH] [--port PORT] [--build [OUT_DIR]]");
      process.exit(0);
    }
  }
  return parsed;
}

function send(res, status, contentType, body) {
  res.writeHead(status, { "Content-Type": contentType });
  res.end(body);
}

function redirect(res, location) {
  res.writeHead(302, { Location: location });
  res.end();
}

function liveRenderOptions() {
  return {
    hrefForFile: fileHref,
    assets: {
      style: "/style.css",
      client: "/client.js",
      highlightCss: "/highlight.css",
      highlightJs: "/highlight.min.js",
      katexCss: "/katex.min.css",
      viteClient: "/@vite/client",
      fontCss: "https://cdn.jsdelivr.net/npm/computer-modern@0.1.2/cmu-serif.css"
    }
  };
}

async function collectSourceFiles(root) {
  const files = [];
  const ignore = await loadGitignore(root);
  for (const sourceRoot of sourceRoots) {
    const absolute = path.join(root, sourceRoot);
    await walk(root, absolute, files, ignore);
  }
  files.sort((a, b) => a.relative.localeCompare(b.relative));
  return files;
}

async function walk(root, dir, files, ignore) {
  let entries;
  try {
    entries = await readdir(dir, { withFileTypes: true });
  } catch (error) {
    if (error.code === "ENOENT") {
      return;
    }
    throw error;
  }

  for (const entry of entries) {
    const absolute = path.join(dir, entry.name);
    const relative = path.relative(root, absolute).replaceAll(path.sep, "/");
    if (entry.isDirectory()) {
      if (!ignore(relative, true)) {
        await walk(root, absolute, files, ignore);
      }
      continue;
    }
    if (ignore(relative, false)) {
      continue;
    }
    if (!entry.isFile() || !sourceExts.has(path.extname(entry.name))) {
      continue;
    }
    files.push({
      absolute,
      relative
    });
  }
}

async function renderPage(files, activeFile, options) {
  let activeSections = [];

  for (const file of files) {
    const source = await readFile(file.absolute, "utf8");
    const sections = parseLiterateSections(source);
    if (file.relative === activeFile.relative) {
      activeSections = sections;
    }
  }

  const nav = renderTreeNav(files, activeFile.relative, options);

  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cppco</title>
  <link rel="stylesheet" href="${options.assets.highlightCss}">
  <link rel="stylesheet" href="${options.assets.katexCss}">
  <link rel="stylesheet" href="${options.assets.fontCss}">
  <link rel="stylesheet" href="${options.assets.style}">
  <script src="${options.assets.highlightJs}" defer></script>
  <script src="${options.assets.client}" defer></script>
  ${options.assets.viteClient ? `<script type="module" src="${options.assets.viteClient}"></script>` : ""}
</head>
<body>
  <main class="layout">
    <aside class="sidebar">
      <nav aria-label="Source files">${nav}</nav>
    </aside>
    <article class="content">
      ${renderFile(activeFile.relative, activeSections)}
    </article>
  </main>
</body>
</html>`;
}

async function renderEmptyPage(options = { assets: { style: "/style.css" } }) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>cppco</title>
  <link rel="stylesheet" href="${options.assets.style}">
  ${options.assets.viteClient ? `<script type="module" src="${options.assets.viteClient}"></script>` : ""}
</head>
<body>
  <main class="empty">
    <h1>cppco</h1>
    <p>No source files were found in src, include, or lib.</p>
  </main>
</body>
</html>`;
}

async function buildStaticSite(root, outDir) {
  const files = await collectSourceFiles(root);
  await rm(outDir, { recursive: true, force: true });
  await mkdir(outDir, { recursive: true });
  await writeFile(path.join(outDir, ".nojekyll"), "");
  await writeFile(path.join(outDir, "style.css"), stylesheet());
  await writeFile(path.join(outDir, "client.js"), clientScript());

  const thirdPartyAssets = await writeStaticThirdPartyAssets(outDir);
  if (thirdPartyAssets.katexCss.local) {
    await writeStaticKatexFonts(outDir);
  }
  if (files.length === 0) {
    await writeFile(path.join(outDir, "index.html"), await renderEmptyPage({ assets: { style: "style.css" } }));
    console.log(`cppco wrote empty site to ${outDir}`);
    return;
  }

  const firstPage = staticPageRel(files[0].relative);
  await writeFile(path.join(outDir, "index.html"), renderStaticIndex(firstPage));

  for (const file of files) {
    const pageRel = staticPageRel(file.relative);
    const pagePath = path.join(outDir, ...pageRel.split("/"));
    await mkdir(path.dirname(pagePath), { recursive: true });
    const html = await renderPage(files, file, staticRenderOptions(pageRel, thirdPartyAssets));
    await writeFile(pagePath, html);
  }

  console.log(`cppco wrote ${files.length} pages to ${outDir}`);
}

function renderStaticIndex(firstPage) {
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta http-equiv="refresh" content="0; url=${escapeHtml(firstPage)}">
  <title>cppco</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <main class="empty">
    <h1>cppco</h1>
    <p><a href="${escapeHtml(firstPage)}">Open the source companion.</a></p>
  </main>
</body>
</html>`;
}

function staticRenderOptions(pageRel, thirdPartyAssets) {
  const pageDir = path.posix.dirname(pageRel);
  const rootRel = relativeUrl(pageDir, ".");
  const assetPrefix = rootRel === "" ? "" : `${rootRel}/`;
  return {
    hrefForFile: (relativePath) => relativeUrl(pageDir, staticPageRel(relativePath)),
    assets: {
      style: `${assetPrefix}style.css`,
      client: `${assetPrefix}client.js`,
      highlightCss: assetUrl(assetPrefix, thirdPartyAssets.highlightCss),
      highlightJs: assetUrl(assetPrefix, thirdPartyAssets.highlightJs),
      katexCss: assetUrl(assetPrefix, thirdPartyAssets.katexCss),
      fontCss: "https://cdn.jsdelivr.net/npm/computer-modern@0.1.2/cmu-serif.css"
    }
  };
}

function staticPageRel(relativePath) {
  return `file/${relativePath.split("/").map(encodeURIComponent).join("/")}/index.html`;
}

function relativeUrl(fromDir, toPath) {
  const rel = path.posix.relative(fromDir, toPath);
  return rel === "" ? "" : rel;
}

function assetUrl(prefix, asset) {
  return asset.local ? `${prefix}${asset.href}` : asset.href;
}

async function writeStaticThirdPartyAssets(outDir) {
  return {
    highlightJs: await writePackageAsset(outDir, "@highlightjs/cdn-assets", "highlight.min.js", "highlight.min.js", "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/highlight.min.js"),
    highlightCss: await writePackageAsset(outDir, "@highlightjs/cdn-assets", "styles/github.min.css", "highlight.css", "https://cdn.jsdelivr.net/npm/@highlightjs/cdn-assets@11.11.1/styles/github.min.css"),
    katexCss: await writePackageAsset(outDir, "katex", "dist/katex.min.css", "katex.min.css", "https://cdn.jsdelivr.net/npm/katex@0.16.22/dist/katex.min.css")
  };
}

async function writePackageAsset(outDir, packageName, packagePath, outputName, fallbackHref) {
  try {
    const packageRoot = path.dirname(require.resolve(`${packageName}/package.json`));
    const assetPath = path.join(packageRoot, packagePath);
    await writeFile(path.join(outDir, outputName), await readFile(assetPath, "utf8"));
    return { local: true, href: outputName };
  } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND" && error.code !== "ERR_MODULE_NOT_FOUND" && error.code !== "ENOENT") {
      throw error;
    }
    return { local: false, href: fallbackHref };
  }
}

async function writeStaticKatexFonts(outDir) {
  const sourceDir = packageAssetPath("katex", "dist/fonts");
  const targetDir = path.join(outDir, "fonts");
  await mkdir(targetDir, { recursive: true });
  const entries = await readdir(sourceDir, { withFileTypes: true });
  for (const entry of entries) {
    if (!entry.isFile() || !isFontAsset(entry.name)) {
      continue;
    }
    await writeFile(path.join(targetDir, entry.name), await readFile(path.join(sourceDir, entry.name)));
  }
}

function renderTreeNav(files, activePath, options) {
  const root = { children: new Map(), file: null };
  for (const file of files) {
    let node = root;
    const parts = file.relative.split("/");
    for (const part of parts) {
      if (!node.children.has(part)) {
        node.children.set(part, { children: new Map(), file: null });
      }
      node = node.children.get(part);
    }
    node.file = file.relative;
  }
  return renderTreeNode(root, activePath, options);
}

function renderTreeNode(node, activePath, options) {
  const entries = [...node.children.entries()].sort(([aName, aNode], [bName, bNode]) => {
    const aIsDir = aNode.children.size > 0;
    const bIsDir = bNode.children.size > 0;
    if (aIsDir !== bIsDir) {
      return aIsDir ? -1 : 1;
    }
    return aName.localeCompare(bName);
  });
  if (entries.length === 0) {
    return "";
  }
  return `<ul class="tree">${entries.map(([name, child]) => renderTreeEntry(name, child, activePath, options)).join("")}</ul>`;
}

function renderTreeEntry(name, node, activePath, options) {
  if (node.file) {
    const active = node.file === activePath ? " aria-current=\"page\"" : "";
    return `<li><a class="file-link" href="${options.hrefForFile(node.file)}"${active}>${escapeHtml(name)}</a></li>`;
  }
  return `<li><span class="tree-dir">${escapeHtml(name)}</span>${renderTreeNode(node, activePath, options)}</li>`;
}

function parseLiterateSections(source) {
  const lines = source.split(/\r?\n/);
  const sections = [];
  let code = [];
  let codeStartLine = 1;
  let docs = [];
  let inDoc = false;
  let docIndent = "";

  for (let lineIndex = 0; lineIndex < lines.length; lineIndex += 1) {
    const line = lines[lineIndex];
    const docStart = line.match(/^(\s*)\/\*\s*@doc\s*$/);
    if (!inDoc && docStart) {
      flushCode();
      inDoc = true;
      docIndent = docStart[1];
      docs = [];
      continue;
    }
    if (inDoc && /^\s*\*\/\s*$/.test(line)) {
      sections.push({ type: "doc", text: docs.join("\n").trim() });
      inDoc = false;
      docIndent = "";
      codeStartLine = lineIndex + 2;
      continue;
    }
    if (inDoc) {
      docs.push(stripDocIndent(line, docIndent));
    } else {
      code.push(line);
    }
  }

  if (inDoc) {
    sections.push({ type: "doc", text: docs.join("\n").trim() });
  } else {
    flushCode();
  }

  return sections;

  function flushCode() {
    const { lines: trimmed, startLine } = trimBlankCodeLines(code, codeStartLine);
    const text = trimmed.join("\n");
    if (text.trim() !== "") {
      sections.push({ type: "code", text, startLine });
    }
    code = [];
  }
}

function stripDocIndent(line, indent) {
  if (line.trim() === "") {
    return "";
  }
  return indent !== "" && line.startsWith(indent) ? line.slice(indent.length) : line;
}

function trimBlankCodeLines(lines, startLine) {
  let first = 0;
  let last = lines.length;
  while (first < last && lines[first].trim() === "") {
    first += 1;
  }
  while (last > first && lines[last - 1].trim() === "") {
    last -= 1;
  }
  return {
    lines: lines.slice(first, last),
    startLine: startLine + first
  };
}

function renderFile(relativePath, sections) {
  const body = sections
    .map((section) => {
      if (section.type === "doc") {
        return `<section class="prose">${markdown.render(section.text)}</section>`;
      }
      return renderCodeBlock(section.text, section.startLine);
    })
    .join("\n");

  return `<section class="file" id="${slug(relativePath)}">
  <h2>${escapeHtml(relativePath)}</h2>
  ${body}
</section>`;
}

function renderCodeBlock(source, startLine) {
  const lines = source.split("\n");
  const numbers = lines.map((_, index) => `<span>${startLine + index}</span>`).join("");
  return `<div class="code-block"><pre class="line-numbers" aria-hidden="true">${numbers}</pre><pre class="code"><code class="language-cpp">${escapeHtml(source)}</code></pre></div>`;
}

function fileHref(relativePath) {
  return `/file/${relativePath.split("/").map(encodeURIComponent).join("/")}`;
}

function slug(value) {
  return value.toLowerCase().replace(/[^a-z0-9_-]+/g, "-").replace(/^-|-$/g, "");
}

function escapeHtml(value) {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function loadGitignore(root) {
  let content = "";
  try {
    content = await readFile(path.join(root, ".gitignore"), "utf8");
  } catch (error) {
    if (error.code !== "ENOENT") {
      throw error;
    }
  }
  const rules = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "" && !line.startsWith("#"))
    .map(parseGitignoreRule);

  return (relativePath, isDirectory) => {
    const normalized = relativePath.replaceAll(path.sep, "/");
    let ignored = false;
    for (const rule of rules) {
      if (gitignoreRuleMatches(rule, normalized, isDirectory)) {
        ignored = !rule.negated;
      }
    }
    return ignored;
  };
}

function parseGitignoreRule(line) {
  let pattern = line;
  let negated = false;
  if (pattern.startsWith("!")) {
    negated = true;
    pattern = pattern.slice(1);
  }
  pattern = pattern.replace(/^\/+/, "");
  const directoryOnly = pattern.endsWith("/");
  const directoryPrefix = pattern.endsWith("/**") ? pattern.slice(0, -3) : "";
  pattern = pattern.replace(/\/+$/, "");
  const basenameOnly = !pattern.includes("/");
  return { pattern, negated, directoryOnly, basenameOnly, directoryPrefix };
}

function gitignoreRuleMatches(rule, relativePath, isDirectory) {
  if (rule.directoryOnly && !isDirectory) {
    return false;
  }
  if (rule.directoryPrefix && (relativePath === rule.directoryPrefix || relativePath.startsWith(`${rule.directoryPrefix}/`))) {
    return true;
  }
  const candidates = rule.basenameOnly ? relativePath.split("/") : [relativePath];
  return candidates.some((candidate) => globToRegExp(rule.pattern, rule.basenameOnly).test(candidate));
}

function globToRegExp(pattern, basenameOnly) {
  let source = "";
  for (let i = 0; i < pattern.length; i += 1) {
    const char = pattern[i];
    const next = pattern[i + 1];
    if (char === "*" && next === "*") {
      source += ".*";
      i += 1;
    } else if (char === "*") {
      source += "[^/]*";
    } else if (char === "?") {
      source += "[^/]";
    } else {
      source += escapeRegExp(char);
    }
  }
  return new RegExp(basenameOnly ? `^${source}$` : `^${source}(?:/.*)?$`);
}

function escapeRegExp(value) {
  return value.replace(/[|\\{}()[\]^$+*?.]/g, "\\$&");
}

async function sendHighlightAsset(res, packagePath, contentType, fallbackUrl) {
  await sendPackageAsset(res, "@highlightjs/cdn-assets", packagePath, contentType, fallbackUrl);
}

async function sendPackageAsset(res, packageName, packagePath, contentType, fallbackUrl) {
  try {
    send(res, 200, contentType, await readFile(packageAssetPath(packageName, packagePath), "utf8"));
  } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND" && error.code !== "ERR_MODULE_NOT_FOUND" && error.code !== "ENOENT") {
      throw error;
    }
    res.writeHead(302, { Location: fallbackUrl });
    res.end();
  }
}

async function sendKatexFont(res, fileName) {
  const decodedName = decodeURIComponent(fileName);
  if (decodedName.includes("/") || decodedName.includes("\\") || !decodedName.startsWith("KaTeX_") || !isFontAsset(decodedName)) {
    send(res, 404, "text/plain; charset=utf-8", "Font not found\n");
    return;
  }

  try {
    send(res, 200, fontContentType(decodedName), await readFile(packageAssetPath("katex", `dist/fonts/${decodedName}`)));
  } catch (error) {
    if (error.code !== "MODULE_NOT_FOUND" && error.code !== "ERR_MODULE_NOT_FOUND" && error.code !== "ENOENT") {
      throw error;
    }
    send(res, 404, "text/plain; charset=utf-8", "Font not found\n");
  }
}

function packageAssetPath(packageName, packagePath) {
  const packageRoot = path.dirname(require.resolve(`${packageName}/package.json`));
  return path.join(packageRoot, packagePath);
}

function isFontAsset(fileName) {
  return [".woff2", ".woff", ".ttf"].includes(path.extname(fileName));
}

function fontContentType(fileName) {
  const ext = path.extname(fileName);
  if (ext === ".woff2") {
    return "font/woff2";
  }
  if (ext === ".woff") {
    return "font/woff";
  }
  return "font/ttf";
}

function watchSourceChanges(vite, root) {
  const watched = [
    path.join(root, ".gitignore"),
    ...sourceRoots.map((sourceRoot) => path.join(root, sourceRoot))
  ];

  vite.watcher.add(watched);
  vite.watcher.on("all", (event, changedPath) => {
    if (!["add", "change", "unlink", "addDir", "unlinkDir"].includes(event) || !isSourceWatchPath(root, changedPath)) {
      return;
    }
    vite.ws.send({ type: "full-reload", path: "*" });
  });
}

function isSourceWatchPath(root, changedPath) {
  const relative = path.relative(root, changedPath).replaceAll(path.sep, "/");
  if (relative === ".gitignore") {
    return true;
  }
  const [sourceRoot] = relative.split("/");
  return sourceRoots.includes(sourceRoot) && (sourceExts.has(path.extname(relative)) || !path.extname(relative));
}

function stylesheet() {
  return `
:root {
  color-scheme: light;
  --bg: #fbfbfa;
  --panel: #ffffff;
  --ink: #1c1f23;
  --muted: #6b7280;
  --line: #d9dde3;
  --code-bg: #f6f8fa;
  --code-ink: #24292f;
  --code-line: #8c959f;
  --accent: #0969da;
  --accent-2: #57606a;
  --serif: "CMU Serif", "Computer Modern Serif", "Latin Modern Roman", Georgia, "Times New Roman", serif;
  --mono: "Fira Mono", "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
}

* {
  box-sizing: border-box;
}

html {
  scroll-behavior: smooth;
}

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: var(--serif);
  line-height: 1.2;
}

.topbar {
  position: sticky;
  top: 0;
  z-index: 5;
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: center;
  min-height: 104px;
  padding: 20px clamp(18px, 4vw, 48px);
  background: color-mix(in srgb, var(--bg) 92%, white);
  border-bottom: 1px solid var(--line);
  backdrop-filter: blur(14px);
}

.eyebrow {
  margin: 0 0 2px;
  color: var(--accent-2);
  font-size: 12px;
  font-weight: 800;
  letter-spacing: 0;
  text-transform: uppercase;
}

h1 {
  margin: 0;
  font-size: clamp(28px, 4vw, 48px);
  line-height: 1;
}

.stats {
  display: grid;
  grid-template-columns: repeat(3, minmax(74px, 1fr));
  gap: 10px;
  margin: 0;
}

.stats div {
  min-width: 74px;
  padding: 8px 10px;
  border: 1px solid var(--line);
  background: var(--panel);
}

.stats dt {
  color: var(--muted);
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
}

.stats dd {
  margin: 0;
  font-size: 22px;
  font-weight: 800;
}

.layout {
  display: grid;
  grid-template-columns: minmax(220px, 300px) minmax(0, 1fr);
  align-items: start;
}

.sidebar {
  position: sticky;
  top: 0px;
  height: calc(100vh - 0px);
  overflow: auto;
  border-right: 1px solid var(--line);
  background: #f3f4f6;
  padding: 18px;
}

.tree {
  list-style: none;
  margin: 0;
  padding: 0;
}

.tree .tree {
  margin-left: 12px;
  padding-left: 10px;
  border-left: 1px solid #d8dee8;
}

.tree li {
  margin: 2px 0;
}

.tree-dir {
  display: block;
  padding: 7px 9px 3px;
  color: #47545c;
  font-size: 11px;
  font-weight: 800;
  text-transform: uppercase;
}

.sidebar a {
  display: block;
  padding: 7px 9px;
  color: var(--ink);
  font-family: var(--mono);
  font-size: 12px;
  text-decoration: none;
  overflow-wrap: anywhere;
}

.sidebar a:hover,
.sidebar a:focus-visible,
.sidebar a[aria-current="page"] {
  background: #e7ecf3;
  outline: none;
}

.sidebar a[aria-current="page"] {
  color: var(--accent);
  font-weight: 800;
}

.content {
  min-width: 0;
  padding: 28px clamp(18px, 4vw, 54px) 80px;
}

.file {
  margin: 0 0 56px;
  scroll-margin-top: 128px;
}

.file > h2 {
  margin: 0 0 18px;
  color: var(--accent);
  font-family: var(--mono);
  font-size: clamp(20px, 2.4vw, 30px);
  overflow-wrap: anywhere;
}

.prose {
  width: 100%;
  margin: 28px 0;
  font-family: var(--serif);
  font-size: 18px;
}

.prose h1,
.prose h2,
.prose h3,
.prose h4,
.prose h5,
.prose h6 {
  margin: 0 0 10px;
  line-height: 1.2;
}

.prose p {
  margin: 0 0 18px;
}

.prose p:last-child {
  margin-bottom: 0;
}

.prose code {
  padding: 0 2px;
  background: transparent;
  color: inherit;
  font-family: var(--mono);
  font-size: 0.96em;
  white-space: nowrap;
}

.prose .katex {
  white-space: normal;
}

.katex-display {
  overflow-x: auto;
  overflow-y: hidden;
  padding: 6px 0;
}

.code-block,
.code,
.prose pre {
  margin: 0;
  overflow: auto;
  border: 1px solid var(--line);
  background: var(--code-bg);
  color: var(--code-ink);
  line-height: 1.55;
}

.code-block {
  display: grid;
  grid-template-columns: max-content minmax(0, 1fr);
  overflow: hidden;
}

.line-numbers {
  margin: 0;
  padding: 14px 10px 14px 12px;
  border-right: 1px solid var(--line);
  color: var(--code-line);
  font-family: var(--mono);
  font-size: 13px;
  line-height: 1.55;
  text-align: right;
  user-select: none;
}

.line-numbers span {
  display: block;
}

.code {
  min-width: 0;
  margin: 0;
  padding: 14px 18px 14px 14px;
  border: 0;
  overflow: auto;
}

.prose pre {
  padding: 14px;
}

.code code,
.prose pre code {
  font-family: var(--mono);
  font-size: 13px;
  white-space: pre;
}

.code code.hljs {
  padding: 0;
  overflow: visible;
  background: transparent;
}

@media (max-width: 820px) {
  .topbar {
    position: static;
    align-items: flex-start;
    flex-direction: column;
  }

  .stats {
    width: 100%;
  }

  .layout {
    display: block;
  }

  .sidebar {
    position: static;
    height: auto;
    max-height: 220px;
    border-right: 0;
    border-bottom: 1px solid var(--line);
  }
}

.empty {
  min-height: 100vh;
  display: grid;
  place-content: center;
  padding: 32px;
  text-align: center;
}
`;
}

function clientScript() {
  return `
window.addEventListener("DOMContentLoaded", () => {
  if (window.hljs) {
    window.hljs.highlightAll();
  }
});
`;
}
