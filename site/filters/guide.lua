-- guide.lua — render the guide's admonition/exercise divs for the PDF and the
-- standalone HTML. Sibling of mkdocs.lua, which does the LaTeX->markdown
-- direction for the generated report pages.
--
-- site/build_guide.py rewrites mkdocs-material's `!!! note "T"` / `??? question "T"`
-- blocks into pandoc fenced divs, because pandoc does not understand the
-- material syntax (it passes the marker through as literal text and mangles the
-- math inside). This filter turns those divs into something printable.
--
-- pandoc DROPS an unrecognised Div in LaTeX output — the content survives but
-- all box structure is silently lost, so an exercise and its solution become
-- indistinguishable prose. Hence the explicit LaTeX below.
--
-- Solutions stay INLINE in print (a PDF cannot collapse), visually separated so
-- the eye can skip them. On the web they collapse; on paper they indent.

local COLOR = {
  note = "guideNote", tip = "guideTip", abstract = "guideNote",
  question = "guideQuestion", success = "guideSuccess",
  warning = "guideWarn", danger = "guideWarn", failure = "guideWarn",
  info = "guideNote", example = "guideNote", quote = "guideNote",
  bug = "guideWarn",
}

-- mdframed is in every TeX Live and needs no tcolorbox/tikz dependency chain.
-- NB: no [most] option — that belongs to tcolorbox; mdframed errors on it.
local PREAMBLE = [[
\usepackage{mdframed}
\usepackage{xcolor}
\definecolor{guideNote}{HTML}{00543C}
\definecolor{guideTip}{HTML}{127A80}
\definecolor{guideQuestion}{HTML}{A8730A}
\definecolor{guideSuccess}{HTML}{00543C}
\definecolor{guideWarn}{HTML}{B3282D}
\newmdenv[
  linewidth=2pt, linecolor=guideNote, topline=false, bottomline=false,
  rightline=false, leftmargin=0pt, innerleftmargin=10pt, innertopmargin=6pt,
  innerbottommargin=6pt, skipabove=8pt, skipbelow=8pt
]{guidebox}
]]

-- Titles must be PARSED as markdown, never interpolated into a raw LaTeX
-- string. Real chapter titles contain `b_n`, `$\theta_E$`, `&`, `%` and em
-- dashes; a raw `\textbf{Exercise 10.1 — b_n exactly}` makes TeX read the
-- underscore as a subscript operator and die with "Missing $ inserted".
-- Round-tripping through the reader lets pandoc escape text and typeset math.
local function title_inlines(title)
  if not title or title == "" then return {} end
  local ok, doc = pcall(pandoc.read, title, "markdown")
  if not ok then return { pandoc.Str(title) } end
  return pandoc.utils.blocks_to_inlines(doc.blocks)
end

function Meta(m)
  -- Inject the preamble without needing a template file.
  local extra = m["header-includes"] or pandoc.MetaList({})
  if extra.t ~= "MetaList" then extra = pandoc.MetaList({ extra }) end
  extra:insert(pandoc.MetaBlocks({ pandoc.RawBlock("latex", PREAMBLE) }))
  m["header-includes"] = extra
  return m
end

function Div(el)
  if not el.classes:includes("admonition") then return nil end

  local class
  for _, c in ipairs(el.classes) do
    if c ~= "admonition" and c ~= "collapsible" then class = c break end
  end
  local title = el.attributes["title"]

  local color = COLOR[class] or "guideNote"
  local tin = title_inlines(title)

  if FORMAT:match("latex") then
    -- Title as a real Para of inlines, bracketed by raw colour commands, so
    -- pandoc's writer escapes the text and typesets any math in it.
    local head = { pandoc.RawInline("latex",
                     "\\textcolor{" .. color .. "}{\\bfseries ") }
    for _, i in ipairs(tin) do head[#head + 1] = i end
    head[#head + 1] = pandoc.RawInline("latex", "}")

    local blocks = {
      pandoc.RawBlock("latex", "\\begin{guidebox}[linecolor=" .. color .. "]"),
    }
    if #tin > 0 then blocks[#blocks + 1] = pandoc.Para(head) end
    for _, b in ipairs(el.content) do blocks[#blocks + 1] = b end
    blocks[#blocks + 1] = pandoc.RawBlock("latex", "\\end{guidebox}")
    return blocks
  end

  if FORMAT:match("html") then
    -- A real <details> for question/success, so the offline HTML keeps the
    -- site's collapse behaviour with no JavaScript.
    local collapsible = (class == "question" or class == "success")
    local open_tag = collapsible
      and ('<details class="adm ' .. class .. '">')
      or ('<div class="adm ' .. class .. '">')
    local close_tag = collapsible and "</details>" or "</div>"

    local blocks = { pandoc.RawBlock("html", open_tag) }
    if #tin > 0 then
      local head = { pandoc.RawInline("html",
                       collapsible and "<summary>" or '<p class="adm-title">') }
      for _, i in ipairs(tin) do head[#head + 1] = i end
      head[#head + 1] = pandoc.RawInline("html",
                          collapsible and "</summary>" or "</p>")
      blocks[#blocks + 1] = pandoc.Plain(head)
    end
    for _, b in ipairs(el.content) do blocks[#blocks + 1] = b end
    blocks[#blocks + 1] = pandoc.RawBlock("html", close_tag)
    return blocks
  end
  return nil
end

-- pdflatex cannot embed SVG. make_figures.py --pdf emits a `<slug>-light.pdf`
-- next to each SVG for exactly this target, so rewrite the extension for LaTeX
-- output only. (The HTML target embeds the SVG directly and is unaffected.)
function Image(el)
  if FORMAT:match("latex") then
    el.src = el.src:gsub("%.svg$", ".pdf")
  end
  return el
end

-- mkdocs-material renders `<figure markdown="span">` itself; pandoc sees the raw
-- HTML and would pass the attribute through into the output. Strip it.
function RawBlock(el)
  if el.format:match("html") then
    el.text = el.text:gsub('%s*markdown="span"', "")
    if FORMAT:match("latex") then
      -- Bare <figure>/<figcaption> tags are meaningless in LaTeX; drop them and
      -- let the images and caption text flow as normal blocks.
      if el.text:match("^%s*</?figure") or el.text:match("^%s*</?figcaption") then
        return {}
      end
    end
  end
  return el
end

return {
  { Meta = Meta },
  { Div = Div, RawBlock = RawBlock, Image = Image },
}
