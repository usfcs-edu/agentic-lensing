-- mkdocs.lua: post-process the pandoc AST for mkdocs-material + python-markdown.
--
-- Runs after --citeproc (flag order on the command line matters), so the
-- bibliography div (#refs) is present in the AST.

local figcount, tabcount = 0, 0

-- MathJax has no \ensuremath; pandoc macro expansion can leave it inside
-- math strings (e.g. from tech-report.sty-style macros). Also replace |
-- with \vert: a raw pipe inside $...$ in a markdown table cell would split
-- the cell (gfm escapes text pipes but not math content).
function Math(el)
  el.text = el.text:gsub("\\ensuremath%s*", "")
  el.text = el.text:gsub("\\|", "\\Vert "):gsub("|", "\\vert ")
  -- arithmatex refuses $...$ whose content starts/ends with whitespace,
  -- which the \vert substitution can introduce at the boundaries.
  el.text = el.text:gsub("^%s+", ""):gsub("%s+$", "")
  return el
end

-- python-markdown only renders markdown inside raw-HTML divs when
-- markdown="1" is present (md_in_html). Also give the bibliography a heading.
function Div(el)
  el.attributes["markdown"] = "1"
  if el.identifier == "refs" then
    return { pandoc.Header(2, "References"), el }
  end
  return el
end

-- gfm drops header ids; emit explicit anchors for labelled sections so
-- pandoc-resolved \ref links (e.g. #sec:model) land somewhere.
function Header(el)
  if el.identifier:find(":") then
    return { pandoc.RawBlock("html",
             string.format('<a id="%s"></a>', el.identifier)), el }
  end
  return el
end

-- Prefix captions to match the PDF ("Figure N:", "Table N:"); numbering is
-- document order, identical to pandoc's internal \ref resolution.
function Figure(el)
  figcount = figcount + 1
  local cap = el.caption.long
  if cap and #cap > 0 and cap[1].content then
    table.insert(cap[1].content, 1, pandoc.Space())
    table.insert(cap[1].content, 1,
      pandoc.Strong(pandoc.Str(string.format("Figure %d:", figcount))))
  end
  return el
end

function Table(el)
  if #el.caption.long > 0 then
    tabcount = tabcount + 1
    local cap = el.caption.long
    if cap[1].content then
      table.insert(cap[1].content, 1, pandoc.Space())
      table.insert(cap[1].content, 1,
        pandoc.Strong(pandoc.Str(string.format("Table %d:", tabcount))))
    end
  end
  return el
end

-- Display math that sits mid-paragraph (no blank line around the equation
-- env in the LaTeX) would be written inline as $$...$$ inside the paragraph,
-- where arithmatex matches the INNER $...$ pair and leaves two literal
-- dollar signs visible. Split such paragraphs so display math stands alone.
local function split_display_math(el, ctor)
  local blocks, cur, found = {}, {}, false
  for _, inl in ipairs(el.content) do
    if inl.t == "Math" and inl.mathtype == "DisplayMath" then
      found = true
      if #cur > 0 then blocks[#blocks + 1] = ctor(cur); cur = {} end
      blocks[#blocks + 1] = pandoc.Para({ inl })
    else
      cur[#cur + 1] = inl
    end
  end
  if not found then return nil end
  if #cur > 0 then blocks[#blocks + 1] = ctor(cur) end
  return blocks
end

function Para(el)
  return split_display_math(el, pandoc.Para)
end

function Plain(el)
  return split_display_math(el, pandoc.Plain)
end

-- pandoc cannot number equations; hand eq refs to MathJax (tags: "ams").
function Link(el)
  local rt = el.attributes["reference-type"]
  if (rt == "ref" or rt == "eqref") and el.target:sub(1, 4) == "#eq:" then
    local label = el.target:sub(2)
    local cmd = (rt == "eqref") and "\\eqref" or "\\ref"
    return pandoc.Math("InlineMath", cmd .. "{" .. label .. "}")
  end
  return el
end
