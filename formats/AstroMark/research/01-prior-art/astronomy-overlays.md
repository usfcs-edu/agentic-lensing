# Prior art: astronomy overlay and region formats

The incumbents AstroMark must live beside — and the domain where the gaps are largest.

## 1. DS9 regions: the format that cannot be replaced and should not be copied

Dominant and immovable. Every viewer reads it. It is also, structurally, the thing AstroMark exists
to improve on.

### The root of it

Six separate things are wrong with the region format, and they share one cause:

> **A DS9 region file is a serialization of what is on the screen, not a record of what is true.**

Seen that way, the individual defects stop looking like oversights and start looking like correct
decisions for a different problem. If the file is *what I drew in this session*, then of course
colour belongs in it, of course window state belongs in it, of course it need not name the image
(you have it open) or the author (it is you), and of course the version is decoration. Every one of
those choices is right for a screen dump and wrong for a scientific record.

That framing is an inference. Everything below it is checkable against the DS9 reference, the
astropy source and JS9's published API.

### The specifics

A line from DS9's own reference test file:

```
circle(202.48643,47.208449,3.9640007") # color=pink width=3 font="times 10 normal roman" text={Circle} tag={foo} tag={foo bar} This is a Comment
```

**38 characters of geometry; 106 characters of portrayal and free text.** Three-quarters of the line
is drawing instructions.

What it lacks entirely: provenance, author, timestamp, review state, uncertainty, a respected version
declaration — and, critically, **any reference to the image it describes.** An `image;`-frame region
file is meaningless without out-of-band knowledge of which image it belongs to.

**The version trap.** DS9 files begin `# Region file format: DS9 version 4.1`, and the dominant
reference parser (astropy `regions`) skips every `#` line except two special cases. The version
declaration is discarded by the reader that matters, so there is no mechanism by which DS9 could ever
ship a breaking change safely.

**The `-` prefix is not negative evidence.** It "flags the region with a boolean NOT for later
analysis" — set arithmetic on a pixel mask, an instruction to exclude those pixels from an aperture.
It says nothing about what the feature *is*. CRTF's `-` is the same. This is worth stating plainly
because it is easy to assume otherwise.

**Exports drop shapes silently.** The reference documents, per target format, long lists of shapes
that are simply *ignored*. Exporting to CIAO drops LINE, VECTOR, PROJECTION, SEGMENT, TEXT, RULER,
COMPASS, ELLIPSE ANNULUS, BOX ANNULUS, EPANDA and BPANDA — with no error. A format whose exporters
lose content without complaining teaches people to distrust the file rather than the tool.

**Window state is filed as record metadata.** astropy `regions` correctly splits `RegionMeta` from
`RegionVisual`, and then files DS9's `select`, `highlite`, `fixed`, `edit`, `move`, `rotate` and
`delete` under **RegionMeta**. Those are GUI affordances — whether you can drag this shape — not
facts about the sky. This is the screen-state thesis showing up in the reference implementation's own
taxonomy.

**`tag={}` is the good part.** Zero or more repeated, unordered labels per region, and JS9 builds a
real boolean algebra over them (`(circle && foo1) || foo2`). Many-to-many grouping — four marks are
images of one source; one mark sits in a blend belonging to two — works natively here, where a
structural container cannot. DS9's own container, `composite`, does *not* survive a round trip:
astropy parses it only to lift metadata onto its members. **The label mechanism survived; the
containment mechanism did not.**

## 2. The same authors got it right elsewhere

**The DS9 Catalog Tool's Symbol Editor stores no colour per row.** It stores a rule evaluated against
the catalogue's own columns — condition `[string equal $Class SNR]` → shape diamond, colour red, size
`$Jmag/2.` — and portrayal falls out of the data. Symbols are converted to regions only at the moment
they must be handed to a dumber consumer.

This is the record/portrayal separation, demonstrated by the same authors, in the same application,
who got it wrong for regions. Take the architecture, not the Tcl: a flat lookup table
`term → {glyph, stroke, dash, colour, z}` rather than a general expression language.

**Which is the fair reading.** DS9 did not fail from ignorance. The same team solved this correctly
for catalogues, at a different time, for a problem framed differently. And two of its choices are
better than most modern formats manage: `tag={}` is stronger grouping than a scalar group id, and the
format proves that a line-oriented text serialization is culturally acceptable in this field — which
is evidence for the compact read surface in `06-llm-ergonomics.md`. The format is thirty years old,
it won, and it is still what everyone reads.

## 3. What JS9 proves about colour carrying meaning

JS9 documents colour as a **first-class region selector**: `JS9.GetRegions("red")`,
`JS9.ListRegions("circle || red")`, `JS9.ChangeRegions("!red", {...})`, with a full boolean algebra.

Once tools grow a query language over colour, **the palette is frozen forever** — changing it breaks
every saved query. This is the concrete downstream cost of letting a hue mean "lens galaxy", and it
is the strongest available argument for the redundant-encoding rule.

JS9 also demonstrates the bolt-on failure: it kept DS9's `# key=value` comment tail *and* added a JSON
object, and documents a "hybrid" line carrying both property systems at once.

## 4. CRTF's one good idea

A single line-level token partitions the file: any line beginning `ann` is an annotation, "used by
display tasks, and are for visual reference only", never fed to masking or measurement. One token,
no schema branching, no separate file.

AstroMark has exactly this need — a scale bar, a compass and a "look here" bracket are portrayal; an
Einstein-radius circle and a mask circle are record. Make it a **required** field with no default, so
omitting it is a schema error rather than a silent promotion of decoration into the record.

## 5. Citizen science, and the strong-lensing cautionary tale

**Space Warps** — the nearest possible neighbour to this project — encoded rejection as *the absence
of markers*: "we interpret no markers being placed as a rejection". The null set carried the
statement "not a lens". **The reasoning behind a negative was never recorded, only its outcome.**
That is the same gap the workshop identified in the current annotations, reached independently by a
strong-lensing project at scale.

**Galaxy Zoo 2** is the positive model, twice over. Its decision tree — 11 tasks, 37 responses, each
with a stable id and a `Next` pointer — is one machine-readable source that mechanically **generates
the released catalogue's column names**, six columns per response, across 304,122 galaxies. And it
encodes non-existence positively: "no bulge" and "star or artifact" are first-class answers with
their own columns, not empty cells.

Its trap, worth avoiding exactly: GZ2 put the human-readable slug *inside* the column name, so
rewording a question would rename every column. Keep the id opaque; hold the slug in a label field.

**Zooniverse** stamps every classification with `workflow_id`, `workflow_version` (minor increments
when task text changes) and `classifier_version`, and its aggregation filters by version *range*.
That is "grade B meant something different before v0.4" as a runtime mechanism. But its export stores
the answer's *display string* rather than an answer id — so rewording a button silently changes the
meaning of every historical record.

## 6. IVOA geometry: the settled answer

**STC-S** tried to put frame, reference position, flavour, fill factor, error, resolution, size, pixel
size and nested set operations *inside* a geometry string. It spent seventeen years as a Working
Draft, contradicted itself in its own text (§3 says region operations are unsupported; §4 defines
them), and was never ratified.

**DALI xtypes** replaced it: geometry is a bare typed numeric array with a one-token discriminator —
`xtype="circle"` with value `148.9 69.1 2.0`. Frames, units and uncertainties are pushed out to
metadata. That is AstroMark's `{kind, u, v, r}` shape, already ratified by a standards body, and it
won a seventeen-year head-to-head against the alternative.
