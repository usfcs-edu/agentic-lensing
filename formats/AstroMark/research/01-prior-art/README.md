# Prior art: how other fields solved this

Sixty-six formats surveyed across eight domains — radiology, medical vocabularies, computer vision,
web and geospatial standards, astronomy overlays, astronomy metadata, notation-design theory, and
accessibility. Primary sources wherever they could be reached.

| file | domain |
|---|---|
| [`medical-imaging.md`](medical-imaging.md) | DICOM SR, GSPS, RT Structure Sets, SEG, and NCI AIM — **the closest and most instructive prior art** |
| [`vocabularies-and-grading.md`](vocabularies-and-grading.md) | RadLex, RadElement, BI-RADS / Lung-RADS / PI-RADS, SNOMED, IHE MRRT, OME-Zarr |
| [`astronomy-overlays.md`](astronomy-overlays.md) | DS9 regions, CRTF, astropy `regions`, Aladin/JS9/Firefly, Zooniverse, Space Warps, Galaxy Zoo |
| [`astronomy-metadata.md`](astronomy-metadata.md) | AVM, FITS/WCS, IVOA (VOTable, VOEvent, MOC, DALI, UCD), SIMBAD/CDS |
| [`notation-design.md`](notation-design.md) | Bertin, WMO station model, FGDC geologic symbols, ISO 19117 and 7010, chart symbols, SMuFL, blazon, chess |
| [`computer-vision-and-web.md`](computer-vision-and-web.md) | COCO, VOC, CVAT, Label Studio; W3C Web Annotation, IIIF, SVG, OGC SE, GeoJSON |
| [`accessibility.md`](accessibility.md) | WCAG, CVD simulation models, Okabe-Ito and Tol palettes, cartographic legibility |

---

## Convergences

Mechanisms that unrelated domains arrived at independently. The synthesis found each of these in
between six and ten of the eight domains — a rule invented separately by radiologists, cartographers,
the IVOA and citizen-science engineers is not optional.

### 1. A term is an opaque code; the display string is documentation

DICOM states it normatively: Code Meaning is **non-normative**, identity is scheme + value.
RadElement mints `RDE1695.2` and treats the display name as a mutable annotation. Galaxy Zoo 2 keeps
`t01.a01` stable while the question wording changes. Zooniverse learned this the hard way — its
export stores the *answer display string* `"Yes"` rather than an answer id, so rewording a button
silently changes the meaning of every historical record.

DICOM states it normatively, and then closes the obvious loophole in the very next sentence:

> "the Value of Code Meaning (0008,0104) **shall never be used as a key, index or decision value**,
> rather the combination of Coding Scheme Designator and Code Value may be used. Code Meaning is a
> purely annotative, descriptive Attribute." … "This does not imply that Code Meaning can be filled
> with arbitrary free text."
> — PS3.3 §8.3

**Seven domains, one answer.** The current LensMark design — identity *is* the 40-character English
label, tested by substring match — is precisely the anti-pattern that sentence exists to forbid.

### 2. Portrayal lives outside the record, keyed by term

DICOM GSPS is a separate instance referencing images. OME-Zarr keeps `image-label.colors` parallel to
the labels. ISO 19117 exists solely to separate data from portrayal. CRTF partitions with a single
token: any line beginning `ann` is display-only and never reaches measurement.

The most damning instance is internal: **DS9 got this wrong for regions and right for catalogues in
the same application.** A DS9 region line is 38 characters of geometry and 106 characters of
portrayal — 74% drawing instructions — while the DS9 Catalog Tool's Symbol Editor stores *rules
evaluated against the data* (`[string equal $Class SNR] → diamond, red`) and no row stores a colour.

The root cause is worth naming, because it generalises: **a DS9 region file is a serialization of
what is on the screen, not a record of what is true.** Six separate defects follow from that one
framing, and each is a *correct* decision for a screen dump. See `astronomy-overlays.md` §1.

### 3. Geometry and semantics are two objects joined by a stable id

Medicine settled this three separate times over thirty years: RT Structure Set (1997) splits one
region across three sequences joined by an integer key; AIM (2008) made the split its founding
thesis; DICOM SR does it with relationship types. Zooniverse reached the same place in 2020 via
ADR-25, having shipped the nested version first and measured the pain. IHE MRRT does it. JS9 does it
in miniature with `parent`/`child`.

### 4. Never join by ordinal index

The corollary, and the trap that catches everyone. Zooniverse's ADR-25 diagnosed positional indexing
as the bug — and then **replaced it with `markIndex`, still an ordinal into the value array**, so
deleting mark 1 silently reassigns every downstream answer. Galaxy Zoo 2 embedded the human slug in
its column names, so rewording a question would rename every column.

Use a stable opaque id. Hold the display string separately.

### 5. Absence and uncertainty are coded values, never nulls

DICOM CID 240 is exactly three codes: Present, Absent, **Undetermined** — and AIM's boolean
`isPresent` is the recorded counter-example, because *a boolean cannot say undetermined*. RadElement
goes to four: present, absent, indeterminate (assessed, cannot decide), unknown (not assessed). WMO
code table 2700 distinguishes 9 = "sky obscured" from a missing observation. Galaxy Zoo 2 makes
"no bulge" a first-class answer with its own column rather than an empty cell.

**This is the answer to `counter_image: not_found` versus `not_searched`,** arrived at independently
by three fields.

### 6. Version the vocabulary separately from the schema; retire, never delete

DICOM PS3.3 §8.11 requires receivers to keep recognising retired codes. RadLex Playbook runs a
four-value lifecycle with a TRIAL tier and a written policy that historical codes are never
overwritten. SNOMED records replacements as **typed** links — *same as*, *replaced by*, *possibly
equivalent to* — because "this term was replaced" and "this term was a duplicate" are different
facts. Zooniverse stamps every record with `workflow_version` and filters by version *range*.

### 7. The rich canonical form needs a flat companion, or it does not get written

Seven domains — **and in four of them the flat companion is what actually shipped.** dcmqi's
authoring JSON over DICOM SR; IVOA's `desise` flat JSON over Turtle; DALI's numeric arrays over
STC-S. This is independent confirmation of the read/write asymmetry in `06-llm-ergonomics.md`,
reached by four standards bodies before this project thought of it.

### 8. Generation provenance is three-valued

DICOM reuses **AUTOMATIC / SEMIAUTOMATIC / MANUAL** with identical glosses across four different
objects. The middle value is the one everyone omits and everyone needs.

### 9. Redundant achromatic encoding, in the data model

DICOM Graphic Layers carry a greyscale value **and** a CIELab colour as paired recommended display
values — since ~1999, and DICOM retired its RGB attribute in favour of device-independent CIELab in
2004. FGDC assigns dash pattern to locational accuracy independently of colour. WCAG 1.4.1 states the
rule normatively.

Astronomy has **no prior art here at all** — not one region format addresses colour-vision
deficiency, and DS9 and CRTF both default everything to green — but the channels (point shapes, dash
patterns, line widths) exist and round-trip through every implementation.

---

## Where the domains genuinely disagree

**How heavy should term identity be?** DICOM's four-field coded entry, RadElement's opaque numeric
`RDE1695.2`, or a prefixed string `lens:arc`. The triple is the most rigorous and the most verbose;
`04-metadata-proposals.md` takes the middle path, and §8 of `medical-imaging.md` explains why the
full apparatus is the thing that killed the one project that tried it.

**Should a grade be stored or derived?** BI-RADS stores an assessed category. PI-RADS **derives** the
overall category from component scores via a published lookup table, storing the evidence rather than
only the verdict. For a project trying to make two annotators grade alike, the
derived model is the stronger one — and it is the direct answer to the undefined A/B/D boundaries.

**Nest or flatten?** Computer-vision formats nest semantics inside the annotation; medicine and
citizen science both moved away from nesting after shipping it. The cost of flattening is that a
model emitting the document must keep two id spaces consistent.

**Is colour record or portrayal?** Everyone says portrayal — **except AVM**, whose
`Spectral.ColorAssignment` records that a given data channel *was mapped* to blue when the picture
was made. That is not a contradiction, it is a distinction AstroMark does not currently make: the
colour of a *mark* is portrayal, but which band became which channel in the underlying rendering is
**record**, and belongs with the image metadata. Worth fixing before circulation.

**How many kinds of doubt are there?** FGDC insists on two orthogonal axes, and its standard exists
because the field's informal terms "have not been always clear whether they reflect uncertainty in a
feature's scientific interpretation, its mapped location, or both." AstroMark has a channel for the
first and none for the second.

**Expressiveness versus emittability** — the deepest conflict, and really an argument about *who
writes the file*. DICOM SR, AIM, STC-S and SLD/SE all say model the domain properly. GeoJSON, DALI
and desise all say close the vocabulary and refuse extension. When the writer is a language model
under a schema, the second camp wins.

---

## The finding that sharpened a definition

RadElement separates **presence** (present / absent / indeterminate / unknown) from the **term**
itself, which prompted the question of whether AstroMark's `polarity` was conflating *is this
feature really there?* with *does it bear on the lensing claim?*

On examination it is **not** conflating them, and the two-axis answer is the wrong lesson to draw.
Presence does not vary in this domain: a mark exists because an annotator drew it. Where presence
genuinely does vary it is already carried, better and at the right level, by
`counter_image: found | not_found | not_searched` — a fact about the *search*, not about a mark.

What the objection did correctly expose is narrower and real: **polarity was being demanded of roles
that cannot bear on the claim at all.** A field star or a masked galaxy asserts nothing for or
against lensing, and in a real deck 82% of items are exactly that. The fix is `takes_polarity` on
each role in the vocabulary — required where true, forbidden where false — rather than a fourth
value or a second axis. See `02-requirements.md` §G.6.

---

## Two warnings worth carrying to the review

**A correct design does not win by being correct.** DICOM GSPS has been the right answer since 1999,
is supported by every PACS, and lost in practice to burning annotations into a flattened image
because that was easier. If AstroMark's overlay path is harder than exporting a PNG with marks drawn
on it, people will export the PNG.

**A well-designed standard with no maintainer dies anyway.** The synthesis counted **eight
documented deaths across five domains, and every one of them was well designed** — AIM's reference
repository archived with its last substantive commit in 2014; ACR Assist; STC-S never ratified after
seventeen years. What they lacked was a maintainer, an encoding, a conformance suite, or worked
examples. For a three-person group that is the governing risk, and it is an argument for the smallest
mechanism that holds rather than the most correct one.

**Space Warps encoded "not a lens" as the absence of markers.** In a strong-lensing citizen-science
project — the nearest possible neighbour to this work — a volunteer placing no marks *was* the
rejection. The reasoning behind a negative was never recorded, only its outcome. That is the same gap
the workshop identified in the current annotations, reached independently, and it is the strongest
argument for making negative evidence a first-class mark.
