import { describe, expect, it } from "vitest";
import PALETTE from "../../lensmark/schema/palette.json";
import STYLE from "../../lensmark/schema/style_defaults.json";
import * as g from "./geometry";
import type { Arrow, Item, LensMarkFile, MaskCircle, PaletteDoc, StyleDefaults } from "./types";

const style = STYLE as StyleDefaults;
const palette = PALETTE as PaletteDoc;
const W = 403, H = 403, m = 403;

function arrow(over: Partial<Arrow> = {}): Arrow {
  return {
    id: "ann-arrow-001", type: "arrow", tail: [0.4, 0.7], head: [0.45, 0.58], color: "cyan", label: "tight arc",
    label_anchor: "auto", show_in_legend: true, created_by: { kind: "human" }, created_at: "t", status: "accepted", ...over,
  };
}
function mask(over: Partial<MaskCircle> = {}): MaskCircle {
  return {
    id: "ann-mask-001", type: "mask_circle", center: [0.1, 0.1], radius_arcsec: 1.0, kind: "galaxy", color: "mask_red",
    show_in_legend: false, created_by: { kind: "human" }, created_at: "t", status: "accepted", ...over,
  };
}
function file(items: Item[], legend: Partial<LensMarkFile["legend"]> = {}): LensMarkFile {
  return {
    schema_version: "lensmark/1.0", id: "deck-01", created: "t", modified: "t",
    image: { file: "deck-01.png", sha256: "0".repeat(64), width: W, height: H, cutout_arcsec: 16, pixel_scale_arcsec: 16 / W, array_origin: "upper", north_up: true, east_left: true },
    system: { theta_e: {}, description: "", description_refs: [], tags: [] }, palette: "lensmark/v1", style_defaults: style,
    legend: { show: true, position: "auto", ...legend }, items, provenance: { proposal_runs: [], critiques: [] },
  };
}

describe("arrowGeometry", () => {
  it("apex sits tip_gap*m short of the head, head base is head_w*m wide", () => {
    const a = g.arrowGeometry([100, 300], [100, 100], style, m);
    expect(a.dir).toEqual([0, -1]);
    expect(a.apex[0]).toBeCloseTo(100, 6);
    expect(a.apex[1]).toBeCloseTo(100 + style.arrow.tip_gap * m, 6);
    const [apex, b1, b2] = a.head;
    expect(apex).toEqual(a.apex);
    expect(Math.hypot(b1[0] - b2[0], b1[1] - b2[1])).toBeCloseTo(style.arrow.head_w * m, 6);
    // base centre is head_len*m behind the apex, and the shaft ends there
    const bc: [number, number] = [(b1[0] + b2[0]) / 2, (b1[1] + b2[1]) / 2];
    expect(Math.hypot(bc[0] - apex[0], bc[1] - apex[1])).toBeCloseTo(style.arrow.head_len * m, 6);
    expect([a.x2, a.y2]).toEqual(bc);
    expect(a.lineW).toBeCloseTo(style.arrow.line_w * m, 9);
  });
  it("degenerate arrows do not produce NaN", () => {
    const a = g.arrowGeometry([50, 50], [50, 50], style, m);
    expect(Number.isFinite(a.apex[0]) && Number.isFinite(a.head[1][1])).toBe(true);
  });
});

describe("stroke patterns", () => {
  it("galaxy dashes, artifact half dashes, star dots", () => {
    const gal = g.maskPattern("galaxy", style, m);
    const art = g.maskPattern("artifact", style, m);
    const star = g.maskPattern("star", style, m);
    expect(gal.dasharray.split(" ").map(Number)).toEqual([+(style.mask_galaxy.dash_len * m).toFixed(3), +(style.mask_galaxy.gap_len * m).toFixed(3)]);
    expect(Number(art.dasharray.split(" ")[0])).toBeCloseTo(style.mask_galaxy.dash_len * m / 2, 2);
    expect(star.dasharray.startsWith("0 ")).toBe(true);
    expect(star.linecap).toBe("round");
    expect(star.strokeWidth).toBeCloseTo(2 * style.mask_star.dot_r * m, 9);
    expect(Number(star.dasharray.split(" ")[1])).toBeCloseTo((2 + style.mask_star.gap_mult) * style.mask_star.dot_r * m, 2);
    expect(gal.dasharray).not.toEqual(star.dasharray);
  });
  it("ring dots use einstein_ring.dot_r", () => {
    const r = g.ringPattern(style, m);
    expect(r.strokeWidth).toBeCloseTo(2 * style.einstein_ring.dot_r * m, 9);
    expect(Number(r.dasharray.split(" ")[1])).toBeCloseTo(5 * style.einstein_ring.dot_r * m, 2);
  });
});

describe("labels", () => {
  it("tail-side by default, offset beyond the tail along head->tail", () => {
    const a = arrow({ tail: [0.5, 0.8], head: [0.5, 0.5] });
    const p = g.labelPlacement(a, style, W, H)!;
    expect(p.side).toBe("tail");
    expect(p.x).toBeCloseTo(0.5 * W, 6);
    expect(p.y).toBeCloseTo(0.8 * H + style.label.offset * m, 3);
  });
  it("auto falls back to the head side when the tail box would leave the image", () => {
    const a = arrow({ tail: [0.5, 0.995], head: [0.5, 0.7] });
    const p = g.labelPlacement(a, style, W, H)!;
    expect(p.side).toBe("head");
    expect(p.y).toBeLessThan(0.7 * H);
  });
  it("explicit head anchor + label_offset nudge, always clamped inside the image", () => {
    const a = arrow({ tail: [0.2, 0.2], head: [0.02, 0.02], label_anchor: "head", label_offset: [0.01, 0] });
    const p = g.labelPlacement(a, style, W, H)!;
    expect(p.x - p.w / 2).toBeGreaterThanOrEqual(g.LABEL_MARGIN_PX + 0.01 * W - 1e-6);
    expect(p.y - p.h / 2).toBeGreaterThanOrEqual(g.LABEL_MARGIN_PX - 1e-6);
  });
  it("no label -> null", () => {
    expect(g.labelPlacement(arrow({ label: "" }), style, W, H)).toBeNull();
  });
});

describe("theta label", () => {
  it("fmtG2 matches Python's .2g", () => {
    expect(g.fmtG2(1.5)).toBe("1.5");
    expect(g.fmtG2(2.0)).toBe("2");
    expect(g.fmtG2(2.13)).toBe("2.1");
    expect(g.fmtG2(2.92)).toBe("2.9");
    expect(g.fmtG2(0.05)).toBe("0.05");
    expect(g.fmtG2(12)).toBe("12");
    expect(g.thetaLabelText(1.5)).toBe("θ_E ≈ 1.5″");
    expect(g.thetaLabelText(1.5, "custom")).toBe("custom");
  });
  it("sits below-right at r + offset*m", () => {
    const p = g.thetaLabelPlacement([200, 200], 40, style, W, H, "θ_E ≈ 1.5″");
    const d = 40 + style.theta_label.offset * m;
    expect(p.x).toBeCloseTo(200 + d * Math.SQRT1_2, 6);
    expect(p.y).toBeCloseTo(200 + d * Math.SQRT1_2, 6);
  });
});

describe("legend", () => {
  it("auto corner = emptiest quadrant, ties -> top_left", () => {
    expect(g.legendCorner(file([]))).toBe("top_left");
    const f = file([arrow({ tail: [0.2, 0.2], head: [0.3, 0.3] }), mask({ center: [0.8, 0.2] }), mask({ id: "ann-mask-002", center: [0.2, 0.8] })]);
    expect(g.legendCorner(f)).toBe("bottom_right");
    expect(g.legendCorner(file([], { position: "bottom_left" }))).toBe("bottom_left");
  });
  it("rows follow legend.order then file order; masks and unlabelled items are skipped; rejected hidden", () => {
    const f = file([
      arrow({ id: "a1", label: "arc" }), arrow({ id: "a2", label: "deflector", color: "green" }),
      arrow({ id: "a3", label: "nope", status: "rejected" }), arrow({ id: "a4", label: "" }), mask({ label: "m" }),
    ], { order: ["a2"] });
    const rows = g.legendRows(f);
    expect(rows.map((r) => r.id)).toEqual(["a2", "a1"]);
    expect(rows[0].text).toBe("→ deflector");
    const lay = g.legendLayout(f)!;
    expect(lay.corner).toBe("top_right");           // both arrows sit top-left
    expect(lay.x + lay.w).toBeCloseTo(W - 2 * style.legend.pad * m, 6);
    expect(lay.h).toBeCloseTo(2 * style.legend.line_h * style.legend.size * m + 2 * style.legend.pad * m, 6);
  });
  it("hidden legend -> null", () => {
    expect(g.legendLayout(file([arrow()], { show: false }))).toBeNull();
  });
});

describe("ids and colours", () => {
  it("nextId fills the first gap like model.py", () => {
    expect(g.nextId([], "arrow")).toBe("ann-arrow-001");
    expect(g.nextId([arrow({ id: "ann-arrow-001" }), arrow({ id: "ann-arrow-003" })], "arrow")).toBe("ann-arrow-002");
    expect(g.nextId([], "mask_circle")).toBe("ann-mask-001");
    expect(g.nextId([], "einstein_ring")).toBe("ann-ring-001");
    expect(g.nextId([], "text")).toBe("ann-text-001");
  });
  it("arrow colours: deflector -> green, else next unused in arrow_order", () => {
    expect(g.nextArrowColor([], palette, "the deflector")).toBe("green");
    expect(g.nextArrowColor([], palette)).toBe("magenta");
    expect(g.nextArrowColor([arrow({ color: "magenta" })], palette)).toBe("cyan");
    const all = palette.arrow_order.filter((c) => c !== "green").map((c, i) => arrow({ id: `a${i}`, color: c }));
    expect(g.nextArrowColor(all, palette)).toBe("magenta");    // wraps to the least used
  });
  it("edit snapshots + delta", () => {
    const a = arrow({ edit_of: { tail: [0.4, 0.7], head: [0.45, 0.58] }, head: [0.45 + 0.1, 0.58] });
    expect(g.geometryOf(arrow())).toEqual({ tail: [0.4, 0.7], head: [0.45, 0.58] });
    expect(g.deltaArcsec(a, W, H, 16)).toBeCloseTo(1.6, 6);
    expect(g.deltaArcsec(arrow(), W, H, 16)).toBeNull();
  });
});
