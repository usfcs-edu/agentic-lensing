// Mirrors tests/test_coords.py against the SAME fixture file, so the Python and TS transforms can
// never drift apart silently.
import { describe, expect, it } from "vitest";
import CASES from "../../tests/fixtures/coords_cases.json";
import * as c from "./coords";

const TOL = 1e-9;

interface Case {
  name: string; W: number; H: number; cutout: number; u: number; v: number;
  north_up: boolean; east_left: boolean; array_origin: string;
  px: number[]; dEdN: number[]; rpa?: number[]; fits?: number[];
}

describe("fixture cases (tests/fixtures/coords_cases.json)", () => {
  for (const cs of CASES.cases as Case[]) {
    it(cs.name, () => {
      const origin = cs.array_origin as c.ArrayOrigin;
      const [x, y] = c.uv_to_px(cs.u, cs.v, cs.W, cs.H);
      expect(Math.abs(x - cs.px[0])).toBeLessThan(1e-6);
      expect(Math.abs(y - cs.px[1])).toBeLessThan(1e-6);
      const [dE, dN] = c.uv_to_dEdN(cs.u, cs.v, cs.W, cs.H, cs.cutout, cs.north_up, cs.east_left);
      expect(Math.abs(dE - cs.dEdN[0])).toBeLessThan(1e-6);
      expect(Math.abs(dN - cs.dEdN[1])).toBeLessThan(1e-6);
      if (cs.rpa) {
        const [r, pa] = c.dEdN_to_rpa(dE, dN);
        expect(Math.abs(r - cs.rpa[0])).toBeLessThan(1e-6);
        expect(Math.abs(pa - cs.rpa[1])).toBeLessThan(1e-6);
      }
      if (cs.fits) {
        const [fx, fy] = c.uv_to_fits(cs.u, cs.v, cs.W, cs.H, origin);
        expect(Math.abs(fx - cs.fits[0])).toBeLessThan(1e-6);
        expect(Math.abs(fy - cs.fits[1])).toBeLessThan(1e-6);
      }
    });
  }
});

describe("arcsec_to_px", () => {
  for (const cs of CASES.arcsec_to_px) {
    it(`${cs.a}" at W=${cs.W}`, () => {
      expect(Math.abs(c.arcsec_to_px(cs.a, cs.W, cs.cutout) - cs.px)).toBeLessThan(TOL);
      expect(Math.abs(c.px_to_arcsec(cs.px, cs.W, cs.cutout) - cs.a)).toBeLessThan(TOL);
    });
  }
});

describe("round trips", () => {
  for (const origin of ["upper", "lower"] as const) {
    for (const eastLeft of [true, false]) {
      it(`origin=${origin} east_left=${eastLeft}`, () => {
        const W = 410, H = 300, cut = 16.0;
        for (let i = 0; i <= 10; i++) {
          for (let j = 0; j <= 10; j++) {
            const u = i / 10, v = j / 10;
            expect(c.dist_uv(c.px_to_uv(...c.uv_to_px(u, v, W, H), W, H), [u, v])).toBeLessThan(TOL);
            expect(c.dist_uv(c.fits_to_uv(...c.uv_to_fits(u, v, W, H, origin), W, H, origin), [u, v])).toBeLessThan(TOL);
            const [dE, dN] = c.uv_to_dEdN(u, v, W, H, cut, true, eastLeft);
            expect(c.dist_uv(c.dEdN_to_uv(dE, dN, W, H, cut, true, eastLeft), [u, v])).toBeLessThan(TOL);
            const [r, pa] = c.dEdN_to_rpa(dE, dN);
            const [dE2, dN2] = c.rpa_to_dEdN(r, pa);
            expect(Math.abs(dE2 - dE)).toBeLessThan(TOL);
            expect(Math.abs(dN2 - dN)).toBeLessThan(TOL);
          }
        }
      });
    }
  }
});

describe("conventions", () => {
  it("flip applied exactly once", () => {
    expect(c.uv_to_fits(0, 0, 10, 10, "upper")[1]).toBeCloseTo(10.5, 9);
    expect(c.uv_to_fits(0, 0, 10, 10, "lower")[1]).toBeCloseTo(0.5, 9);
  });
  it("screen angle: 0 right, 90 up, CCW", () => {
    expect(c.screen_angle_deg(0, 0, 1, 0)).toBeCloseTo(0, 9);
    expect(c.screen_angle_deg(0, 0, 0, -1)).toBeCloseTo(90, 9);
    expect(c.screen_angle_deg(0, 0, -1, 0)).toBeCloseTo(180, 9);
    expect(c.screen_angle_deg(0, 0, 0, 1)).toBeCloseTo(270, 9);
  });
  it("PA wraps into [0, 360) like Python's %", () => {
    expect(c.pymod(-90, 360)).toBe(270);
    expect(c.dEdN_to_rpa(-4, 0)[1]).toBeCloseTo(270, 9);
  });
  it("polar about an arbitrary centre", () => {
    const [r, pa] = c.uv_to_rpa(0.25, 0.25, 400, 400, 16.0, 0.25, 0.5);
    expect(r).toBeCloseTo(4.0, 9);
    expect(pa).toBeCloseTo(0.0, 9);
    expect(c.dist_arcsec([0.25, 0.5], [0.5, 0.5], 400, 400, 16.0)).toBeCloseTo(4.0, 9);
  });
});
