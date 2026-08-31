/**
 * Coordinate transforms - a one-to-one mirror of lensmark/coords.py (same function names, same
 * conventions). Pure functions; tested against ../../tests/fixtures/coords_cases.json (the same
 * fixture pytest uses) in coords.test.ts.
 *
 * Canonical geometry is normalized (u, v): u in [0, 1] left -> right, v in [0, 1] top -> bottom of
 * the displayed image. Physical sizes are arcsec; pixels are derived, never stored.
 *
 *   display px    x = u * W                y = v * H
 *   pixel scale   ps = cutout_arcsec / W   (arcsec per display pixel)
 *   sky offsets   dx = (u - 0.5) * W * ps  (+right)   dy = (0.5 - v) * H * ps (+up)
 *                 dE = -dx if east_left    dN = dy if north_up
 *   polar         r = hypot(dE, dN)        PA = atan2(dE, dN) deg, North (0) through East (90)
 *   FITS / DS9    1-based, y up: x = u*W + 0.5; y = (1-v)*H + 0.5 (array_origin "upper") else v*H + 0.5
 *   screen angle  0 = +x (right), 90 = up, counter-clockwise (21_annotate.py arrow convention)
 */
export type ArrayOrigin = "upper" | "lower";
export type XY = [number, number];

/** Python-style modulo: the result has the sign of the divisor (JS `%` keeps the dividend's sign). */
export function pymod(a: number, n: number): number {
  return ((a % n) + n) % n;
}

export function uv_to_px(u: number, v: number, W: number, H: number): XY {
  return [u * W, v * H];
}

export function px_to_uv(x: number, y: number, W: number, H: number): XY {
  return [x / W, y / H];
}

/** arcsec per display pixel. */
export function pixel_scale(W: number, cutout_arcsec: number): number {
  return cutout_arcsec / W;
}

export function arcsec_to_px(a: number, W: number, cutout_arcsec: number): number {
  return (a * W) / cutout_arcsec;
}

export function px_to_arcsec(p: number, W: number, cutout_arcsec: number): number {
  return (p * cutout_arcsec) / W;
}

export function uv_to_fits(u: number, v: number, W: number, H: number, array_origin: ArrayOrigin = "upper"): XY {
  const x = u * W + 0.5;
  const y = array_origin === "upper" ? (1.0 - v) * H + 0.5 : v * H + 0.5;
  return [x, y];
}

export function fits_to_uv(x: number, y: number, W: number, H: number, array_origin: ArrayOrigin = "upper"): XY {
  const u = (x - 0.5) / W;
  const v = array_origin === "upper" ? 1.0 - (y - 0.5) / H : (y - 0.5) / H;
  return [u, v];
}

export function uv_to_dEdN(u: number, v: number, W: number, H: number, cutout_arcsec: number,
                           north_up = true, east_left = true): XY {
  const ps = pixel_scale(W, cutout_arcsec);
  const dx = (u - 0.5) * W * ps;
  const dy = (0.5 - v) * H * ps;
  const dE = east_left ? -dx : dx;
  const dN = north_up ? dy : -dy;
  return [dE, dN];
}

export function dEdN_to_uv(dE: number, dN: number, W: number, H: number, cutout_arcsec: number,
                           north_up = true, east_left = true): XY {
  const ps = pixel_scale(W, cutout_arcsec);
  const dx = east_left ? -dE : dE;
  const dy = north_up ? dN : -dN;
  const u = dx / (W * ps) + 0.5;
  const v = 0.5 - dy / (H * ps);
  return [u, v];
}

/** (r arcsec, PA deg from North through East in [0, 360)). */
export function dEdN_to_rpa(dE: number, dN: number): XY {
  const r = Math.hypot(dE, dN);
  const pa = pymod((Math.atan2(dE, dN) * 180) / Math.PI, 360.0);
  return [r, pa];
}

export function rpa_to_dEdN(r: number, pa_deg: number): XY {
  const p = (pa_deg * Math.PI) / 180;
  return [r * Math.sin(p), r * Math.cos(p)];
}

/** Polar coordinates of (u, v) about an arbitrary centre (cu, cv) (default: image centre). */
export function uv_to_rpa(u: number, v: number, W: number, H: number, cutout_arcsec: number,
                          cu = 0.5, cv = 0.5, north_up = true, east_left = true): XY {
  const [dE, dN] = uv_to_dEdN(u, v, W, H, cutout_arcsec, north_up, east_left);
  const [cE, cN] = uv_to_dEdN(cu, cv, W, H, cutout_arcsec, north_up, east_left);
  return dEdN_to_rpa(dE - cE, dN - cN);
}

/** Angle of the vector (x0,y0)->(x1,y1) in screen convention: 0 = right, 90 = up, CCW, [0, 360). */
export function screen_angle_deg(x0: number, y0: number, x1: number, y1: number): number {
  return pymod((Math.atan2(-(y1 - y0), x1 - x0) * 180) / Math.PI, 360.0);
}

export function dist_uv(a: ArrayLike<number>, b: ArrayLike<number>): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** Angular separation between two (u, v) points. */
export function dist_arcsec(a: ArrayLike<number>, b: ArrayLike<number>, W: number, H: number, cutout_arcsec: number): number {
  const ps = pixel_scale(W, cutout_arcsec);
  return Math.hypot((a[0] - b[0]) * W * ps, (a[1] - b[1]) * H * ps);
}
