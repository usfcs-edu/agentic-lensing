/**
 * Plain-object types mirroring lensmark/model.py (pydantic, extra="forbid") and the HTTP shapes in
 * API.md. The browser state IS the LensMarkFile JSON; every key written here must exist in model.py
 * or the PUT is rejected with 422.
 */

// ----------------------------------------------------------------------------- enums
export type ColorName = "magenta" | "cyan" | "green" | "yellow" | "white" | "orange" | "gray" | "mask_red" | "ring_white";
export type Effort = "low" | "medium" | "high" | "xhigh" | "max";
export type Status = "proposed" | "accepted" | "edited" | "rejected" | "invalid";
export type Verdict = "correct" | "wrong_position" | "wrong_label" | "wrong_type" | "wrong_size" | "spurious" | "redundant" | "missed_by_model";
export type ItemType = "arrow" | "mask_circle" | "einstein_ring" | "text";
export type MaskKind = "galaxy" | "star" | "artifact";
export type LegendPosition = "auto" | "top_left" | "top_right" | "bottom_left" | "bottom_right";
export type Grade = "A" | "B" | "C" | "D";
export type SystemVerdict = "likely_lens" | "possible" | "not_lens" | "unclear";
export type UV = [number, number];

export const VERDICTS: Verdict[] = ["correct", "wrong_position", "wrong_label", "wrong_type", "wrong_size", "spurious", "redundant", "missed_by_model"];
/** Keyboard 1-7 -> verdict (CONTRACT.md "Keyboard"). */
export const KEY_VERDICTS: Verdict[] = ["correct", "wrong_position", "wrong_label", "wrong_type", "wrong_size", "spurious", "redundant"];
export const LEGEND_POSITIONS: LegendPosition[] = ["auto", "top_left", "top_right", "bottom_left", "bottom_right"];
export const CORNERS: Exclude<LegendPosition, "auto">[] = ["top_left", "top_right", "bottom_left", "bottom_right"];
export const SYSTEM_VERDICTS: SystemVerdict[] = ["likely_lens", "possible", "not_lens", "unclear"];
export const THETA_E_VERDICTS = ["correct", "too_small", "too_large", "missing", "spurious"] as const;
export type ThetaEVerdict = (typeof THETA_E_VERDICTS)[number];

// ----------------------------------------------------------------------------- provenance / review
export interface CreatedBy {
  kind: "human" | "claude" | "import" | "voice";
  model?: string | null;
  effort?: Effort | null;
  run_id?: string | null;
  reviewer?: string | null;
}

export interface Review {
  verdict: Verdict;
  severity?: "minor" | "major" | null;
  comment?: string;
  reviewer?: string | null;
  reviewed_at?: string | null;
  delta_arcsec?: number | null;
}

// ----------------------------------------------------------------------------- items
export interface ItemBase {
  id: string;
  label?: string | null;
  color: ColorName;
  show_in_legend: boolean;
  style?: Record<string, unknown> | null;
  created_by: CreatedBy;
  created_at: string;
  status: Status;
  invalid_reason?: string | null;
  edit_of?: Record<string, unknown> | null;
  review?: Review | null;
  notes?: string | null;
}

export interface Arrow extends ItemBase {
  type: "arrow";
  tail: UV;
  head: UV;
  label_anchor: "tail" | "head" | "auto";
  label_offset?: UV | null;
}

export interface MaskCircle extends ItemBase {
  type: "mask_circle";
  center: UV;
  radius_arcsec: number;
  kind: MaskKind;
}

export interface EinsteinRing extends ItemBase {
  type: "einstein_ring";
  center: UV;
  theta_e_arcsec: number;
  center_ref?: string | null;
  label_pos?: UV | null;
}

export interface TextNote extends ItemBase {
  type: "text";
  pos: UV;
  text: string;
}

export type Item = Arrow | MaskCircle | EinsteinRing | TextNote;

// ----------------------------------------------------------------------------- file blocks
export interface Wcs { ra_deg: number; dec_deg: number; rot_deg?: number }

export interface ImageMeta {
  file: string;
  sha256: string;
  width: number;
  height: number;
  cutout_arcsec: number;
  pixel_scale_arcsec: number;
  native_pixel_scale_arcsec?: number | null;
  array_origin: "upper" | "lower";
  north_up: boolean;
  east_left: boolean;
  survey?: string | null;
  instrument?: string | null;
  filters?: string[] | null;
  wcs?: Wcs | null;
  render_recipe?: Record<string, unknown> | null;
  scale_source?: "config" | "override" | "header" | "assumed" | null;
}

export interface ThetaE {
  value_arcsec?: number | null;
  method?: string | null;
  alt_arcsec?: number | null;
  uncertainty_arcsec?: number | null;
}

export interface SystemBlock {
  object_id?: string | null;
  rank?: number | null;
  grade?: Grade | null;
  score_1_4?: number | null;
  confidence_lmh?: "L" | "M" | "H" | null;
  p_lens?: number | null;
  confidence?: number | null;
  theta_e: ThetaE;
  verdict?: SystemVerdict | null;
  description: string;
  description_refs: string[];
  tags: string[];
}

export interface Legend {
  show: boolean;
  position: LegendPosition;
  order?: string[] | null;
}

export interface ProposalRun {
  run_id: string;
  model: string;
  effort?: string | null;
  engine?: string;
  prompt_sha256?: string | null;
  fewshot_sha256?: string | null;
  started_at?: string | null;
  duration_s?: number | null;
  usage?: Record<string, unknown> | null;
  cost_usd?: number | null;
  num_turns?: number | null;
  n_items_proposed?: number;
  n_invalid?: number;
  n_repaired?: number;
  parse_ok?: boolean;
  proposal_file?: string | null;
  error?: string | null;
  proposed_system?: Record<string, unknown> | null;
}

export interface Provenance {
  proposal_runs: ProposalRun[];
  critiques: string[];
  log?: string | null;
}

export interface RenderInfo {
  renderer: string;
  output: string;
  of_json_sha256: string;
  rendered_at?: string | null;
}

/** Style constants (fractions of min(W, H)); == lensmark/schema/style_defaults.json. */
export interface StyleDefaults {
  unit: string;
  arrow: { line_w: number; head_len: number; head_w: number; tip_gap: number; default_len: number };
  mask_galaxy: { line_w: number; stroke: string; dash_len: number; gap_len: number };
  mask_star: { line_w: number; stroke: string; dot_r: number; gap_mult: number };
  einstein_ring: { stroke: string; dot_r: number; gap_mult: number };
  text: { size: number; halo_px: number };
  label: { size: number; font: string; halo_px: number; halo: string; offset: number };
  theta_label: { size: number; offset: number };
  legend: { size: number; pad: number; bg: string; fg: string; glyph: string; line_h: number };
}

export interface LensMarkFile {
  schema_version: "lensmark/1.0";
  id: string;
  created: string;
  modified: string;
  image: ImageMeta;
  coordinates?: Record<string, unknown>;
  system: SystemBlock;
  palette: string;
  style_defaults: StyleDefaults;
  legend: Legend;
  items: Item[];
  provenance: Provenance;
  render?: RenderInfo | null;
}

// ----------------------------------------------------------------------------- critique
export interface CritiqueItem {
  item_id: string;
  verdict: Verdict;
  severity?: "minor" | "major" | null;
  comment: string;
  delta_arcsec?: number | null;
}

export interface CritiquePanel {
  completeness?: number | null;
  geometric_accuracy?: number | null;
  label_quality?: number | null;
  description_quality?: number | null;
  theta_e_verdict?: ThetaEVerdict | null;
  theta_e_human_arcsec?: number | null;
  free_text: string;
  would_use_as_fewshot?: boolean | null;
}

export interface Critique {
  schema_version: "lensmark-critique/1.0";
  image_id: string;
  run_id: string;
  model?: string | null;
  effort?: string | null;
  reviewer: string;
  reviewed_at: string;
  lead_time_s?: number | null;
  items: CritiqueItem[];
  panel: CritiquePanel;
  counts: Record<string, number>;
}

// ----------------------------------------------------------------------------- voice / patch
export interface PatchOp {
  op: "add" | "update" | "delete";
  id?: string | null;
  item?: Record<string, unknown> | null;
  set?: Record<string, unknown> | null;
  confidence?: number | null;
  rationale?: string | null;
}

export interface Patch {
  schema_version: string;
  transcript: string;
  ops: PatchOp[];
  clarification?: string | null;
}

// ----------------------------------------------------------------------------- API shapes (API.md)
export interface PaletteDoc {
  version?: string;
  colors: Record<string, string>;
  arrow_order: ColorName[];
  deflector: ColorName;
  reserved: Record<string, ItemType>;
}

export interface StyleResponse { palette: PaletteDoc; style_defaults: StyleDefaults }

export interface ModelInfo { alias: string; id: string; label: string; supports_effort: boolean; price_in: number; price_out: number }
export interface ModelsResponse { models: ModelInfo[]; efforts: Effort[]; default: { model: string; effort: string } }

export interface CampaignConfig {
  schema_version?: string;
  cutout_arcsec: number;
  cutout_arcsec_source?: string;
  reviewer?: string;
  default_model?: string;
  default_effort?: string;
  campaign?: string;
  [k: string]: unknown;
}

export interface HealthResponse { version: string; campaign_dir: string; engine: string; claude_bin?: string | null; claude_version?: string | null; n_images: number }

export interface ImageSummary {
  id: string;
  file: string;
  width: number;
  height: number;
  cutout_arcsec: number;
  scale_source: string;
  has_json: boolean;
  has_annot: boolean;
  annot_stale: boolean;
  n_items: number;
  by_status: Record<string, number>;
  grade?: Grade | null;
  verdict?: SystemVerdict | null;
  theta_e_arcsec?: number | null;
  rank?: number | null;
  modified?: string | null;
  n_proposals: number;
}

export interface PutResponse { ok: boolean; modified: string; render?: RenderInfo | null; lint: string[] }
export interface ExportResponse { files: string[] }
export interface ProposeStartResponse { run_id: string }
export type SsePhase = "queued" | "started" | "thinking" | "partial" | "tool" | "validated" | "done" | "error";
export interface SseEvent { phase: SsePhase; detail?: string; text?: string; cost_usd?: number; n_items?: number; run?: ProposalRun }

export type ToolName = "select" | "arrow" | "galaxy" | "star" | "ring" | "text";
export const TOOLS: ToolName[] = ["select", "arrow", "galaxy", "star", "ring", "text"];
