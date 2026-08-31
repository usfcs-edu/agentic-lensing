"""Deterministic PIL renderer: ``<id>.lensmark.json`` + original -> ``<id>.annot.png`` (see CONTRACT.md)."""
from .draw import (cli_render, is_stale, label_boxes, png_bytes, render_image, render_png_bytes,  # noqa: F401
                   render_to_file, visible_items)
