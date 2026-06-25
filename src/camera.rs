//! Pan/zoom camera mapping the unbounded world plane to screen pixels.
//!
//! The world unit is the **millimeter** (see `docs/architecture/12-ui-rendering.md`). `zoom` is a
//! unitless magnification, and `ui_scale` is the display's physical pixels per mm at zoom 1.0, so
//! the on-screen pixels-per-mm is `zoom * ui_scale`. Pan is unlimited; zoom is clamped. All screen
//! coordinates are physical pixels, matching winit cursor positions and the wgpu surface size.
//!
//! The canvas does not own the whole window — egui chrome (`docs/architecture/12-ui-rendering.md`)
//! docks panels around a central region. Coordinate mapping is therefore relative to the **canvas
//! rect** `[x, y, w, h]` (physical px, within the surface): the camera's `pan` world point sits at
//! the rect's center, not the window center. Pass the current central rect to the methods below.

pub const MIN_ZOOM: f32 = 0.15;
pub const MAX_ZOOM: f32 = 5.0;

/// Screen-space offset (physical px) applied to a cursor point before mapping it to world space.
/// The GPU rasterizes geometry by sampling at pixel **centers**, so a thin feature (a 1 px wire)
/// appears about half a pixel down-right of its mathematical position; nudging the incoming screen
/// coordinate by the same amount makes hit-testing land where the geometry is actually drawn. This
/// stays on the input side, leaving the renderer a pure projection. See
/// `docs/architecture/12-ui-rendering.md`.
const CURSOR_SS_OFFSET: [f32; 2] = [2.0, 2.0];

/// Logical pixels per millimeter at scale factor 1.0, using the conventional 96 px/inch reference
/// (the same approximation CSS's `px` uses). Best-effort: exact when the OS's logical pixels track
/// real size.
pub const LOGICAL_PX_PER_MM: f32 = 96.0 / 25.4;

#[derive(Clone, Copy)]
pub struct Camera {
    /// World point (millimeters) shown at the center of the viewport.
    pub pan: [f32; 2],
    /// Unitless magnification.
    pub zoom: f32,
    /// Display physical pixels per mm at zoom 1.0 (`scale_factor * LOGICAL_PX_PER_MM`).
    pub ui_scale: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pan: [0.0, 0.0],
            zoom: 1.0,
            ui_scale: LOGICAL_PX_PER_MM,
        }
    }
}

impl Camera {
    /// On-screen physical pixels per world millimeter.
    pub fn px_per_mm(&self) -> f32 {
        self.zoom * self.ui_scale
    }

    /// Map a screen point (physical px, full-window) to world mm, relative to the canvas `rect`
    /// `[x, y, w, h]` (physical px). The `pan` world point maps to the rect's center.
    pub fn screen_to_world(&self, screen: [f32; 2], rect: [f32; 4]) -> [f32; 2] {
        let e = self.px_per_mm();
        let cx = rect[0] + rect[2] * 0.5;
        let cy = rect[1] + rect[3] * 0.5;
        // Offset the screen point before conversion (see `CURSOR_SS_OFFSET`).
        [
            (screen[0] - CURSOR_SS_OFFSET[0] - cx) / e + self.pan[0],
            (screen[1] - CURSOR_SS_OFFSET[1] - cy) / e + self.pan[1],
        ]
    }

    /// Zoom toward `anchor` (a screen point), keeping the world point under it fixed. `rect` is the
    /// canvas rect `[x, y, w, h]` in physical px.
    pub fn zoom_at(&mut self, anchor: [f32; 2], rect: [f32; 4], factor: f32) {
        let before = self.screen_to_world(anchor, rect);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let after = self.screen_to_world(anchor, rect);
        self.pan[0] += before[0] - after[0];
        self.pan[1] += before[1] - after[1];
    }

    /// Pan so a drag of `delta` screen pixels keeps the grabbed world point under the cursor.
    pub fn pan_by_screen(&mut self, delta: [f32; 2]) {
        let e = self.px_per_mm();
        self.pan[0] -= delta[0] / e;
        self.pan[1] -= delta[1] / e;
    }
}
