//! Pan/zoom camera mapping the unbounded world plane to screen pixels.
//!
//! The world unit is the **millimeter** (see `docs/architecture/12-ui-rendering.md`). `zoom` is a
//! unitless magnification, and `ui_scale` is the display's physical pixels per mm at zoom 1.0, so
//! the on-screen pixels-per-mm is `zoom * ui_scale`. Pan is unlimited; zoom is clamped. All screen
//! coordinates are physical pixels, matching winit cursor positions and the wgpu surface size.

pub const MIN_ZOOM: f32 = 0.15;
pub const MAX_ZOOM: f32 = 5.0;

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

    pub fn screen_to_world(&self, screen: [f32; 2], viewport: [f32; 2]) -> [f32; 2] {
        let e = self.px_per_mm();
        [
            (screen[0] - viewport[0] * 0.5) / e + self.pan[0],
            (screen[1] - viewport[1] * 0.5) / e + self.pan[1],
        ]
    }

    /// Zoom toward `anchor` (a screen point), keeping the world point under it fixed.
    pub fn zoom_at(&mut self, anchor: [f32; 2], viewport: [f32; 2], factor: f32) {
        let before = self.screen_to_world(anchor, viewport);
        self.zoom = (self.zoom * factor).clamp(MIN_ZOOM, MAX_ZOOM);
        let after = self.screen_to_world(anchor, viewport);
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
