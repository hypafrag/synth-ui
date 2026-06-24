//! Pan/zoom camera mapping the unbounded world plane to screen pixels.
//!
//! Pan is unlimited; zoom is clamped to a fixed range (see
//! `docs/architecture/12-ui-rendering.md`). All screen coordinates are physical pixels, matching
//! winit cursor positions and the wgpu surface size.

pub const MIN_ZOOM: f32 = 0.15;
pub const MAX_ZOOM: f32 = 5.0;

#[derive(Clone, Copy)]
pub struct Camera {
    /// World point shown at the center of the viewport.
    pub pan: [f32; 2],
    pub zoom: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            pan: [0.0, 0.0],
            zoom: 1.0,
        }
    }
}

impl Camera {
    pub fn screen_to_world(&self, screen: [f32; 2], viewport: [f32; 2]) -> [f32; 2] {
        [
            (screen[0] - viewport[0] * 0.5) / self.zoom + self.pan[0],
            (screen[1] - viewport[1] * 0.5) / self.zoom + self.pan[1],
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
        self.pan[0] -= delta[0] / self.zoom;
        self.pan[1] -= delta[1] / self.zoom;
    }
}
