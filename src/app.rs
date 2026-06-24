//! The winit application: window, input handling, and per-frame draw.
//!
//! Controls (MVP):
//! - mouse wheel — zoom toward cursor
//! - left-drag empty canvas — pan
//! - left-drag node body — move node
//! - left-drag port → port — connect (output↔input, matching kind)
//! - right-click port — disconnect wires touching it
//! - Space — start/stop audio
//! Hovered module name shows in the window title (no in-canvas text yet).

use std::sync::Arc;

use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::audio::Audio;
use crate::camera::Camera;
use crate::graph::{GraphView, PortRef};
use crate::render::{self, Renderer, build_scene};

#[derive(Clone, Default)]
enum Drag {
    #[default]
    None,
    Pan,
    Node {
        id: String,
        /// world_cursor − node_center at grab time.
        offset: [f32; 2],
    },
    Wire {
        src: PortRef,
    },
}

/// Logical pixels per millimeter at scale factor 1.0, using the conventional 96 px/inch
/// reference (the same approximation CSS uses). Best-effort: exact when the OS's logical pixels
/// track real size.
const LOGICAL_PX_PER_MM: f32 = 96.0 / 25.4;

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    view: GraphView,
    camera: Camera,
    audio: Audio,
    cursor: [f32; 2],
    drag: Drag,
    hover_id: Option<String>,
    /// Whether the cursor is over the play/pause button.
    btn_hover: bool,
    /// Physical pixels per mm: `scale_factor * LOGICAL_PX_PER_MM`.
    ui_scale: f32,
}

impl App {
    pub fn new(view: GraphView) -> Self {
        Self {
            window: None,
            renderer: None,
            view,
            camera: Camera::default(),
            audio: Audio::default(),
            cursor: [0.0, 0.0],
            drag: Drag::None,
            hover_id: None,
            btn_hover: false,
            ui_scale: LOGICAL_PX_PER_MM,
        }
    }

    fn viewport(&self) -> [f32; 2] {
        self.renderer
            .as_ref()
            .map(|r| r.viewport())
            .unwrap_or([1.0, 1.0])
    }

    fn cursor_world(&self) -> [f32; 2] {
        self.camera.screen_to_world(self.cursor, self.viewport())
    }

    fn redraw(&self) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    fn node_center(&self, id: &str) -> Option<[f32; 2]> {
        self.view
            .geoms()
            .into_iter()
            .find(|g| g.id == id)
            .map(|g| g.rect.center())
    }

    fn update_title(&self) {
        let Some(w) = &self.window else { return };
        let state = if self.audio.playing { "playing" } else { "stopped" };
        let hovered = self.hover_id.as_ref().and_then(|id| {
            self.view
                .patch
                .nodes
                .iter()
                .find(|n| &n.id == id)
                .map(|n| format!("   —   {} ({})", n.id, n.ty))
        });
        w.set_title(&format!(
            "synth-ui   [{state}]{}",
            hovered.unwrap_or_default()
        ));
    }

    fn on_cursor(&mut self, p: [f32; 2]) {
        let delta = [p[0] - self.cursor[0], p[1] - self.cursor[1]];
        self.cursor = p;
        let over_btn = self.over_play_button(p);
        if over_btn != self.btn_hover {
            self.btn_hover = over_btn;
            self.redraw();
        }
        let world = self.cursor_world();
        match self.drag.clone() {
            Drag::Pan => {
                self.camera.pan_by_screen(delta);
                self.redraw();
            }
            Drag::Node { id, offset } => {
                self.view
                    .move_node(&id, [world[0] - offset[0], world[1] - offset[1]]);
                self.redraw();
            }
            Drag::Wire { .. } => self.redraw(),
            Drag::None => {
                let hit = self.view.hit_node(world);
                if hit != self.hover_id {
                    self.hover_id = hit;
                    self.update_title();
                    self.redraw();
                }
            }
        }
    }

    /// True if `screen` (physical pixels) is over the play/pause button.
    fn over_play_button(&self, screen: [f32; 2]) -> bool {
        let (x, y, w, h) = render::play_button_rect(self.ui_scale);
        screen[0] >= x && screen[0] <= x + w && screen[1] >= y && screen[1] <= y + h
    }

    fn on_mouse(&mut self, state: ElementState, button: MouseButton) {
        // Toolbar clicks are handled in screen space and never reach the canvas.
        if state == ElementState::Pressed && button == MouseButton::Left {
            if self.over_play_button(self.cursor) {
                self.audio.toggle(&self.view.patch, &self.view.registry);
                self.update_title();
                self.redraw();
                return;
            }
            if self.cursor[1] < render::toolbar_height(self.ui_scale) {
                return; // swallow clicks on the toolbar bar (no pan)
            }
        }

        let world = self.cursor_world();
        match (state, button) {
            (ElementState::Pressed, MouseButton::Left) => {
                if let Some(port) = self.view.hit_port(world) {
                    self.drag = Drag::Wire { src: port };
                } else if let Some(id) = self.view.hit_node(world) {
                    let center = self.node_center(&id).unwrap_or(world);
                    self.drag = Drag::Node {
                        id,
                        offset: [world[0] - center[0], world[1] - center[1]],
                    };
                } else {
                    self.drag = Drag::Pan;
                }
            }
            (ElementState::Pressed, MouseButton::Right) => {
                if let Some(port) = self.view.hit_port(world) {
                    if self.view.disconnect_port(&port) {
                        self.audio
                            .rebuild_if_playing(&self.view.patch, &self.view.registry);
                        self.update_title();
                        self.redraw();
                    }
                }
            }
            (ElementState::Released, MouseButton::Left) => {
                if let Drag::Wire { src } = self.drag.clone() {
                    if let Some(target) = self.view.hit_port(world) {
                        if self.view.try_connect(&src, &target) {
                            self.audio
                                .rebuild_if_playing(&self.view.patch, &self.view.registry);
                        }
                    }
                }
                self.drag = Drag::None;
                self.redraw();
            }
            _ => {}
        }
    }

    fn on_wheel(&mut self, delta: MouseScrollDelta) {
        let step = match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.02,
        };
        let factor = (1.0 + step * 0.12).clamp(0.5, 2.0);
        self.camera.zoom_at(self.cursor, self.viewport(), factor);
        self.redraw();
    }

    fn on_key(&mut self, event: KeyEvent) {
        if event.state == ElementState::Pressed
            && !event.repeat
            && event.physical_key == PhysicalKey::Code(KeyCode::Space)
        {
            self.audio.toggle(&self.view.patch, &self.view.registry);
            self.update_title();
            self.redraw();
        }
    }

    fn draw(&mut self) {
        let world = self.cursor_world();
        let geoms = self.view.geoms();
        let wires = self.view.wire_segments(&geoms);
        let pending = match &self.drag {
            Drag::Wire { src } => Some((src.pos, world)),
            _ => None,
        };
        let hover = match self.drag {
            Drag::None => self.hover_id.as_deref(),
            _ => None,
        };
        let (tris, lines) = build_scene(&geoms, &wires, pending, hover);
        if let Some(r) = &mut self.renderer {
            let ui = render::build_toolbar(r.viewport(), self.audio.playing, self.ui_scale, self.btn_hover);
            r.render(&self.camera, &tris, &lines, &ui);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("synth-ui   [stopped]"))
                .expect("create window"),
        );
        self.ui_scale = window.scale_factor() as f32 * LOGICAL_PX_PER_MM;
        self.renderer = Some(Renderer::new(window.clone()));
        self.window = Some(window);
        self.redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
                self.redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.ui_scale = scale_factor as f32 * LOGICAL_PX_PER_MM;
                self.redraw();
            }
            WindowEvent::RedrawRequested => self.draw(),
            WindowEvent::CursorMoved { position, .. } => {
                self.on_cursor([position.x as f32, position.y as f32]);
            }
            WindowEvent::MouseInput { state, button, .. } => self.on_mouse(state, button),
            WindowEvent::MouseWheel { delta, .. } => self.on_wheel(delta),
            WindowEvent::KeyboardInput { event, .. } => self.on_key(event),
            _ => {}
        }
    }
}
