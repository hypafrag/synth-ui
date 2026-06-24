//! The winit application: window, egui integration, and per-frame draw.
//!
//! The canvas does not own the window — egui chrome (`crate::chrome`) tiles dockable panels around a
//! fixed central canvas. Canvas pointer interaction is read from an egui interaction `Response`
//! (delivered as [`CanvasInput`]) rather than raw winit events, so it respects which surface the
//! pointer is over and never fights egui's routing. Status shows in the status bar, not the title.
//!
//! Canvas controls:
//! - scroll — zoom toward cursor
//! - left-drag empty canvas — pan
//! - left-drag node body — move node
//! - left-drag port → port — connect (output↔input, matching kind)
//! - right-click port — disconnect wires touching it
//! - Space — start/stop audio · L — autolayout
//! Drag a row from the Modules palette onto the canvas to add that module.

use std::collections::HashMap;
use std::sync::Arc;

use synth_core::model::Patch;
use synth_core::module::Icon;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::audio::Audio;
use crate::camera::{self, Camera};
use crate::chrome::{CanvasInput, Chrome, ChromeAction, ChromeInputs};
use crate::graph::{GraphView, PortRef};
use crate::layout;
use crate::render::{self, EguiFrame, Renderer, build_scene};

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

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    view: GraphView,
    camera: Camera,
    audio: Audio,
    drag: Drag,
    hover_id: Option<String>,
    /// Last canvas pointer position in world mm (for drawing the pending wire).
    canvas_world: [f32; 2],
    /// Module icon atlas: the distinct icons and a `type_id → atlas index` map.
    icon_list: Vec<Icon>,
    icon_index: HashMap<String, usize>,
    // egui chrome integration.
    egui_ctx: egui::Context,
    egui_state: Option<egui_winit::State>,
    chrome: Chrome,
    /// The central canvas region in physical px `[x, y, w, h]`, reported by egui each frame.
    canvas_rect: [f32; 4],
}

/// Physical pixels per mm for the given window scale factor (the chrome/world mm→px scale).
fn ui_scale_for(scale_factor: f64) -> f32 {
    scale_factor as f32 * camera::LOGICAL_PX_PER_MM
}

/// An empty patch, for File → New.
fn empty_patch() -> Patch {
    Patch::from_yaml("version: 1\nnodes: []\nwires: []\n").expect("empty patch parses")
}

impl App {
    pub fn new(view: GraphView) -> Self {
        let (icon_list, icon_index) = view.icon_atlas();
        let chrome = Chrome::new(&view.registry);
        Self {
            window: None,
            renderer: None,
            view,
            camera: Camera::default(),
            audio: Audio::default(),
            drag: Drag::None,
            hover_id: None,
            canvas_world: [0.0, 0.0],
            icon_list,
            icon_index,
            egui_ctx: egui::Context::default(),
            egui_state: None,
            chrome,
            canvas_rect: [0.0, 0.0, 1.0, 1.0],
        }
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

    /// Recompute the module icon atlas (after the set of node types may have changed) and re-upload
    /// it to the renderer.
    fn rebuild_atlas(&mut self) {
        let (list, index) = self.view.icon_atlas();
        self.icon_list = list;
        self.icon_index = index;
        if let Some(r) = &mut self.renderer {
            r.set_icons(&self.icon_list);
        }
    }

    /// Apply this frame's canvas pointer interaction (from egui) to the camera and graph.
    fn process_canvas_input(&mut self, ci: &CanvasInput, ppp: f32) {
        let Some(p_pts) = ci.pointer else {
            // Pointer not over the canvas (and not mid-drag): drop any hover highlight.
            if matches!(self.drag, Drag::None) {
                self.hover_id = None;
            }
            return;
        };
        let phys = [p_pts[0] * ppp, p_pts[1] * ppp];
        let world = self.camera.screen_to_world(phys, self.canvas_rect);
        self.canvas_world = world;

        if ci.drag_started {
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
        } else if ci.dragging {
            match self.drag.clone() {
                Drag::Pan => {
                    self.camera
                        .pan_by_screen([ci.drag_delta[0] * ppp, ci.drag_delta[1] * ppp]);
                }
                Drag::Node { id, offset } => {
                    self.view
                        .move_node(&id, [world[0] - offset[0], world[1] - offset[1]]);
                }
                Drag::Wire { .. } | Drag::None => {}
            }
        }

        if ci.drag_stopped {
            if let Drag::Wire { src } = self.drag.clone() {
                if let Some(target) = self.view.hit_port(world) {
                    if self.view.try_connect(&src, &target) {
                        self.audio
                            .rebuild_if_playing(&self.view.patch, &self.view.registry);
                    }
                }
            }
            self.drag = Drag::None;
        }

        if ci.secondary_clicked {
            if let Some(port) = self.view.hit_port(world) {
                if self.view.disconnect_port(&port) {
                    self.audio
                        .rebuild_if_playing(&self.view.patch, &self.view.registry);
                }
            }
        }

        if ci.scroll_y != 0.0 {
            let factor = (1.0 + ci.scroll_y * 0.0024).clamp(0.5, 2.0);
            self.camera.zoom_at(phys, self.canvas_rect, factor);
        }

        // Hover highlight only when not dragging.
        let hit = if matches!(self.drag, Drag::None) {
            self.view.hit_node(world)
        } else {
            None
        };
        self.hover_id = hit;
    }

    fn on_key(&mut self, event: KeyEvent) {
        if event.state != ElementState::Pressed
            || event.repeat
            || self.egui_ctx.wants_keyboard_input()
        {
            return;
        }
        match event.physical_key {
            PhysicalKey::Code(KeyCode::Space) => {
                self.audio.toggle(&self.view.patch, &self.view.registry);
                self.redraw();
            }
            PhysicalKey::Code(KeyCode::KeyL) => {
                layout::autolayout_full(&mut self.view);
                self.redraw();
            }
            _ => {}
        }
    }

    /// Apply one chrome action (file IO, audio, graph edits). Returns true if the icon atlas needs
    /// rebuilding (the set of node types may have changed).
    fn apply_action(&mut self, el: &ActiveEventLoop, action: ChromeAction, ppp: f32) -> bool {
        match action {
            ChromeAction::ToggleAudio => {
                self.audio.toggle(&self.view.patch, &self.view.registry);
                false
            }
            ChromeAction::Arrange => {
                layout::autolayout_full(&mut self.view);
                false
            }
            ChromeAction::AddNode { ty, screen_pos } => {
                let phys = [screen_pos[0] * ppp, screen_pos[1] * ppp];
                let world = self.camera.screen_to_world(phys, self.canvas_rect);
                self.view.add_node(&ty, world);
                self.audio
                    .rebuild_if_playing(&self.view.patch, &self.view.registry);
                true
            }
            ChromeAction::New => {
                self.view = GraphView::new(empty_patch());
                self.hover_id = None;
                self.audio.stop();
                true
            }
            ChromeAction::Open(path) => {
                match std::fs::read_to_string(&path)
                    .map_err(|e| e.to_string())
                    .and_then(|y| Patch::from_yaml(&y).map_err(|e| e.to_string()))
                {
                    Ok(patch) => {
                        let mut view = GraphView::new(patch);
                        if view.patch.layout.is_empty() {
                            layout::autolayout_full(&mut view);
                        }
                        self.view = view;
                        self.hover_id = None;
                        self.audio
                            .rebuild_if_playing(&self.view.patch, &self.view.registry);
                        true
                    }
                    Err(e) => {
                        eprintln!("open failed: {e}");
                        false
                    }
                }
            }
            ChromeAction::Save(path) => {
                match self
                    .view
                    .patch
                    .to_yaml()
                    .map_err(|e| e.to_string())
                    .and_then(|y| std::fs::write(&path, y).map_err(|e| e.to_string()))
                {
                    Ok(()) => {}
                    Err(e) => eprintln!("save failed: {e}"),
                }
                false
            }
            ChromeAction::Quit => {
                el.exit();
                false
            }
        }
    }

    fn draw(&mut self, el: &ActiveEventLoop) {
        let Some(window) = self.window.clone() else {
            return;
        };
        if self.egui_state.is_none() || self.renderer.is_none() {
            return;
        }

        let raw_input = self.egui_state.as_mut().unwrap().take_egui_input(&window);

        // Run egui, building the chrome. A cloned context avoids borrowing `self.egui_ctx` while the
        // closure mutably borrows `self.chrome` and reads other disjoint fields.
        let ctx = self.egui_ctx.clone();
        let inputs = ChromeInputs {
            registry: &self.view.registry,
            audio_playing: self.audio.playing,
            audio_status: &self.audio.status,
            hover: self.hover_id.as_ref().and_then(|id| {
                self.view
                    .patch
                    .nodes
                    .iter()
                    .find(|n| &n.id == id)
                    .map(|n| (n.id.clone(), n.ty.clone()))
            }),
            node_count: self.view.patch.nodes.len(),
        };
        let mut chrome_out = None;
        let full_output = ctx.run(raw_input, |ctx| {
            chrome_out = Some(self.chrome.build(ctx, &inputs));
        });
        let chrome_out = chrome_out.expect("chrome built");
        drop(inputs);

        self.egui_state
            .as_mut()
            .unwrap()
            .handle_platform_output(&window, full_output.platform_output);

        let ppp = full_output.pixels_per_point;
        // Keep redrawing while egui wants to animate (dock dragging, hovers, etc.).
        let want_continuous = full_output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .map(|v| v.repaint_delay.is_zero())
            .unwrap_or(false);

        // Central canvas region: egui points → physical px.
        self.canvas_rect = match chrome_out.canvas_rect {
            Some(r) => [r.min.x * ppp, r.min.y * ppp, r.width() * ppp, r.height() * ppp],
            None => [0.0, 0.0, 0.0, 0.0],
        };

        // Canvas pointer interaction, then chrome actions; rebuild the icon atlas if types changed.
        self.process_canvas_input(&chrome_out.canvas_input, ppp);
        let mut needs_atlas = false;
        for action in chrome_out.actions {
            needs_atlas |= self.apply_action(el, action, ppp);
        }
        if needs_atlas {
            self.rebuild_atlas();
        }

        let paint_jobs = ctx.tessellate(full_output.shapes, ppp);

        // Build the canvas scene (after edits so a just-added node is included).
        let world = self.canvas_world;
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
        let icons = render::build_icons(&geoms, &self.icon_index, self.icon_list.len());
        let egui_frame = EguiFrame {
            textures_delta: full_output.textures_delta,
            paint_jobs,
            pixels_per_point: ppp,
        };
        if let Some(r) = &mut self.renderer {
            r.render(
                &self.camera,
                self.canvas_rect,
                &tris,
                &lines,
                &icons,
                egui_frame,
            );
        }

        if want_continuous {
            window.request_redraw();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(Window::default_attributes().with_title("synth-ui"))
                .expect("create window"),
        );
        self.camera.ui_scale = ui_scale_for(window.scale_factor());
        let mut renderer = Renderer::new(window.clone());
        renderer.set_icons(&self.icon_list);
        self.egui_state = Some(egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            None,
            None,
        ));
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.redraw();
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Offer every event to egui first (it needs them for its own input + canvas interaction),
        // and honor its repaint requests.
        let window = self.window.clone();
        if let (Some(state), Some(win)) = (self.egui_state.as_mut(), window.as_ref()) {
            let resp = state.on_window_event(win, &event);
            if resp.repaint {
                win.request_redraw();
            }
        }

        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(size) => {
                if let Some(r) = &mut self.renderer {
                    r.resize(size.width, size.height);
                }
                self.redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.camera.ui_scale = ui_scale_for(scale_factor);
                self.redraw();
            }
            WindowEvent::RedrawRequested => self.draw(el),
            WindowEvent::KeyboardInput { event, .. } => self.on_key(event),
            _ => {}
        }
    }
}
