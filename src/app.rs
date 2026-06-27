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

use synth_core::model::{Node, ParamValue, Patch};
use synth_core::module::{Icon, ParamDesc, ParamKind};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

use crate::audio::Audio;
use crate::camera::{self, Camera};
use crate::chrome::{CanvasInput, Chrome, ChromeAction, ChromeInputs};
use crate::graph::{GraphView, PortRef, HEADER_H_MM, PARAM_ROW_MM, PORT_ROW_MM};
use crate::layout;
use crate::render::{self, EguiFrame, Renderer, SceneHover, build_scene};

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

/// What the cursor is currently hovering on the canvas (when idle).
#[derive(Clone, Default)]
enum Hover {
    #[default]
    None,
    Node(String),
    Port(PortRef),
    /// Index into `patch.wires`.
    Wire(usize),
}

/// Half-width of a wire's hover hit band, in **screen** millimeters (the band is 3 mm wide on
/// screen, constant regardless of zoom). Converted to a world tolerance via the camera zoom: at
/// zoom `z`, one world mm renders as `z` screen mm, so the world tolerance is this divided by `z`.
const WIRE_HIT_HALF_MM: f32 = 1.5;

pub struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    view: GraphView,
    camera: Camera,
    audio: Audio,
    drag: Drag,
    hover: Hover,
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
            hover: Hover::None,
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

    /// Draw the generic in-node param editors over the canvas (`13-ui-module-api.md`): one egui
    /// widget per exposed param, positioned in world space below the node's port rows. Edits
    /// persist into the patch live; the return value is whether an edit was **committed** this frame
    /// (so the caller rebuilds the engine once per edit, not every drag frame). `canvas` is the
    /// canvas region in egui points.
    fn render_param_editors(&mut self, ctx: &egui::Context, canvas: egui::Rect) -> bool {
        let pts_per_mm = self.camera.px_per_mm() / ctx.pixels_per_point();
        // LOD: hide editors when a row would be too small to read/use — they never touch the
        // high-cardinality path (`12-ui-rendering.md`).
        if PARAM_ROW_MM * pts_per_mm < 12.0 {
            return false;
        }
        // Editors live on their own egui layer with a world→screen transform, so egui scales and
        // pans the widgets exactly like the canvas (no per-zoom drift, no pan lag). The layer's
        // coordinate unit is "points at zoom 1" (world mm × `LOGICAL_PX_PER_MM`) so egui's default
        // font/padding read correctly at zoom 1; the transform's scaling is then just the zoom.
        let d = camera::LOGICAL_PX_PER_MM;
        let scaling = pts_per_mm / d;
        let pan = self.camera.pan;
        let center = canvas.center();
        let transform = egui::emath::TSTransform::new(
            egui::vec2(center.x - pts_per_mm * pan[0], center.y - pts_per_mm * pan[1]),
            scaling,
        );
        // Clip to the canvas, expressed in the layer's coordinate space (so it maps back to the
        // canvas rect after the transform — keeps editors from bleeding over the panels).
        let clip = egui::Rect::from_min_max(
            transform.inverse().mul_pos(canvas.min),
            transform.inverse().mul_pos(canvas.max),
        );

        // Geometry + param descriptors per node, index-aligned with `patch.nodes` (geoms preserves
        // node order). Computed before the mutable edit loop to avoid aliasing `view`.
        let geoms = self.view.geoms();
        let descs: Vec<_> = self
            .view
            .patch
            .nodes
            .iter()
            .map(|n| self.view.node_params(n))
            .collect();
        let nodes = &mut self.view.patch.nodes;

        // One Area per node, anchored at that node's param strip. egui decides which layer owns a
        // pointer by an area's *bounding rect*, so each area must be tight to its own widgets — a
        // single shared area (or one anchored away from its content) blankets the canvas with dead
        // zones. We anchor at the node's design-space position, lay widgets out locally, and
        // disable screen-constraining so the layer transform alone places them.
        let mut committed = false;
        for (i, g) in geoms.iter().enumerate() {
            let params = &descs[i];
            if params.is_empty() {
                continue;
            }
            let node = &mut nodes[i];
            let top = g.rect.y + HEADER_H_MM + g.port_rows as f32 * PORT_ROW_MM;
            let row_h = (PARAM_ROW_MM - 1.2) * d;
            let layer = egui::Area::new(egui::Id::new(("node-params", g.id.as_str())))
                .order(egui::Order::Middle)
                .constrain(false)
                .fixed_pos(egui::pos2((g.rect.x + 1.5) * d, (top + 0.6) * d))
                .show(ctx, |ui| {
                    ui.set_clip_rect(clip);
                    ui.set_max_width((g.rect.w - 3.0) * d);
                    ui.spacing_mut().item_spacing.y = 1.2 * d;
                    for pd in params {
                        if editor_widget(ui, pd, node, row_h) {
                            committed = true;
                        }
                    }
                })
                .response
                .layer_id;
            ctx.set_transform_layer(layer, transform);
        }
        committed
    }

    // (param editor widget is a free function below)

    /// Apply this frame's canvas pointer interaction (from egui) to the camera and graph.
    fn process_canvas_input(&mut self, ci: &CanvasInput, ppp: f32) {
        let Some(p_pts) = ci.pointer else {
            // Pointer not over the canvas (and not mid-drag): drop any hover highlight.
            if matches!(self.drag, Drag::None) {
                self.hover = Hover::None;
            }
            return;
        };
        let phys = [p_pts[0] * ppp, p_pts[1] * ppp];
        let world = self.camera.screen_to_world(phys, self.canvas_rect);
        self.canvas_world = world;

        if ci.drag_started {
            // Hit-test at the press origin (where the button went down), not at `world`: egui
            // reports `drag_started` only after the pointer passes its drag threshold, so `world`
            // has already nudged off a small port marker. The press origin is exactly where the
            // user aimed, so a press that began on a port reliably starts a wire.
            let grab = ci
                .press_origin
                .map(|p| self.camera.screen_to_world([p[0] * ppp, p[1] * ppp], self.canvas_rect))
                .unwrap_or(world);
            if let Some(port) = self.view.hit_port(grab) {
                self.drag = Drag::Wire { src: port };
            } else if let Some(id) = self.view.hit_node(grab) {
                let center = self.node_center(&id).unwrap_or(grab);
                self.drag = Drag::Node {
                    id,
                    offset: [grab[0] - center[0], grab[1] - center[1]],
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
                // Resolve the drop at the exact release position, mirroring the press side — not at
                // a possibly-stale `world`.
                let drop = ci
                    .release_pos
                    .map(|p| self.camera.screen_to_world([p[0] * ppp, p[1] * ppp], self.canvas_rect))
                    .unwrap_or(world);
                if let Some(target) = self.view.hit_port(drop) {
                    if self.view.try_connect(&src, &target) {
                        self.audio
                            .rebuild_if_playing(&self.view.patch, &self.view.registry);
                    }
                }
            }
            self.drag = Drag::None;
        }

        if ci.secondary_clicked {
            // Right-click on a wired input inlet removes that one connection, named by its
            // endpoints (which output feeds it) — never a positional inlet index. Right-click on an
            // output removes all its fan-out.
            let changed = if let Some((from, to)) = self.view.wire_at_input(world) {
                self.view.disconnect_wire(&from, &to)
            } else if let Some(port) = self.view.hit_port(world) {
                port.is_output && self.view.disconnect_output(&port.node, &port.port)
            } else {
                false
            };
            if changed {
                self.audio
                    .rebuild_if_playing(&self.view.patch, &self.view.registry);
            }
        }

        if ci.scroll_y != 0.0 {
            let factor = (1.0 + ci.scroll_y * 0.0024).clamp(0.5, 2.0);
            self.camera.zoom_at(phys, self.canvas_rect, factor);
        }

        self.hover = match &self.drag {
            // Idle: prefer the most specific target — a port, then a wire (1 mm band), then the
            // node body.
            Drag::None => {
                if let Some(port) = self.view.hit_port(world) {
                    Hover::Port(port)
                } else if let Some(wire) =
                    self.view.hit_wire(world, WIRE_HIT_HALF_MM / self.camera.zoom)
                {
                    Hover::Wire(wire)
                } else if let Some(id) = self.view.hit_node(world) {
                    Hover::Node(id)
                } else {
                    Hover::None
                }
            }
            // Dragging a wire: highlight a compatible drop target under the cursor — a port of the
            // opposite direction on a different node (the same rule `try_connect` accepts), so
            // outputs light up inputs and inputs light up outputs.
            Drag::Wire { src } => match self.view.hit_port(world) {
                Some(p) if p.is_output != src.is_output && p.node != src.node => Hover::Port(p),
                _ => Hover::None,
            },
            _ => Hover::None,
        };
    }

    /// A human-readable description of the currently hovered item, for the status bar.
    fn hover_detail(&self) -> Option<String> {
        match &self.hover {
            Hover::None => None,
            Hover::Node(id) => self
                .view
                .patch
                .nodes
                .iter()
                .find(|n| &n.id == id)
                .map(|n| format!("node  {}  ({})", n.id, n.ty)),
            Hover::Port(p) => {
                let dir = if p.is_output { "output" } else { "input" };
                Some(format!("{dir}  {}.{}", p.node, p.port))
            }
            Hover::Wire(idx) => {
                let w = self.view.patch.wires.get(*idx)?;
                Some(format!(
                    "wire  {}.{} -> {}.{}",
                    w.from.node(),
                    w.from.port(),
                    w.to.node(),
                    w.to.port()
                ))
            }
        }
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
                self.hover = Hover::None;
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
                        self.hover = Hover::None;
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

        // Status-bar detail for the hovered item (from last frame's hover).
        let hover_detail = self.hover_detail();

        // Run egui, building the chrome. A cloned context avoids borrowing `self.egui_ctx` while the
        // closure mutably borrows `self.chrome` and reads other disjoint fields.
        let ctx = self.egui_ctx.clone();
        let mut chrome_out = None;
        let mut params_committed = false;
        let full_output = ctx.run(raw_input, |ctx| {
            // Build chrome in an inner scope so its borrow of `view`/`audio` ends before the param
            // editors take a mutable borrow of `view`.
            let out = {
                let inputs = ChromeInputs {
                    registry: &self.view.registry,
                    audio_playing: self.audio.playing,
                    audio_status: &self.audio.status,
                    hover: hover_detail.clone(),
                    node_count: self.view.patch.nodes.len(),
                };
                self.chrome.build(ctx, &inputs)
            };
            let ppp = ctx.pixels_per_point();
            // Central canvas region: egui points → physical px.
            self.canvas_rect = match out.canvas_rect {
                Some(r) => [r.min.x * ppp, r.min.y * ppp, r.width() * ppp, r.height() * ppp],
                None => [0.0, 0.0, 0.0, 0.0],
            };
            // Apply this frame's pan/zoom/drag FIRST so the camera is current, then draw the param
            // editors glued to it — both the editors and the wgpu canvas use the same post-input
            // camera this frame, so there is no one-frame lag between them.
            self.process_canvas_input(&out.canvas_input, ppp);
            if let Some(crect) = out.canvas_rect {
                params_committed = self.render_param_editors(ctx, crect);
            }
            chrome_out = Some(out);
        });
        let chrome_out = chrome_out.expect("chrome built");
        // Rebuild only when an edit was committed (drag released / field defocused), so dragging a
        // value recompiles the engine once on release, not every frame.
        if params_committed {
            self.audio
                .rebuild_if_playing(&self.view.patch, &self.view.registry);
        }

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
        // `self.hover` maps straight to the scene highlight: idle hover, or — while dragging a wire
        // — the compatible drop-target port under the cursor.
        let hover = match &self.hover {
            Hover::None => SceneHover::default(),
            Hover::Node(id) => SceneHover {
                node: Some(id.as_str()),
                ..Default::default()
            },
            Hover::Port(p) => SceneHover {
                port: Some((p.node.as_str(), p.port.as_str(), p.is_output)),
                ..Default::default()
            },
            Hover::Wire(i) => SceneHover {
                wire: Some(*i),
                ..Default::default()
            },
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

/// Draw one param's editor (laid out in the current `ui`, sized to the node width × `row_h`),
/// persisting any change into `node.params` immediately so the value tracks the widget. Returns
/// whether the edit was **committed** this frame — the signal to rebuild the engine. A drag commits
/// on release; a typed value commits on Return / losing focus (never per keystroke). Float → a drag
/// value (click to type a value); other kinds are added in the audio_output slice.
fn editor_widget(ui: &mut egui::Ui, pd: &ParamDesc, node: &mut Node, row_h: f32) -> bool {
    match &pd.kind {
        ParamKind::Float { min, max } => {
            let mut v = node
                .params
                .get(&pd.name)
                .and_then(|x| x.as_f64())
                .or_else(|| pd.default.as_f64())
                .unwrap_or(0.0);
            let resp = ui.add_sized(
                [ui.available_width(), row_h],
                egui::DragValue::new(&mut v)
                    .speed(0.01)
                    .range(*min as f64..=*max as f64),
            );
            if resp.changed() {
                node.params.insert(pd.name.clone(), ParamValue::Float(v));
            }
            // Two independent commit triggers: a drag commits on release; a typed value commits on
            // Return / losing focus — never per keystroke.
            resp.drag_stopped() || resp.lost_focus()
        }
        // Int / Bool / Choice editors land with the audio_output dropdowns (next slice).
        _ => false,
    }
}
