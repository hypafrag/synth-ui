//! egui chrome: menu bar, toolbar, status bar, and dockable panels around a fixed central
//! **canvas**, laid out with `egui_tiles`.
//!
//! See `docs/architecture/12-ui-rendering.md`. The canvas is a **bare tile** in a tiling tree: it
//! has no tab bar (tab bars exist only inside `Tabs` containers, and the canvas is a plain child of
//! a linear split) and its `pane_ui` returns `UiResponse::None`, so it can never be dragged. Panels
//! (currently the modules palette) live in draggable tab tiles; dragging one shows native drop
//! zones and re-docks it to the canvas's left/right/bottom edge. `egui_tiles` is pure tiling — there
//! are no floating windows — so panels always stay docked.
//!
//! Canvas input is read here through an egui interaction `Response` (returned as [`CanvasInput`]),
//! not via raw winit events, so pan/zoom/drag integrate cleanly with egui's pointer routing.
//!
//! `build` mutates only the dock/icon state it owns and returns the central [`ChromeOut`] (canvas
//! rect + canvas input + a list of [`ChromeAction`]s) for the app to apply, keeping side effects
//! (audio, file IO, graph edits) out of the UI.

use std::collections::HashMap;
use std::path::PathBuf;

use egui_tiles::{
    Behavior, Container, ContainerKind, SimplificationOptions, Tile, TileId, Tiles, Tree, UiResponse,
};

use synth_core::module::{Icon, Registry};
use synth_core::modules::icons;

/// A tile in the dock tree.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Pane {
    Canvas,
    Palette,
}

/// Generic dark "window" background for docked panels, so they read as chrome rather than letting
/// the wgpu canvas show through. Hardcoded (not a platform color) and distinct from the canvas.
const PANEL_BG: egui::Color32 = egui::Color32::from_rgb(32, 34, 38);

/// The egui drag payload carried from a palette row to the canvas: the module type id.
#[derive(Clone)]
struct ModulePayload(String);

/// One thing the app should do as a result of this frame's chrome interaction.
pub enum ChromeAction {
    ToggleAudio,
    Arrange,
    /// A module dropped from the palette onto the canvas at `screen_pos` (egui points).
    AddNode { ty: String, screen_pos: [f32; 2] },
    New,
    Open(PathBuf),
    Save(PathBuf),
    Quit,
}

/// Read-only state the chrome reflects this frame.
pub struct ChromeInputs<'a> {
    pub registry: &'a Registry,
    pub audio_playing: bool,
    pub audio_status: &'a str,
    /// Pre-formatted description of the hovered canvas item, shown in the status bar.
    pub hover: Option<String>,
    pub node_count: usize,
}

/// Pointer interaction over the canvas tile, in egui points (screen space). Driven by an egui
/// `Response`, so it already respects which surface the pointer is over (panels, menus) — the app
/// applies it to the camera/graph.
#[derive(Default, Clone)]
pub struct CanvasInput {
    /// Pointer position while hovering or dragging the canvas, if any.
    pub pointer: Option<[f32; 2]>,
    pub drag_started: bool,
    pub dragging: bool,
    pub drag_stopped: bool,
    /// Incremental primary-drag delta this frame.
    pub drag_delta: [f32; 2],
    pub secondary_clicked: bool,
    /// Raw scroll over the canvas this frame (for zoom).
    pub scroll_y: f32,
}

/// Result of building the chrome for one frame.
pub struct ChromeOut {
    /// The central canvas region in egui points (full-window space).
    pub canvas_rect: Option<egui::Rect>,
    pub canvas_input: CanvasInput,
    pub actions: Vec<ChromeAction>,
}

/// Persistent chrome state across frames: the dock tree, the palette catalog, and cached egui
/// textures for palette icons.
pub struct Chrome {
    tree: Tree<Pane>,
    canvas_id: TileId,
    palette_types: Vec<String>,
    icon_cache: HashMap<String, egui::TextureHandle>,
}

impl Chrome {
    pub fn new(registry: &Registry) -> Self {
        let (tree, canvas_id) = default_tree();
        let mut palette_types: Vec<String> = registry
            .module_type_ids()
            .chain(registry.source_type_ids())
            .map(str::to_string)
            .collect();
        palette_types.push("audio_output".to_string());
        palette_types.sort();
        palette_types.dedup();
        Self {
            tree,
            canvas_id,
            palette_types,
            icon_cache: HashMap::new(),
        }
    }

    pub fn build(&mut self, ctx: &egui::Context, inp: &ChromeInputs) -> ChromeOut {
        let mut actions = Vec::new();

        egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
            egui::menu::bar(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("New").clicked() {
                        actions.push(ChromeAction::New);
                        ui.close_menu();
                    }
                    if ui.button("Open Patch...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("patch", &["yml", "yaml"])
                            .pick_file()
                        {
                            actions.push(ChromeAction::Open(path));
                        }
                        ui.close_menu();
                    }
                    if ui.button("Save Patch As...").clicked() {
                        if let Some(path) = rfd::FileDialog::new()
                            .add_filter("patch", &["yml", "yaml"])
                            .set_file_name("patch.yml")
                            .save_file()
                        {
                            actions.push(ChromeAction::Save(path));
                        }
                        ui.close_menu();
                    }
                    ui.separator();
                    if ui.button("Quit").clicked() {
                        actions.push(ChromeAction::Quit);
                        ui.close_menu();
                    }
                });
                ui.menu_button("View", |ui| {
                    // Toggle the palette by rebuilding the layout with or without it.
                    let mut shown = self
                        .tree
                        .tiles
                        .iter()
                        .any(|(_, t)| matches!(t, Tile::Pane(Pane::Palette)));
                    if ui.checkbox(&mut shown, "Modules palette").changed() {
                        let (tree, canvas_id) = if shown {
                            default_tree()
                        } else {
                            canvas_only_tree()
                        };
                        self.tree = tree;
                        self.canvas_id = canvas_id;
                        ui.close_menu();
                    }
                });
            });
        });

        egui::TopBottomPanel::top("toolbar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let play = if inp.audio_playing { "Stop" } else { "Play" };
                if ui.button(play).clicked() {
                    actions.push(ChromeAction::ToggleAudio);
                }
                if ui.button("Arrange").clicked() {
                    actions.push(ChromeAction::Arrange);
                }
            });
        });

        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let state = if inp.audio_playing { "playing" } else { "stopped" };
                ui.label(state);
                let extra = inp.audio_status;
                if !extra.is_empty() && extra != "playing" && extra != "stopped" {
                    ui.separator();
                    ui.label(extra);
                }
                ui.separator();
                ui.label(format!("{} nodes", inp.node_count));
                if let Some(detail) = &inp.hover {
                    ui.separator();
                    ui.label(detail);
                }
            });
        });

        // The tiling tree fills the central region. A frameless panel paints no background, so the
        // transparent canvas tile reveals the wgpu canvas underneath.
        let mut behavior = TilesBehavior {
            registry: inp.registry,
            palette_types: &self.palette_types,
            icon_cache: &mut self.icon_cache,
            canvas_rect: None,
            canvas_input: CanvasInput::default(),
        };
        egui::CentralPanel::default()
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                self.tree.ui(&mut behavior, ui);
            });
        let canvas_rect = behavior.canvas_rect;
        let canvas_input = behavior.canvas_input;

        // Keep the canvas a bare, tab-less tile: if a panel was dropped dead-center onto it (which
        // would tab-stack it), snap the layout back to default.
        self.rescue_canvas();
        // Keep every panel in a tab tile so it always has a draggable tab bar — dragging an
        // individual tab out drops it as a bare (undraggable) pane otherwise.
        self.ensure_panels_tabbed();

        // Drag-to-add: on release over the canvas with a module payload, emit AddNode.
        if ctx.input(|i| i.pointer.any_released()) {
            if let (Some(rect), Some(pos)) = (canvas_rect, ctx.pointer_interact_pos()) {
                if rect.contains(pos) {
                    if let Some(payload) = egui::DragAndDrop::take_payload::<ModulePayload>(ctx) {
                        actions.push(ChromeAction::AddNode {
                            ty: payload.0.clone(),
                            screen_pos: [pos.x, pos.y],
                        });
                    }
                }
            }
        }

        ChromeOut {
            canvas_rect,
            canvas_input,
            actions,
        }
    }

    /// If the canvas tile has ended up inside a `Tabs` container (e.g. a panel was dropped onto its
    /// center), it would gain a tab bar — restore the default tab-less layout.
    fn rescue_canvas(&mut self) {
        let in_tabs = self
            .tree
            .tiles
            .parent_of(self.canvas_id)
            .and_then(|p| self.tree.tiles.get(p))
            .is_some_and(|t| matches!(t, Tile::Container(c) if c.kind() == ContainerKind::Tabs));
        if in_tabs {
            let (tree, canvas_id) = default_tree();
            self.tree = tree;
            self.canvas_id = canvas_id;
        }
    }

    /// Wrap any panel pane that is not already inside a `Tabs` container in one, so panels always
    /// keep a draggable tab bar. Dropping an *individual* tab to form a new split otherwise leaves a
    /// bare pane, which — since panels aren't body-draggable — would be stuck. The canvas is never
    /// wrapped (it must stay tab-less). Mirrors `egui_tiles`' own in-place wrap: the pane keeps its
    /// tile id by becoming a one-tab `Tabs` container, so the parent's child list is untouched.
    fn ensure_panels_tabbed(&mut self) {
        let to_wrap: Vec<(TileId, Pane)> = self
            .tree
            .tiles
            .iter()
            .filter_map(|(id, tile)| match tile {
                Tile::Pane(p) if *p != Pane::Canvas => {
                    let parent_is_tabs = self
                        .tree
                        .tiles
                        .parent_of(*id)
                        .and_then(|pp| self.tree.tiles.get(pp))
                        .is_some_and(|t| {
                            matches!(t, Tile::Container(c) if c.kind() == ContainerKind::Tabs)
                        });
                    (!parent_is_tabs).then_some((*id, *p))
                }
                _ => None,
            })
            .collect();

        for (id, pane) in to_wrap {
            let inner = self.tree.tiles.insert_pane(pane);
            if let Some(slot) = self.tree.tiles.get_mut(id) {
                *slot = Tile::Container(Container::new_tabs(vec![inner]));
            }
        }
    }
}

/// The default layout: a narrow modules palette (draggable tab) to the left of the canvas.
fn default_tree() -> (Tree<Pane>, TileId) {
    let mut tiles = Tiles::default();
    let canvas = tiles.insert_pane(Pane::Canvas);
    let palette = tiles.insert_pane(Pane::Palette);
    let palette_tab = tiles.insert_tab_tile(vec![palette]);
    let root = tiles.insert_horizontal_tile(vec![palette_tab, canvas]);
    if let Some(Tile::Container(Container::Linear(lin))) = tiles.get_mut(root) {
        lin.shares.set_share(palette_tab, 0.22);
        lin.shares.set_share(canvas, 0.78);
    }
    (Tree::new("synth_dock", root, tiles), canvas)
}

/// Layout with the palette hidden: just the canvas.
fn canvas_only_tree() -> (Tree<Pane>, TileId) {
    let mut tiles = Tiles::default();
    let canvas = tiles.insert_pane(Pane::Canvas);
    (Tree::new("synth_dock", canvas, tiles), canvas)
}

/// The icon for a placeable type: the engine-special sink, a registered module/source, or the
/// unknown fallback. Mirrors `graph::GraphView::icon_atlas`.
fn icon_for(registry: &Registry, ty: &str) -> Icon {
    if ty == "audio_output" {
        icons::AUDIO_OUTPUT
    } else {
        registry.icon(ty).unwrap_or(icons::UNKNOWN)
    }
}

/// A 32×32 monochrome [`Icon`] as an egui image (lit bits → light gray, rest transparent).
fn icon_image(icon: &Icon) -> egui::ColorImage {
    let mut pixels = vec![egui::Color32::TRANSPARENT; 32 * 32];
    for (row, &bits) in icon.iter().enumerate() {
        for x in 0..32usize {
            if (bits >> (31 - x)) & 1 == 1 {
                pixels[row * 32 + x] = egui::Color32::from_gray(225);
            }
        }
    }
    egui::ColorImage {
        size: [32, 32],
        pixels,
    }
}

/// Per-frame `egui_tiles` behavior: renders panes, reports the canvas rect + input, and caches
/// palette icon textures.
struct TilesBehavior<'a> {
    registry: &'a Registry,
    palette_types: &'a [String],
    icon_cache: &'a mut HashMap<String, egui::TextureHandle>,
    canvas_rect: Option<egui::Rect>,
    canvas_input: CanvasInput,
}

impl TilesBehavior<'_> {
    fn icon_tex(&mut self, ui: &egui::Ui, ty: &str) -> egui::TextureId {
        if !self.icon_cache.contains_key(ty) {
            let img = icon_image(&icon_for(self.registry, ty));
            let handle = ui
                .ctx()
                .load_texture(format!("palette-icon-{ty}"), img, egui::TextureOptions::NEAREST);
            self.icon_cache.insert(ty.to_string(), handle);
        }
        self.icon_cache[ty].id()
    }

    fn palette_ui(&mut self, ui: &mut egui::Ui) {
        // 3 mm side insets (UI dims are authored in millimeters; mm → egui points via 96/25.4).
        let inset = 2.0 * crate::camera::LOGICAL_PX_PER_MM;
        egui::Frame::none()
            .inner_margin(egui::Margin::symmetric(inset, 0.0))
            .show(ui, |ui| {
                ui.add_space(2.0);
                ui.label("Drag a module onto the canvas:");
                ui.separator();
                let types = self.palette_types;
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for ty in types {
                        let tex = self.icon_tex(ui, ty);
                        let id = egui::Id::new(("palette-row", ty.as_str()));
                        ui.dnd_drag_source(id, ModulePayload(ty.clone()), |ui| {
                            ui.horizontal(|ui| {
                                ui.add(egui::Image::new(egui::load::SizedTexture::new(
                                    tex,
                                    egui::vec2(18.0, 18.0),
                                )));
                                ui.label(ty.as_str());
                            });
                        });
                    }
                });
            });
    }

    /// Render the canvas tile: claim its rect, read pointer interaction, draw nothing (transparent).
    fn canvas_ui(&mut self, ui: &mut egui::Ui) {
        let rect = ui.max_rect();
        self.canvas_rect = Some(rect);
        let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let pointer = resp
            .interact_pointer_pos()
            .or(resp.hover_pos())
            .map(|p| [p.x, p.y]);
        let scroll_y = if resp.hovered() {
            ui.input(|i| i.raw_scroll_delta.y)
        } else {
            0.0
        };
        let d = resp.drag_delta();
        self.canvas_input = CanvasInput {
            pointer,
            drag_started: resp.drag_started(),
            dragging: resp.dragged(),
            drag_stopped: resp.drag_stopped(),
            drag_delta: [d.x, d.y],
            secondary_clicked: resp.secondary_clicked(),
            scroll_y,
        };
    }
}

impl Behavior<Pane> for TilesBehavior<'_> {
    fn pane_ui(&mut self, ui: &mut egui::Ui, _tile_id: TileId, pane: &mut Pane) -> UiResponse {
        match pane {
            // Canvas stays transparent so the wgpu canvas shows through.
            Pane::Canvas => self.canvas_ui(ui),
            Pane::Palette => {
                ui.painter().rect_filled(ui.max_rect(), 0.0, PANEL_BG);
                self.palette_ui(ui);
            }
        }
        // Neither pane is draggable by its body; panels are dragged via their tab.
        UiResponse::None
    }

    /// Match the tab bar to the panel body color (dark window chrome).
    fn tab_bar_color(&self, _visuals: &egui::Visuals) -> egui::Color32 {
        PANEL_BG
    }

    fn tab_title_for_pane(&mut self, pane: &Pane) -> egui::WidgetText {
        match pane {
            Pane::Canvas => "Canvas".into(),
            Pane::Palette => "Modules".into(),
        }
    }

    fn simplification_options(&self) -> SimplificationOptions {
        SimplificationOptions {
            // Keep a single-panel tab tile so the palette retains a draggable tab; never force the
            // bare canvas pane into a tab.
            prune_single_child_tabs: false,
            all_panes_must_have_tabs: false,
            ..Default::default()
        }
    }

    fn is_tab_closable(&self, _tiles: &Tiles<Pane>, _tile_id: TileId) -> bool {
        false
    }
}
