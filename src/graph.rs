//! The editable patch graph: node geometry, port positions, hit-testing, and wire edits.
//!
//! Holds the in-memory [`Patch`] (`synth_core::model`) and derives a readable node layout from
//! its `layout` block plus each node's ports (from the module registry). This is the data the
//! renderer and input handling both read; geometry is recomputed on demand (node counts are
//! small for the MVP).

use std::collections::HashMap;

use synth_core::model::{Endpoint, Node, Patch, Wire};
use synth_core::modules::icons;
use synth_core::module::{Icon, Inputs, Registry};

// Node geometry constants, in millimeters (the world unit; see 12-ui-rendering.md). Shared by
// hit-testing, rendering, and autolayout so they agree. Tweak any of these to resize uniformly.
const NODE_W_MM: f32 = 28.0;
pub const HEADER_H_MM: f32 = 6.0;
const PORT_ROW_MM: f32 = 5.0;
/// Padding below the last port row (bottom inner margin).
const PAD_MM: f32 = 2.5;
pub const PORT_R_MM: f32 = 1.4;

/// Node box size `[w, h]` in mm for `rows` port rows. The single source of truth for node size,
/// used by both `node_size` (descriptor-driven) and `geoms`.
fn size_from_rows(rows: usize) -> [f32; 2] {
    [
        NODE_W_MM,
        HEADER_H_MM + rows.max(1) as f32 * PORT_ROW_MM + PAD_MM,
    ]
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Rect {
    fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.x && p[0] <= self.x + self.w && p[1] >= self.y && p[1] <= self.y + self.h
    }

    pub fn center(&self) -> [f32; 2] {
        [self.x + self.w * 0.5, self.y + self.h * 0.5]
    }
}

#[derive(Clone)]
pub struct PortGeom {
    pub name: String,
    pub is_output: bool,
    pub pos: [f32; 2],
}

#[derive(Clone)]
pub struct NodeGeom {
    pub id: String,
    pub ty: String,
    pub known: bool,
    pub rect: Rect,
    pub inputs: Vec<PortGeom>,
    pub outputs: Vec<PortGeom>,
}

/// A reference to a concrete port on a node, produced by hit-testing.
#[derive(Clone)]
pub struct PortRef {
    pub node: String,
    pub port: String,
    pub is_output: bool,
    pub pos: [f32; 2],
}

/// A wire resolved to its drawn endpoints, tagged with its index in `patch.wires` so rendering and
/// hit-testing can identify the same wire.
#[derive(Clone, Copy)]
pub struct WireSeg {
    pub index: usize,
    pub a: [f32; 2],
    pub b: [f32; 2],
}

/// Distance from point `p` to the segment `a`–`b`, all in world mm.
fn dist_point_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let (abx, aby) = (b[0] - a[0], b[1] - a[1]);
    let (apx, apy) = (p[0] - a[0], p[1] - a[1]);
    let len2 = abx * abx + aby * aby;
    let t = if len2 > 0.0 {
        ((apx * abx + apy * aby) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (a[0] + t * abx, a[1] + t * aby);
    ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()
}

pub struct GraphView {
    pub patch: Patch,
    pub registry: Registry,
}

impl GraphView {
    pub fn new(patch: Patch) -> Self {
        Self {
            patch,
            registry: Registry::with_builtins(),
        }
    }

    /// The port names of a node, as `(inputs, outputs, known)`. Mirrors the engine's resolution:
    /// `audio_output` is the special sink (`ch0..chN`), everything else comes from the registry
    /// descriptor; an unknown type has no ports.
    fn node_ports(&self, node: &Node) -> (Vec<String>, Vec<String>, bool) {
        if node.ty == "audio_output" {
            let channels = node
                .params
                .get("channels")
                .and_then(|v| v.as_i64())
                .unwrap_or(2)
                .max(1) as usize;
            let inputs = (0..channels).map(|c| format!("ch{c}")).collect();
            return (inputs, Vec::new(), true);
        }
        let desc = if let Some(src) = self.registry.source(&node.ty) {
            (src.describe)(&node.params)
        } else if let Some(entry) = self.registry.get(&node.ty) {
            (entry.describe)(&node.params)
        } else {
            return (Vec::new(), Vec::new(), false);
        };
        let inputs = match desc.inputs {
            Inputs::Fixed(ports) => ports.into_iter().map(|p| p.name).collect(),
            // A variadic port shows one inlet per connected wire plus one empty spare to drag the
            // next wire into; all share the port name. The spare is UI-only — it materializes into
            // the engine only once wired (see `docs/architecture/10-module-contract.md`).
            Inputs::Variadic(port) => {
                let connected = self
                    .patch
                    .wires
                    .iter()
                    .filter(|w| w.to.node() == node.id && w.to.port() == port.name)
                    .count();
                vec![port.name; connected + 1]
            }
        };
        let outputs = desc.outputs.iter().map(|p| p.name.clone()).collect();
        (inputs, outputs, true)
    }

    /// A node's box size `[w, h]` in mm, derived from its descriptor (port counts). The single
    /// entry point for node size — shared by rendering, hit-testing, and autolayout. When custom
    /// module UI exists later, this is where a reported size would be substituted.
    pub fn node_size(&self, node: &Node) -> [f32; 2] {
        let (ins, outs, _) = self.node_ports(node);
        size_from_rows(ins.len().max(outs.len()))
    }

    /// Compute geometry for every node from the layout block (center positions). Nodes without a
    /// layout entry get a deterministic staggered fallback so they are at least visible.
    pub fn geoms(&self) -> Vec<NodeGeom> {
        let mut out = Vec::with_capacity(self.patch.nodes.len());
        for (i, node) in self.patch.nodes.iter().enumerate() {
            let (ins, outs, known) = self.node_ports(node);
            let [w, h] = size_from_rows(ins.len().max(outs.len()));
            let center = self.patch.layout.get(&node.id).map(|p| [p[0] as f32, p[1] as f32]).unwrap_or_else(|| {
                [((i % 5) as f32) * 40.0 - 80.0, ((i / 5) as f32) * 40.0 - 40.0]
            });
            let rect = Rect {
                x: center[0] - w * 0.5,
                y: center[1] - h * 0.5,
                w,
                h,
            };
            let port_y =
                |idx: usize| rect.y + HEADER_H_MM + idx as f32 * PORT_ROW_MM + PORT_ROW_MM * 0.5;
            let inputs = ins
                .into_iter()
                .enumerate()
                .map(|(idx, name)| PortGeom {
                    name,
                    is_output: false,
                    pos: [rect.x, port_y(idx)],
                })
                .collect();
            let outputs = outs
                .into_iter()
                .enumerate()
                .map(|(idx, name)| PortGeom {
                    name,
                    is_output: true,
                    pos: [rect.x + rect.w, port_y(idx)],
                })
                .collect();
            out.push(NodeGeom {
                id: node.id.clone(),
                ty: node.ty.clone(),
                known,
                rect,
                inputs,
                outputs,
            });
        }
        out
    }

    /// Distinct icons for the patch's node types and a `type_id → atlas index` map.
    /// `audio_output` (engine-special) and unknown types get their fallback icons.
    pub fn icon_atlas(&self) -> (Vec<Icon>, HashMap<String, usize>) {
        let mut list = Vec::new();
        let mut map = HashMap::new();
        for node in &self.patch.nodes {
            if map.contains_key(&node.ty) {
                continue;
            }
            let icon = if node.ty == "audio_output" {
                icons::AUDIO_OUTPUT
            } else {
                self.registry.icon(&node.ty).unwrap_or(icons::UNKNOWN)
            };
            map.insert(node.ty.clone(), list.len());
            list.push(icon);
        }
        (list, map)
    }

    /// The topmost node whose body contains `world`, if any (last drawn = topmost).
    pub fn hit_node(&self, world: [f32; 2]) -> Option<String> {
        self.geoms()
            .into_iter()
            .rev()
            .find(|g| g.rect.contains(world))
            .map(|g| g.id)
    }

    /// The port whose marker contains `world`, if any. Tests against the marker's actual visual
    /// quad (a `2·PORT_R_MM` square centered on the port) — no extra hit padding, so the clickable
    /// area is exactly what is drawn. On overlap, the nearest center wins.
    pub fn hit_port(&self, world: [f32; 2]) -> Option<PortRef> {
        let mut best: Option<(f32, PortRef)> = None;
        for g in self.geoms() {
            for p in g.inputs.iter().chain(g.outputs.iter()) {
                let (dx, dy) = (world[0] - p.pos[0], world[1] - p.pos[1]);
                if dx.abs() <= PORT_R_MM && dy.abs() <= PORT_R_MM {
                    let d2 = dx * dx + dy * dy;
                    if best.as_ref().map_or(true, |(bd, _)| d2 < *bd) {
                        best = Some((
                            d2,
                            PortRef {
                                node: g.id.clone(),
                                port: p.name.clone(),
                                is_output: p.is_output,
                                pos: p.pos,
                            },
                        ));
                    }
                }
            }
        }
        best.map(|(_, r)| r)
    }

    /// Add a new node of type `ty` at world `center` (mm), with no params. Returns the generated
    /// id, unique within the patch (`"{ty}-{n}"` for the smallest free `n`). Used by the palette's
    /// drag-to-add. Unknown/incomplete types are allowed — they render with the "unknown" header
    /// until configured, matching how the editor already tolerates partial patches.
    pub fn add_node(&mut self, ty: &str, center: [f32; 2]) -> String {
        let mut n = 1;
        let id = loop {
            let candidate = format!("{ty}-{n}");
            if !self.patch.nodes.iter().any(|node| node.id == candidate) {
                break candidate;
            }
            n += 1;
        };
        self.patch.nodes.push(Node {
            id: id.clone(),
            ty: ty.to_string(),
            params: Default::default(),
        });
        self.patch
            .layout
            .insert(id.clone(), [center[0] as f64, center[1] as f64]);
        id
    }

    /// Whether `port` is the variadic input of node `node_id` — i.e. one that accepts many wires
    /// (each a distinct inlet) rather than a single one. `audio_output` and unknown types have no
    /// variadic inputs.
    fn is_variadic_input(&self, node_id: &str, port: &str) -> bool {
        let Some(node) = self.patch.nodes.iter().find(|n| n.id == node_id) else {
            return false;
        };
        let desc = if let Some(src) = self.registry.source(&node.ty) {
            (src.describe)(&node.params)
        } else if let Some(entry) = self.registry.get(&node.ty) {
            (entry.describe)(&node.params)
        } else {
            return false;
        };
        matches!(desc.inputs, Inputs::Variadic(p) if p.name == port)
    }

    /// Move a node to a new center position (updates the layout block).
    pub fn move_node(&mut self, id: &str, center: [f32; 2]) {
        self.patch
            .layout
            .insert(id.to_string(), [center[0] as f64, center[1] as f64]);
    }

    /// Try to connect two ports. Returns true if a wire was added. One must be an output and the
    /// other an input; self-wires are rejected. A pre-existing wire into the target input is
    /// replaced (one wire per input). All ports carry the one unified channel type, so there is no
    /// kind to match.
    pub fn try_connect(&mut self, a: &PortRef, b: &PortRef) -> bool {
        let (out, inp) = match (a.is_output, b.is_output) {
            (true, false) => (a, b),
            (false, true) => (b, a),
            _ => return false,
        };
        if out.node == inp.node {
            return false;
        }
        // A fixed input takes one wire, so a new connection replaces any existing one. A variadic
        // input instead grows: each wire is a distinct inlet, so we append rather than replace
        // (see `docs/architecture/10-module-contract.md`).
        if !self.is_variadic_input(&inp.node, &inp.port) {
            self.patch
                .wires
                .retain(|w| !(w.to.node() == inp.node && w.to.port() == inp.port));
        }
        self.patch.wires.push(Wire {
            from: Endpoint(out.node.clone(), out.port.clone()),
            to: Endpoint(inp.node.clone(), inp.port.clone()),
        });
        true
    }

    /// The wire drawn into the input inlet under `world`, as its `(from, to)` endpoints — i.e. the
    /// concrete connection identified by *which output feeds it*, not by any positional inlet
    /// number. `None` if no wired inlet is under the cursor. This is how the editor names the wire
    /// to disconnect: a variadic input's inlets are indistinguishable by name, but each carries a
    /// distinct source endpoint, so the source identifies the wire unambiguously.
    pub fn wire_at_input(&self, world: [f32; 2]) -> Option<(Endpoint, Endpoint)> {
        let geoms = self.geoms();
        let segs = self.wire_segments(&geoms);
        segs.iter()
            .filter_map(|s| {
                let (dx, dy) = (world[0] - s.b[0], world[1] - s.b[1]);
                (dx.abs() <= PORT_R_MM && dy.abs() <= PORT_R_MM).then_some((s.index, dx * dx + dy * dy))
            })
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| {
                let w = &self.patch.wires[i];
                (w.from.clone(), w.to.clone())
            })
    }

    /// Remove the wire `from` → `to`, identified by its endpoints. Returns whether one was removed.
    pub fn disconnect_wire(&mut self, from: &Endpoint, to: &Endpoint) -> bool {
        if let Some(i) = self.patch.wires.iter().position(|w| &w.from == from && &w.to == to) {
            self.patch.wires.remove(i);
            true
        } else {
            false
        }
    }

    /// Remove every wire fanning out from output `(node, port)`. Returns whether any were removed.
    pub fn disconnect_output(&mut self, node: &str, port: &str) -> bool {
        let before = self.patch.wires.len();
        self.patch
            .wires
            .retain(|w| !(w.from.node() == node && w.from.port() == port));
        self.patch.wires.len() != before
    }

    /// Resolve each wire to its drawn endpoints (tagged with its `patch.wires` index). Wires whose
    /// endpoints can't be resolved (missing node/port) are skipped.
    pub fn wire_segments(&self, geoms: &[NodeGeom]) -> Vec<WireSeg> {
        let by_id: HashMap<&str, &NodeGeom> = geoms.iter().map(|g| (g.id.as_str(), g)).collect();
        let mut segs = Vec::new();
        // A variadic input has several inlets sharing one name; the n-th wire into that name maps
        // to the n-th inlet. Track, per (node, input port), how many wires we've already placed.
        let mut inlet_seen: HashMap<(&str, &str), usize> = HashMap::new();
        for (index, w) in self.patch.wires.iter().enumerate() {
            let from = by_id.get(w.from.node());
            let to = by_id.get(w.to.node());
            if let (Some(fg), Some(tg)) = (from, to) {
                let fp = fg.outputs.iter().find(|p| p.name == w.from.port());
                let nth = inlet_seen
                    .entry((w.to.node(), w.to.port()))
                    .or_insert(0);
                let tp = tg
                    .inputs
                    .iter()
                    .filter(|p| p.name == w.to.port())
                    .nth(*nth);
                *nth += 1;
                if let (Some(fp), Some(tp)) = (fp, tp) {
                    segs.push(WireSeg {
                        index,
                        a: fp.pos,
                        b: tp.pos,
                    });
                }
            }
        }
        segs
    }

    /// The `patch.wires` index of the wire whose drawn segment is closest to `world`, within
    /// `max_dist` (world mm). `None` if no wire is close enough. `max_dist` is a **world** tolerance;
    /// the caller derives it from a fixed *screen* size and the camera zoom so the hit band is a
    /// constant width on screen regardless of zoom (see `app::WIRE_HIT_HALF_MM`).
    pub fn hit_wire(&self, world: [f32; 2], max_dist: f32) -> Option<usize> {
        let geoms = self.geoms();
        let mut best: Option<(f32, usize)> = None;
        for s in self.wire_segments(&geoms) {
            let d = dist_point_segment(world, s.a, s.b);
            if d <= max_dist && best.map_or(true, |(bd, _)| d < bd) {
                best = Some((d, s.index));
            }
        }
        best.map(|(_, i)| i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_view() -> GraphView {
        GraphView::new(Patch::from_yaml("version: 1\nnodes: []\nwires: []\n").unwrap())
    }

    #[test]
    fn add_node_inserts_with_layout_and_unique_id() {
        let mut view = empty_view();
        let a = view.add_node("sine_generator", [10.0, -5.0]);
        let b = view.add_node("sine_generator", [20.0, 5.0]);

        // Two distinct nodes with distinct ids.
        assert_ne!(a, b);
        assert_eq!(view.patch.nodes.len(), 2);
        assert!(view.patch.nodes.iter().any(|n| n.id == a && n.ty == "sine_generator"));

        // Each id has a layout entry at the requested center.
        assert_eq!(view.patch.layout.get(&a), Some(&[10.0, -5.0]));
        assert_eq!(view.patch.layout.get(&b), Some(&[20.0, 5.0]));
    }

    fn out_port(node: &str) -> PortRef {
        PortRef { node: node.into(), port: "out".into(), is_output: true, pos: [0.0, 0.0] }
    }
    fn in_port(node: &str, port: &str) -> PortRef {
        PortRef { node: node.into(), port: port.into(), is_output: false, pos: [0.0, 0.0] }
    }

    #[test]
    fn variadic_input_appends_each_connection() {
        let mut view = GraphView::new(
            Patch::from_yaml(
                "version: 1\n\
                 nodes:\n\
                 \x20 - { id: a, type: const_generator }\n\
                 \x20 - { id: b, type: const_generator }\n\
                 \x20 - { id: mix, type: mix }\n",
            )
            .unwrap(),
        );

        assert!(view.try_connect(&out_port("a"), &in_port("mix", "in")));
        assert_eq!(view.patch.wires.len(), 1);

        // A second source into the variadic `in` must ADD a wire (a new inlet), not replace the
        // first — every wire to a variadic port is a distinct input.
        assert!(view.try_connect(&out_port("b"), &in_port("mix", "in")));
        assert_eq!(
            view.patch.wires.len(),
            2,
            "second connection to the variadic input replaced the first instead of appending"
        );
        let froms: Vec<&str> = view.patch.wires.iter().map(|w| w.from.node()).collect();
        assert!(froms.contains(&"a") && froms.contains(&"b"));
    }

    #[test]
    fn disconnecting_one_variadic_inlet_keeps_the_others() {
        let mut view = GraphView::new(
            Patch::from_yaml(
                "version: 1\n\
                 nodes:\n\
                 \x20 - { id: a, type: const_generator }\n\
                 \x20 - { id: b, type: const_generator }\n\
                 \x20 - { id: mul, type: mul }\n\
                 wires:\n\
                 \x20 - { from: [a, out], to: [mul, in] }\n\
                 \x20 - { from: [b, out], to: [mul, in] }\n\
                 layout:\n\
                 \x20 a: [-40, -10]\n\
                 \x20 b: [-40, 10]\n\
                 \x20 mul: [0, 0]\n",
            )
            .unwrap(),
        );
        assert_eq!(view.patch.wires.len(), 2);

        // The mul node has two connected inlets (plus a spare). Right-clicking the first inlet
        // resolves to the wire whose source feeds it (a → mul); disconnecting that wire by its
        // endpoints must leave the other connection (b → mul) intact.
        let geoms = view.geoms();
        let mul = geoms.iter().find(|g| g.id == "mul").unwrap();
        let (from, to) = view.wire_at_input(mul.inputs[0].pos).expect("wire at first inlet");
        assert_eq!(from.node(), "a");
        assert_eq!(to.node(), "mul");

        assert!(view.disconnect_wire(&from, &to));
        assert_eq!(
            view.patch.wires.len(),
            1,
            "disconnecting one variadic inlet removed the other connection too"
        );
        // The first connection (a) was removed; the second (b) remains.
        assert_eq!(view.patch.wires[0].from.node(), "b");
    }

    #[test]
    fn fixed_input_replaces_on_reconnect() {
        let mut view = GraphView::new(
            Patch::from_yaml(
                "version: 1\n\
                 nodes:\n\
                 \x20 - { id: a, type: const_generator }\n\
                 \x20 - { id: b, type: const_generator }\n\
                 \x20 - { id: osc, type: sine_generator }\n",
            )
            .unwrap(),
        );

        assert!(view.try_connect(&out_port("a"), &in_port("osc", "frequency")));
        assert!(view.try_connect(&out_port("b"), &in_port("osc", "frequency")));
        // A fixed input takes one wire: the second connection replaces the first.
        assert_eq!(view.patch.wires.len(), 1);
        assert_eq!(view.patch.wires[0].from.node(), "b");
    }

    #[test]
    fn hit_wire_uses_1mm_band() {
        let view = GraphView::new(
            Patch::from_yaml(
                "version: 1\n\
                 nodes:\n\
                 \x20 - { id: a, type: const_generator }\n\
                 \x20 - { id: b, type: sine_generator }\n\
                 wires:\n\
                 \x20 - { from: [a, out], to: [b, frequency] }\n\
                 layout:\n\
                 \x20 a: [-25, 0]\n\
                 \x20 b: [25, 0]\n",
            )
            .unwrap(),
        );
        let geoms = view.geoms();
        let segs = view.wire_segments(&geoms);
        assert_eq!(segs.len(), 1);
        let s = segs[0];
        let mid = [(s.a[0] + s.b[0]) * 0.5, (s.a[1] + s.b[1]) * 0.5];

        // With a 0.5 mm world tolerance: on the wire and just inside hit; well outside misses.
        assert_eq!(view.hit_wire(mid, 0.5), Some(0));
        assert_eq!(view.hit_wire([mid[0], mid[1] + 0.4], 0.5), Some(0));
        assert_eq!(view.hit_wire([mid[0], mid[1] + 5.0], 0.5), None);
    }
}
