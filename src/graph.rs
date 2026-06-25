//! The editable patch graph: node geometry, port positions, hit-testing, and wire edits.
//!
//! Holds the in-memory [`Patch`] (`synth_core::model`) and derives a readable node layout from
//! its `layout` block plus each node's ports (from the module registry). This is the data the
//! renderer and input handling both read; geometry is recomputed on demand (node counts are
//! small for the MVP).

use std::collections::HashMap;

use synth_core::model::{Endpoint, Node, Patch, Wire};
use synth_core::modules::icons;
use synth_core::module::{Icon, Registry, SignalKind};

// Node geometry constants, in millimeters (the world unit; see 12-ui-rendering.md). Shared by
// hit-testing, rendering, and autolayout so they agree. Tweak any of these to resize uniformly.
const NODE_W_MM: f32 = 28.0;
pub const HEADER_H_MM: f32 = 6.0;
const PORT_ROW_MM: f32 = 5.0;
/// Padding below the last port row (bottom inner margin).
const PAD_MM: f32 = 2.5;
pub const PORT_R_MM: f32 = 1.4;
const PORT_HIT_R_MM: f32 = 2.6;

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
    pub kind: SignalKind,
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
    pub kind: SignalKind,
    pub pos: [f32; 2],
}

/// A wire resolved to its drawn endpoints, tagged with its index in `patch.wires` so rendering and
/// hit-testing can identify the same wire.
#[derive(Clone, Copy)]
pub struct WireSeg {
    pub index: usize,
    pub a: [f32; 2],
    pub b: [f32; 2],
    pub kind: SignalKind,
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

    /// The ports of a node, as `(name, kind)` for inputs and outputs. Mirrors the engine's
    /// resolution: `audio_output` is the special sink (`ch0..chN`), everything else comes from the
    /// registry descriptor; an unknown type has no ports.
    fn node_ports(&self, node: &Node) -> (Vec<(String, SignalKind)>, Vec<(String, SignalKind)>, bool) {
        if node.ty == "audio_output" {
            let channels = node
                .params
                .get("channels")
                .and_then(|v| v.as_i64())
                .unwrap_or(2)
                .max(1) as usize;
            let inputs = (0..channels)
                .map(|c| (format!("ch{c}"), SignalKind::Sample))
                .collect();
            return (inputs, Vec::new(), true);
        }
        let desc = if let Some(src) = self.registry.source(&node.ty) {
            (src.describe)(&node.params)
        } else if let Some(entry) = self.registry.get(&node.ty) {
            (entry.describe)(&node.params)
        } else {
            return (Vec::new(), Vec::new(), false);
        };
        let inputs = desc.inputs.iter().map(|p| (p.name.clone(), p.kind)).collect();
        let outputs = desc.outputs.iter().map(|p| (p.name.clone(), p.kind)).collect();
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
                .map(|(idx, (name, kind))| PortGeom {
                    name,
                    kind,
                    is_output: false,
                    pos: [rect.x, port_y(idx)],
                })
                .collect();
            let outputs = outs
                .into_iter()
                .enumerate()
                .map(|(idx, (name, kind))| PortGeom {
                    name,
                    kind,
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

    /// The port whose marker is within hit range of `world`, if any.
    pub fn hit_port(&self, world: [f32; 2]) -> Option<PortRef> {
        let mut best: Option<(f32, PortRef)> = None;
        for g in self.geoms() {
            for p in g.inputs.iter().chain(g.outputs.iter()) {
                let d2 = (p.pos[0] - world[0]).powi(2) + (p.pos[1] - world[1]).powi(2);
                if d2 <= PORT_HIT_R_MM * PORT_HIT_R_MM
                    && best.as_ref().map_or(true, |(bd, _)| d2 < *bd)
                {
                    best = Some((
                        d2,
                        PortRef {
                            node: g.id.clone(),
                            port: p.name.clone(),
                            is_output: p.is_output,
                            kind: p.kind,
                            pos: p.pos,
                        },
                    ));
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

    /// Move a node to a new center position (updates the layout block).
    pub fn move_node(&mut self, id: &str, center: [f32; 2]) {
        self.patch
            .layout
            .insert(id.to_string(), [center[0] as f64, center[1] as f64]);
    }

    /// Try to connect two ports. Returns true if a wire was added. One must be an output and the
    /// other an input; kinds must match; self-wires are rejected. A pre-existing wire into the
    /// target input is replaced (one wire per input).
    pub fn try_connect(&mut self, a: &PortRef, b: &PortRef) -> bool {
        let (out, inp) = match (a.is_output, b.is_output) {
            (true, false) => (a, b),
            (false, true) => (b, a),
            _ => return false,
        };
        if out.kind != inp.kind || out.node == inp.node {
            return false;
        }
        self.patch
            .wires
            .retain(|w| !(w.to.node() == inp.node && w.to.port() == inp.port));
        self.patch.wires.push(Wire {
            from: Endpoint(out.node.clone(), out.port.clone()),
            to: Endpoint(inp.node.clone(), inp.port.clone()),
        });
        true
    }

    /// Remove every wire touching `port`. For an input that is its single incoming wire; for an
    /// output it is all fan-out wires from it.
    pub fn disconnect_port(&mut self, port: &PortRef) -> bool {
        let before = self.patch.wires.len();
        if port.is_output {
            self.patch
                .wires
                .retain(|w| !(w.from.node() == port.node && w.from.port() == port.port));
        } else {
            self.patch
                .wires
                .retain(|w| !(w.to.node() == port.node && w.to.port() == port.port));
        }
        self.patch.wires.len() != before
    }

    /// Resolve each wire to its drawn endpoints (tagged with its `patch.wires` index). Wires whose
    /// endpoints can't be resolved (missing node/port) are skipped.
    pub fn wire_segments(&self, geoms: &[NodeGeom]) -> Vec<WireSeg> {
        let by_id: HashMap<&str, &NodeGeom> = geoms.iter().map(|g| (g.id.as_str(), g)).collect();
        let mut segs = Vec::new();
        for (index, w) in self.patch.wires.iter().enumerate() {
            let from = by_id.get(w.from.node());
            let to = by_id.get(w.to.node());
            if let (Some(fg), Some(tg)) = (from, to) {
                let fp = fg.outputs.iter().find(|p| p.name == w.from.port());
                let tp = tg.inputs.iter().find(|p| p.name == w.to.port());
                if let (Some(fp), Some(tp)) = (fp, tp) {
                    segs.push(WireSeg {
                        index,
                        a: fp.pos,
                        b: tp.pos,
                        kind: fp.kind,
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
