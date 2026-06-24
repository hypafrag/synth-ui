//! The editable patch graph: node geometry, port positions, hit-testing, and wire edits.
//!
//! Holds the in-memory [`Patch`] (`synth_core::model`) and derives a readable node layout from
//! its `layout` block plus each node's ports (from the module registry). This is the data the
//! renderer and input handling both read; geometry is recomputed on demand (node counts are
//! small for the MVP).

use std::collections::HashMap;

use synth_core::model::{Endpoint, Node, Patch, Wire};
use synth_core::module::{Registry, SignalKind};

// Node geometry constants, in world units. Shared by hit-testing and rendering so they agree.
const NODE_W: f32 = 150.0;
const HEADER_H: f32 = 26.0;
const PORT_ROW: f32 = 22.0;
const PAD_BOTTOM: f32 = 10.0;
pub const PORT_R: f32 = 6.0;
const PORT_HIT_R: f32 = 10.0;

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

    /// Compute geometry for every node from the layout block (center positions). Nodes without a
    /// layout entry get a deterministic staggered fallback so they are at least visible.
    pub fn geoms(&self) -> Vec<NodeGeom> {
        let mut out = Vec::with_capacity(self.patch.nodes.len());
        for (i, node) in self.patch.nodes.iter().enumerate() {
            let (ins, outs, known) = self.node_ports(node);
            let rows = ins.len().max(outs.len()).max(1) as f32;
            let h = HEADER_H + rows * PORT_ROW + PAD_BOTTOM;
            let center = self.patch.layout.get(&node.id).map(|p| [p[0] as f32, p[1] as f32]).unwrap_or_else(|| {
                [((i % 5) as f32) * 200.0 - 400.0, ((i / 5) as f32) * 200.0 - 200.0]
            });
            let rect = Rect {
                x: center[0] - NODE_W * 0.5,
                y: center[1] - h * 0.5,
                w: NODE_W,
                h,
            };
            let port_y = |idx: usize| rect.y + HEADER_H + idx as f32 * PORT_ROW + PORT_ROW * 0.5;
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
                known,
                rect,
                inputs,
                outputs,
            });
        }
        out
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
                if d2 <= PORT_HIT_R * PORT_HIT_R && best.as_ref().map_or(true, |(bd, _)| d2 < *bd) {
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

    /// Resolve each wire to its `(from_pos, to_pos, kind)` world endpoints for drawing.
    pub fn wire_segments(&self, geoms: &[NodeGeom]) -> Vec<([f32; 2], [f32; 2], SignalKind)> {
        let by_id: HashMap<&str, &NodeGeom> = geoms.iter().map(|g| (g.id.as_str(), g)).collect();
        let mut segs = Vec::new();
        for w in &self.patch.wires {
            let from = by_id.get(w.from.node());
            let to = by_id.get(w.to.node());
            if let (Some(fg), Some(tg)) = (from, to) {
                let fp = fg.outputs.iter().find(|p| p.name == w.from.port());
                let tp = tg.inputs.iter().find(|p| p.name == w.to.port());
                if let (Some(fp), Some(tp)) = (fp, tp) {
                    segs.push((fp.pos, tp.pos, fp.kind));
                }
            }
        }
        segs
    }
}
