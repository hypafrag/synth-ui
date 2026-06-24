//! Full layered (signal-flow) autolayout (`docs/architecture/14-ui-autolayout.md`).
//!
//! Assigns every node a position: break cycles (DFS back edges), longest-path columns with the
//! `audio_output` sink anchored on the right, barycenter crossing reduction, then mm coordinate
//! assignment using each node's `node_size`. Writes the result into `patch.layout`. This is the
//! **full** pass — it positions all nodes; the partial/pinned case is not handled here.

use std::collections::{HashMap, HashSet};

use crate::graph::GraphView;

/// Horizontal gap between columns, in mm (added to each column's width).
const COL_GAP_MM: f32 = 18.0;
/// Vertical gap between stacked nodes in a column, in mm.
const ROW_GAP_MM: f32 = 7.0;
/// Crossing-reduction sweeps (alternating down/up).
const SWEEPS: usize = 4;

/// Re-arrange every node in the patch into a readable layered layout, overwriting `layout`.
pub fn autolayout_full(view: &mut GraphView) {
    let n = view.patch.nodes.len();
    if n == 0 {
        return;
    }

    let sizes: Vec<[f32; 2]> = view.patch.nodes.iter().map(|nd| view.node_size(nd)).collect();
    let index: HashMap<&str, usize> = view
        .patch
        .nodes
        .iter()
        .enumerate()
        .map(|(i, nd)| (nd.id.as_str(), i))
        .collect();

    // Directed edges from wires (deduplicated, self-loops dropped).
    let mut edge_set: HashSet<(usize, usize)> = HashSet::new();
    for w in &view.patch.wires {
        if let (Some(&a), Some(&b)) = (index.get(w.from.node()), index.get(w.to.node())) {
            if a != b {
                edge_set.insert((a, b));
            }
        }
    }
    let edges: Vec<(usize, usize)> = edge_set.iter().copied().collect();

    // Make a DAG: drop back edges (deterministic DFS).
    let back = back_edges(n, &edges);
    let fwd: Vec<(usize, usize)> = edges.iter().copied().filter(|e| !back.contains(e)).collect();

    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut preds: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in &fwd {
        succ[u].push(v);
        preds[v].push(u);
    }

    // Longest-path columns over the DAG.
    let order = topo(n, &fwd);
    let mut layer = vec![0usize; n];
    for &u in &order {
        for &v in &succ[u] {
            if layer[v] < layer[u] + 1 {
                layer[v] = layer[u] + 1;
            }
        }
    }
    // Anchor every audio_output sink to the rightmost column.
    let max_layer = layer.iter().copied().max().unwrap_or(0);
    for (i, nd) in view.patch.nodes.iter().enumerate() {
        if nd.ty == "audio_output" {
            layer[i] = max_layer;
        }
    }
    let max_layer = layer.iter().copied().max().unwrap_or(0);

    // Group nodes by column (initial order = node index, deterministic).
    let mut columns: Vec<Vec<usize>> = vec![Vec::new(); max_layer + 1];
    for i in 0..n {
        columns[layer[i]].push(i);
    }
    let mut posidx = vec![0usize; n];
    for col in &columns {
        for (k, &node) in col.iter().enumerate() {
            posidx[node] = k;
        }
    }

    // Crossing reduction: alternating barycenter sweeps.
    for sweep in 0..SWEEPS {
        if sweep % 2 == 0 {
            for l in 1..=max_layer {
                barycenter_sort(&mut columns[l], &preds, &posidx);
                for (k, &node) in columns[l].iter().enumerate() {
                    posidx[node] = k;
                }
            }
        } else {
            for l in (0..max_layer).rev() {
                barycenter_sort(&mut columns[l], &succ, &posidx);
                for (k, &node) in columns[l].iter().enumerate() {
                    posidx[node] = k;
                }
            }
        }
    }

    // Column x's: cumulative, each column as wide as its widest node + the gap.
    let mut col_x = vec![0.0f32; max_layer + 1];
    let mut x = 0.0f32;
    for l in 0..=max_layer {
        let w = columns[l]
            .iter()
            .map(|&i| sizes[i][0])
            .fold(0.0f32, f32::max)
            .max(1.0);
        col_x[l] = x + w * 0.5;
        x += w + COL_GAP_MM;
    }

    // Node y's: stack each column top-to-bottom, centered on 0.
    let mut centers = vec![[0.0f32; 2]; n];
    for l in 0..=max_layer {
        let col = &columns[l];
        let total_h: f32 = col.iter().map(|&i| sizes[i][1]).sum::<f32>()
            + ROW_GAP_MM * col.len().saturating_sub(1) as f32;
        let mut y = -total_h * 0.5;
        for &i in col {
            let h = sizes[i][1];
            centers[i] = [col_x[l], y + h * 0.5];
            y += h + ROW_GAP_MM;
        }
    }

    // Center the whole layout on the origin (layout is relative to canvas center).
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (f32::MAX, f32::MIN, f32::MAX, f32::MIN);
    for c in &centers {
        min_x = min_x.min(c[0]);
        max_x = max_x.max(c[0]);
        min_y = min_y.min(c[1]);
        max_y = max_y.max(c[1]);
    }
    let (mid_x, mid_y) = ((min_x + max_x) * 0.5, (min_y + max_y) * 0.5);

    let ids: Vec<String> = view.patch.nodes.iter().map(|nd| nd.id.clone()).collect();
    view.patch.layout.clear();
    for (i, id) in ids.into_iter().enumerate() {
        view.patch
            .layout
            .insert(id, [(centers[i][0] - mid_x) as f64, (centers[i][1] - mid_y) as f64]);
    }
}

/// Order a column by the average position of each node's neighbors (stable; nodes with no
/// neighbors keep their current order via their own position as the key).
fn barycenter_sort(col: &mut [usize], neighbors: &[Vec<usize>], posidx: &[usize]) {
    let key = |node: usize| -> f32 {
        let ns = &neighbors[node];
        if ns.is_empty() {
            posidx[node] as f32
        } else {
            ns.iter().map(|&m| posidx[m] as f32).sum::<f32>() / ns.len() as f32
        }
    };
    col.sort_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap_or(std::cmp::Ordering::Equal));
}

/// Kahn topological order over a DAG; sources emitted in index order for determinism.
fn topo(n: usize, edges: &[(usize, usize)]) -> Vec<usize> {
    let mut indeg = vec![0usize; n];
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        succ[u].push(v);
        indeg[v] += 1;
    }
    let mut queue: Vec<usize> = (0..n).filter(|&i| indeg[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    let mut head = 0;
    while head < queue.len() {
        let u = queue[head];
        head += 1;
        order.push(u);
        for &v in &succ[u] {
            indeg[v] -= 1;
            if indeg[v] == 0 {
                queue.push(v);
            }
        }
    }
    order
}

/// Edges whose target is an ancestor on the DFS stack — the cycle-breaking back edges.
fn back_edges(n: usize, edges: &[(usize, usize)]) -> HashSet<(usize, usize)> {
    let mut succ: Vec<Vec<usize>> = vec![Vec::new(); n];
    for &(u, v) in edges {
        succ[u].push(v);
    }
    let mut color = vec![0u8; n]; // 0 = unvisited, 1 = on stack, 2 = done
    let mut back = HashSet::new();
    for s in 0..n {
        if color[s] != 0 {
            continue;
        }
        color[s] = 1;
        let mut stack: Vec<(usize, usize)> = vec![(s, 0)];
        while let Some(&mut (node, ref mut ci)) = stack.last_mut() {
            if *ci < succ[node].len() {
                let v = succ[node][*ci];
                *ci += 1;
                match color[v] {
                    0 => {
                        color[v] = 1;
                        stack.push((v, 0));
                    }
                    1 => {
                        back.insert((node, v));
                    }
                    _ => {}
                }
            } else {
                color[node] = 2;
                stack.pop();
            }
        }
    }
    back
}

#[cfg(test)]
mod tests {
    use super::*;
    use synth_core::model::Patch;

    const CHAIN: &str = r#"
nodes:
  - id: freq
    type: const_generator
  - id: osc
    type: sine_generator
  - id: out
    type: audio_output
    params: { channels: 1 }
wires:
  - { from: [freq, out], to: [osc, frequency] }
  - { from: [osc, out], to: [out, ch0] }
"#;

    #[test]
    fn lays_out_chain_left_to_right() {
        let mut view = GraphView::new(Patch::from_yaml(CHAIN).unwrap());
        autolayout_full(&mut view);
        let x = |id: &str| view.patch.layout.get(id).unwrap()[0];
        // freq → osc → out, so x strictly increases along the signal path.
        assert!(x("freq") < x("osc"));
        assert!(x("osc") < x("out"));
    }

    #[test]
    fn every_node_positioned() {
        let mut view = GraphView::new(Patch::from_yaml(CHAIN).unwrap());
        autolayout_full(&mut view);
        assert_eq!(view.patch.layout.len(), view.patch.nodes.len());
    }

    #[test]
    fn terminates_on_cycle() {
        // A feedback loop must not hang (back edges are dropped for layering).
        let yaml = r#"
nodes:
  - id: a
    type: mul
  - id: b
    type: mul
  - id: out
    type: audio_output
    params: { channels: 1 }
wires:
  - { from: [a, out], to: [b, a] }
  - { from: [b, out], to: [a, a] }
  - { from: [b, out], to: [out, ch0] }
"#;
        let mut view = GraphView::new(Patch::from_yaml(yaml).unwrap());
        autolayout_full(&mut view);
        assert_eq!(view.patch.layout.len(), 3);
    }
}
