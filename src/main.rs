//! synth-ui — visual patch editor (MVP).
//!
//! Loads a patch YAML (a path argument, or an embedded demo) and opens a wgpu canvas to view and
//! edit it. See `app::App` for the controls. Autolayout is not wired up yet, so a loaded patch
//! is expected to carry a `layout` block; nodes without one fall back to a staggered position.

mod app;
mod audio;
mod camera;
mod graph;
mod render;

use winit::event_loop::EventLoop;

use synth_core::model::Patch;

use crate::app::App;
use crate::graph::GraphView;

/// A small audible demo patch (220 Hz tone) plus a couple of unwired modules to show ports.
const DEMO: &str = r#"
version: 1
nodes:
  - id: freq
    type: const_generator
    params: { value: 220.0 }
  - id: amp
    type: const_generator
    params: { value: 0.2 }
  - id: osc
    type: sine_generator
  - id: out
    type: audio_output
    params: { device: default, channels: 2 }
  - id: env
    type: adsr_envelope
  - id: vca
    type: mul
wires:
  - { from: [freq, out], to: [osc, frequency] }
  - { from: [amp,  out], to: [osc, amplitude] }
  - { from: [osc, out], to: [out, ch0] }
  - { from: [osc, out], to: [out, ch1] }
layout:
  freq: [-320, -140]
  amp:  [-320, 60]
  osc:  [-60, -60]
  out:  [260, -60]
  env:  [-60, 200]
  vca:  [200, 200]
"#;

fn main() {
    let patch = match std::env::args().nth(1) {
        Some(path) => {
            let yaml = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("failed to read patch '{path}': {e}"));
            Patch::from_yaml(&yaml).unwrap_or_else(|e| panic!("failed to parse patch: {e}"))
        }
        None => Patch::from_yaml(DEMO).expect("embedded demo patch parses"),
    };

    let event_loop = EventLoop::new().expect("create event loop");
    let mut app = App::new(GraphView::new(patch));
    event_loop.run_app(&mut app).expect("run app");
}
