//! Start/stop wrapper around the core engine + cpal output.
//!
//! Holds the live `cpal::Stream` as an opaque `Box<dyn Any>` so the UI needs no direct cpal
//! dependency — dropping it stops audio. Rebuilding the engine on each (re)start applies any
//! topology edits made since.

use std::any::Any;

use synth_core::audio::run_default_output;
use synth_core::model::Patch;
use synth_core::module::Registry;
use synth_core::plan_engine::PlanEngine;

/// Maximum audio block size (frames) the engine pre-allocates for.
const MAX_FRAMES: usize = 16384;

#[derive(Default)]
pub struct Audio {
    stream: Option<Box<dyn Any>>,
    pub playing: bool,
    pub status: String,
}

impl Audio {
    pub fn toggle(&mut self, patch: &Patch, registry: &Registry) {
        if self.playing {
            self.stop();
        } else {
            self.start(patch, registry);
        }
    }

    pub fn start(&mut self, patch: &Patch, registry: &Registry) {
        self.stream = None; // release any prior device stream before opening a new one
        match PlanEngine::build(patch, registry, MAX_FRAMES) {
            Ok(engine) => match run_default_output(engine) {
                Ok(stream) => {
                    self.stream = Some(Box::new(stream));
                    self.playing = true;
                    self.status = "playing".to_string();
                }
                Err(e) => {
                    self.playing = false;
                    self.status = format!("audio error: {e}");
                    eprintln!("{}", self.status);
                }
            },
            Err(e) => {
                self.playing = false;
                self.status = format!("build error: {e}");
                eprintln!("{}", self.status);
            }
        }
    }

    pub fn stop(&mut self) {
        self.stream = None;
        self.playing = false;
        self.status = "stopped".to_string();
    }

    /// Re-apply the current patch to the running stream (called after a topology edit).
    pub fn rebuild_if_playing(&mut self, patch: &Patch, registry: &Registry) {
        if self.playing {
            self.start(patch, registry);
        }
    }
}
