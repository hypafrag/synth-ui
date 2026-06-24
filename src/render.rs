//! Custom wgpu canvas renderer.
//!
//! Draws the graph as flat geometry: node bodies/headers/ports as triangles and wires as lines,
//! all in world coordinates transformed to the screen by a camera uniform in the vertex shader
//! (see `docs/architecture/12-ui-rendering.md`). No text yet — module names surface in the
//! window title.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

use synth_core::module::SignalKind;

use crate::camera::Camera;
use crate::graph::{HEADER_H_MM, NodeGeom, PORT_R_MM};

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pos: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniform {
    viewport: [f32; 2],
    pan: [f32; 2],
    zoom: f32,
    _pad: [f32; 3],
}

// Palette.
const BG: wgpu::Color = wgpu::Color {
    r: 0.10,
    g: 0.11,
    b: 0.13,
    a: 1.0,
};
const NODE_BODY: [f32; 4] = [0.18, 0.19, 0.22, 1.0];
const NODE_HEADER: [f32; 4] = [0.28, 0.30, 0.37, 1.0];
const NODE_HEADER_HOVER: [f32; 4] = [0.34, 0.52, 0.74, 1.0];
const NODE_UNKNOWN: [f32; 4] = [0.55, 0.24, 0.26, 1.0];
const PORT_SAMPLE: [f32; 4] = [0.32, 0.78, 0.66, 1.0];
const PORT_EVENT: [f32; 4] = [0.92, 0.58, 0.22, 1.0];
const WIRE: [f32; 4] = [0.74, 0.78, 0.83, 1.0];
const WIRE_PENDING: [f32; 4] = [0.96, 0.86, 0.28, 1.0];

// Screen-space toolbar, sized in millimeters (best-effort physical units). Callers pass `scale`
// = physical pixels per mm (derived from the window scale factor; see `app::App::ui_scale`), and
// every dimension below is multiplied by it. Top-left origin.
const TOOLBAR_H_MM: f32 = 12.0;
const BTN_MM: f32 = 8.5;
const MARGIN_MM: f32 = 1.75;
const BTN_GAP_MM: f32 = 2.0;
const TOOLBAR_BG: [f32; 4] = [0.13, 0.14, 0.17, 1.0];
const BTN_BG: [f32; 4] = [0.22, 0.24, 0.29, 1.0];
const BTN_BG_HOVER: [f32; 4] = [0.31, 0.34, 0.41, 1.0];
const ICON_PLAY: [f32; 4] = [0.46, 0.86, 0.52, 1.0];
const ICON_PAUSE: [f32; 4] = [0.96, 0.82, 0.36, 1.0];
const ICON_ARRANGE: [f32; 4] = [0.62, 0.74, 0.92, 1.0];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ToolBtn {
    Play,
    Arrange,
}

/// The toolbar bar height in physical pixels.
pub fn toolbar_height(scale: f32) -> f32 {
    TOOLBAR_H_MM * scale
}

/// A toolbar button's rect `(x, y, w, h)` in physical pixels.
pub fn button_rect(which: ToolBtn, scale: f32) -> (f32, f32, f32, f32) {
    let b = BTN_MM * scale;
    let y = (TOOLBAR_H_MM - BTN_MM) * 0.5 * scale;
    let slot = match which {
        ToolBtn::Play => 0.0,
        ToolBtn::Arrange => 1.0,
    };
    let x = (MARGIN_MM + slot * (BTN_MM + BTN_GAP_MM)) * scale;
    (x, y, b, b)
}

/// The toolbar button under `screen` (physical pixels), if any.
pub fn hit_button(screen: [f32; 2], scale: f32) -> Option<ToolBtn> {
    for which in [ToolBtn::Play, ToolBtn::Arrange] {
        let (x, y, w, h) = button_rect(which, scale);
        if screen[0] >= x && screen[0] <= x + w && screen[1] >= y && screen[1] <= y + h {
            return Some(which);
        }
    }
    None
}

fn port_color(kind: SignalKind) -> [f32; 4] {
    match kind {
        SignalKind::Sample => PORT_SAMPLE,
        SignalKind::Event => PORT_EVENT,
    }
}

fn push_rect(v: &mut Vec<Vertex>, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
    let (x0, y0, x1, y1) = (x, y, x + w, y + h);
    let p = |px, py| Vertex {
        pos: [px, py],
        color,
    };
    v.extend_from_slice(&[
        p(x0, y0),
        p(x1, y0),
        p(x1, y1),
        p(x0, y0),
        p(x1, y1),
        p(x0, y1),
    ]);
}

fn push_line(v: &mut Vec<Vertex>, a: [f32; 2], b: [f32; 2], color: [f32; 4]) {
    v.push(Vertex { pos: a, color });
    v.push(Vertex { pos: b, color });
}

fn push_tri(v: &mut Vec<Vertex>, a: [f32; 2], b: [f32; 2], c: [f32; 2], color: [f32; 4]) {
    v.push(Vertex { pos: a, color });
    v.push(Vertex { pos: b, color });
    v.push(Vertex { pos: c, color });
}

/// Build the toolbar triangles in physical-pixel coordinates (drawn with the screen-space
/// uniform). `scale` is physical pixels per mm. Shows a play glyph when stopped, pause when
/// playing.
pub fn build_toolbar(
    viewport: [f32; 2],
    playing: bool,
    scale: f32,
    hovered: Option<ToolBtn>,
) -> Vec<Vertex> {
    let mut v = Vec::new();
    let s = scale;
    push_rect(&mut v, 0.0, 0.0, viewport[0], TOOLBAR_H_MM * s, TOOLBAR_BG);

    // Play / pause: triangle when stopped, two bars when playing.
    button_bg(&mut v, ToolBtn::Play, s, hovered == Some(ToolBtn::Play));
    let (bx, by, bw, bh) = button_rect(ToolBtn::Play, s);
    let (cx, cy) = (bx + bw * 0.5, by + bh * 0.5);
    if playing {
        let (w, h, gap) = (1.5 * s, 5.0 * s, 1.4 * s);
        push_rect(&mut v, cx - gap * 0.5 - w, cy - h * 0.5, w, h, ICON_PAUSE);
        push_rect(&mut v, cx + gap * 0.5, cy - h * 0.5, w, h, ICON_PAUSE);
    } else {
        let (l, r, hh) = (2.0 * s, 3.0 * s, 2.7 * s);
        push_tri(&mut v, [cx - l, cy - hh], [cx - l, cy + hh], [cx + r, cy], ICON_PLAY);
    }

    // Arrange: three vertical bars of varying height (a "lay out" glyph).
    button_bg(&mut v, ToolBtn::Arrange, s, hovered == Some(ToolBtn::Arrange));
    let (ax, ay, aw, ah) = button_rect(ToolBtn::Arrange, s);
    let (acx, acy) = (ax + aw * 0.5, ay + ah * 0.5);
    let (bw2, gap2) = (1.2 * s, 1.0 * s);
    let heights = [3.0 * s, 5.2 * s, 4.0 * s];
    let mut bx2 = acx - (bw2 * 3.0 + gap2 * 2.0) * 0.5;
    for h in heights {
        push_rect(&mut v, bx2, acy - h * 0.5, bw2, h, ICON_ARRANGE);
        bx2 += bw2 + gap2;
    }
    v
}

fn button_bg(v: &mut Vec<Vertex>, which: ToolBtn, scale: f32, hovered: bool) {
    let (x, y, w, h) = button_rect(which, scale);
    push_rect(v, x, y, w, h, if hovered { BTN_BG_HOVER } else { BTN_BG });
}

/// Build triangle and line vertices for the current scene.
pub fn build_scene(
    geoms: &[NodeGeom],
    wires: &[([f32; 2], [f32; 2], SignalKind)],
    pending: Option<([f32; 2], [f32; 2])>,
    hover: Option<&str>,
) -> (Vec<Vertex>, Vec<Vertex>) {
    let mut tris = Vec::new();
    let mut lines = Vec::new();

    for g in geoms {
        let r = g.rect;
        push_rect(&mut tris, r.x, r.y, r.w, r.h, NODE_BODY);
        let header = if !g.known {
            NODE_UNKNOWN
        } else if hover == Some(g.id.as_str()) {
            NODE_HEADER_HOVER
        } else {
            NODE_HEADER
        };
        push_rect(&mut tris, r.x, r.y, r.w, HEADER_H_MM, header);
        for p in g.inputs.iter().chain(g.outputs.iter()) {
            push_rect(
                &mut tris,
                p.pos[0] - PORT_R_MM,
                p.pos[1] - PORT_R_MM,
                PORT_R_MM * 2.0,
                PORT_R_MM * 2.0,
                port_color(p.kind),
            );
        }
    }

    for (a, b, kind) in wires {
        push_line(&mut lines, *a, *b, port_color(*kind));
    }
    let _ = WIRE; // reserved neutral wire color
    if let Some((a, b)) = pending {
        push_line(&mut lines, a, b, WIRE_PENDING);
    }

    (tris, lines)
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    tri_pipeline: wgpu::RenderPipeline,
    line_pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    ui_uniform_buf: wgpu::Buffer,
    ui_bind_group: wgpu::BindGroup,
    max_dim: u32,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let width = size.width.max(1);
        let height = size.height.max(1);

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance.create_surface(window).expect("create surface");
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::default(),
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }))
        .expect("no GPU adapter");
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("synth-ui device"),
                required_features: wgpu::Features::empty(),
                // The adapter's real limits — `downlevel_defaults` caps textures at 2048, too small
                // for a retina window. On Apple Silicon this is 16384.
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
            },
            None,
        ))
        .expect("request device");

        // Clamp the surface to the device's max texture size so configure never overflows.
        let max_dim = device.limits().max_texture_dimension_2d;
        let width = width.min(max_dim);
        let height = height.min(max_dim);

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("canvas shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera uniform"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("camera bind layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera bind group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        // Second uniform/bind group for screen-space UI (pan = viewport/2, zoom = 1, so vertices
        // are interpreted as raw top-left pixel coordinates by the shared shader).
        let ui_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ui uniform"),
            size: std::mem::size_of::<Uniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ui_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ui bind group"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: ui_uniform_buf.as_entire_binding(),
            }],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("canvas pipeline layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
            ],
        };

        let make_pipeline = |topology: wgpu::PrimitiveTopology| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("canvas pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vs",
                    buffers: &[vertex_layout.clone()],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "fs",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: config.format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let tri_pipeline = make_pipeline(wgpu::PrimitiveTopology::TriangleList);
        let line_pipeline = make_pipeline(wgpu::PrimitiveTopology::LineList);

        Self {
            surface,
            device,
            queue,
            config,
            tri_pipeline,
            line_pipeline,
            uniform_buf,
            bind_group,
            ui_uniform_buf,
            ui_bind_group,
            max_dim,
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.min(self.max_dim);
        let height = height.min(self.max_dim);
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn viewport(&self) -> [f32; 2] {
        [self.config.width as f32, self.config.height as f32]
    }

    pub fn render(&mut self, camera: &Camera, tris: &[Vertex], lines: &[Vertex], ui_tris: &[Vertex]) {
        let viewport = self.viewport();
        let uniform = Uniform {
            viewport,
            pan: camera.pan,
            zoom: camera.px_per_mm(),
            _pad: [0.0; 3],
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));
        let ui_uniform = Uniform {
            viewport,
            pan: [viewport[0] * 0.5, viewport[1] * 0.5],
            zoom: 1.0,
            _pad: [0.0; 3],
        };
        self.queue
            .write_buffer(&self.ui_uniform_buf, 0, bytemuck::bytes_of(&ui_uniform));

        let tri_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tri verts"),
            contents: bytemuck::cast_slice(tris),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let line_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("line verts"),
            contents: bytemuck::cast_slice(lines),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let ui_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ui verts"),
            contents: bytemuck::cast_slice(ui_tris),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let frame = match self.surface.get_current_texture() {
            Ok(f) => f,
            Err(_) => {
                self.surface.configure(&self.device, &self.config);
                match self.surface.get_current_texture() {
                    Ok(f) => f,
                    Err(_) => return,
                }
            }
        };
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("canvas pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(BG),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            if !tris.is_empty() {
                pass.set_pipeline(&self.tri_pipeline);
                pass.set_vertex_buffer(0, tri_buf.slice(..));
                pass.draw(0..tris.len() as u32, 0..1);
            }
            if !lines.is_empty() {
                pass.set_pipeline(&self.line_pipeline);
                pass.set_vertex_buffer(0, line_buf.slice(..));
                pass.draw(0..lines.len() as u32, 0..1);
            }
            // Screen-space UI (toolbar) on top, using the UI bind group.
            if !ui_tris.is_empty() {
                pass.set_pipeline(&self.tri_pipeline);
                pass.set_bind_group(0, &self.ui_bind_group, &[]);
                pass.set_vertex_buffer(0, ui_buf.slice(..));
                pass.draw(0..ui_tris.len() as u32, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        frame.present();
    }
}

const SHADER: &str = r#"
// Scalar padding (not vec3) so the struct stays 32 bytes, matching the Rust `Uniform`.
// A `vec3<f32>` would force 16-byte alignment and a 48-byte struct.
struct U {
    viewport: vec2<f32>,
    pan: vec2<f32>,
    zoom: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> u: U;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) c: vec4<f32>) -> VOut {
    let screen = (p - u.pan) * u.zoom + u.viewport * 0.5;
    let ndc = vec2<f32>(screen.x / u.viewport.x * 2.0 - 1.0, 1.0 - screen.y / u.viewport.y * 2.0);
    var o: VOut;
    o.pos = vec4<f32>(ndc, 0.0, 1.0);
    o.color = c;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return i.color;
}
"#;
