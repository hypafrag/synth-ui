//! Custom wgpu canvas renderer + egui chrome compositing.
//!
//! The frame is two sequential passes into one surface view, one command encoder, one submit
//! (see `docs/architecture/12-ui-rendering.md`):
//!   pass 1 — canvas: clear, scissored to the central canvas rect; instanced node geometry,
//!            icons, and wires in world coordinates via a camera uniform.
//!   pass 2 — egui:   load (no clear), full surface; menu/toolbar/status bars + dockable panels.
//!
//! The canvas no longer owns the whole window: egui docks panels around a central region, so the
//! camera uniform carries that region's rect and the canvas pass is scissored to it.

use std::collections::HashMap;
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use winit::window::Window;

use synth_core::module::{Icon, SignalKind};

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
pub struct IconVertex {
    pos: [f32; 2],
    uv: [f32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Uniform {
    /// Full surface size in physical px (used to normalize to NDC).
    surface: [f32; 2],
    /// Canvas rect top-left in physical px, within the surface.
    canvas_origin: [f32; 2],
    /// Canvas rect size in physical px.
    canvas_size: [f32; 2],
    /// World point (mm) shown at the canvas-rect center.
    pan: [f32; 2],
    /// On-screen physical px per world mm.
    zoom: f32,
    _pad: [f32; 3],
}

/// The tessellated egui chrome for one frame, handed to [`Renderer::render`].
pub struct EguiFrame {
    pub textures_delta: egui::TexturesDelta,
    pub paint_jobs: Vec<egui::epaint::ClippedPrimitive>,
    pub pixels_per_point: f32,
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

// Icon quad size and inset from the node's top-left corner, in mm.
const ICON_MM: f32 = 4.2;
const ICON_INSET_MM: f32 = 1.0;

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

/// Icon quads (one per node) in the node's top-left corner, with UVs into the icon atlas band.
pub fn build_icons(
    geoms: &[NodeGeom],
    index: &HashMap<String, usize>,
    count: usize,
) -> Vec<IconVertex> {
    let mut v = Vec::new();
    if count == 0 {
        return v;
    }
    let inv = 1.0 / count as f32;
    for g in geoms {
        let idx = index.get(&g.ty).copied().unwrap_or(0);
        let (v0, v1) = (idx as f32 * inv, (idx + 1) as f32 * inv);
        let (x0, y0) = (g.rect.x + ICON_INSET_MM, g.rect.y + ICON_INSET_MM);
        let (x1, y1) = (x0 + ICON_MM, y0 + ICON_MM);
        let p = |x, y, u, w| IconVertex {
            pos: [x, y],
            uv: [u, w],
        };
        v.extend_from_slice(&[
            p(x0, y0, 0.0, v0),
            p(x1, y0, 1.0, v0),
            p(x1, y1, 1.0, v1),
            p(x0, y0, 0.0, v0),
            p(x1, y1, 1.0, v1),
            p(x0, y1, 0.0, v1),
        ]);
    }
    v
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
    icon_pipeline: wgpu::RenderPipeline,
    icon_tex_layout: wgpu::BindGroupLayout,
    icon_sampler: wgpu::Sampler,
    icon_atlas: Option<(wgpu::BindGroup, usize)>,
    egui_renderer: egui_wgpu::Renderer,
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

        // Icon pass: a textured quad per node sampling the icon atlas with a nearest filter.
        let icon_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("icon shader"),
            source: wgpu::ShaderSource::Wgsl(ICON_SHADER.into()),
        });
        let icon_tex_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("icon texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let icon_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("icon sampler (nearest)"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });
        let icon_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("icon pipeline layout"),
                bind_group_layouts: &[&bind_layout, &icon_tex_layout],
                push_constant_ranges: &[],
            });
        let icon_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("icon pipeline"),
            layout: Some(&icon_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &icon_shader,
                entry_point: "vs",
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<IconVertex>() as u64,
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
                            format: wgpu::VertexFormat::Float32x2,
                        },
                    ],
                }],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &icon_shader,
                entry_point: "fs",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // egui composites onto the surface format, no depth, no MSAA, no dithering.
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, None, 1, false);

        Self {
            surface,
            device,
            queue,
            config,
            tri_pipeline,
            line_pipeline,
            uniform_buf,
            bind_group,
            icon_pipeline,
            icon_tex_layout,
            icon_sampler,
            icon_atlas: None,
            egui_renderer,
            max_dim,
        }
    }

    /// Upload the module icons into the atlas texture (one 32×32 band per icon, stacked
    /// vertically) and build the icon bind group. Call once after construction and again whenever
    /// the set of node types changes (e.g. a new module is added from the palette).
    pub fn set_icons(&mut self, icons: &[Icon]) {
        let count = icons.len();
        if count == 0 {
            self.icon_atlas = None;
            return;
        }
        let (w, h) = (32u32, 32 * count as u32);
        // write_texture requires bytes_per_row to be a multiple of 256; pad each row.
        let stride = 256usize;
        let mut data = vec![0u8; stride * h as usize];
        for (i, icon) in icons.iter().enumerate() {
            for (row, &bits) in icon.iter().enumerate() {
                let y = i * 32 + row;
                for x in 0..32u32 {
                    if (bits >> (31 - x)) & 1 == 1 {
                        data[y * stride + x as usize] = 255;
                    }
                }
            }
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("icon atlas"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(stride as u32),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("icon atlas bind"),
            layout: &self.icon_tex_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.icon_sampler),
                },
            ],
        });
        self.icon_atlas = Some((bind, count));
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

    /// Render one frame: the canvas (scissored to `canvas_rect`, physical px `[x, y, w, h]`) then
    /// the egui chrome on top.
    pub fn render(
        &mut self,
        camera: &Camera,
        canvas_rect: [f32; 4],
        tris: &[Vertex],
        lines: &[Vertex],
        icon_verts: &[IconVertex],
        egui: EguiFrame,
    ) {
        let surface = self.viewport();
        let uniform = Uniform {
            surface,
            canvas_origin: [canvas_rect[0], canvas_rect[1]],
            canvas_size: [canvas_rect[2], canvas_rect[3]],
            pan: camera.pan,
            zoom: camera.px_per_mm(),
            _pad: [0.0; 3],
        };
        self.queue
            .write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&uniform));

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
        let icon_buf = (!icon_verts.is_empty()).then(|| {
            self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("icon verts"),
                contents: bytemuck::cast_slice(icon_verts),
                usage: wgpu::BufferUsages::VERTEX,
            })
        });

        // Scissor the canvas pass to the central rect, clamped to the surface. An empty rect skips
        // the canvas geometry entirely (panels cover the whole window).
        let (sw, sh) = (self.config.width, self.config.height);
        let sx = (canvas_rect[0].max(0.0) as u32).min(sw);
        let sy = (canvas_rect[1].max(0.0) as u32).min(sh);
        let sgw = (canvas_rect[2].max(0.0) as u32).min(sw - sx);
        let sgh = (canvas_rect[3].max(0.0) as u32).min(sh - sy);
        let canvas_visible = sgw > 0 && sgh > 0;

        let screen_desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [sw, sh],
            pixels_per_point: egui.pixels_per_point,
        };

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

        // egui buffer/texture uploads happen on the same encoder, before its pass.
        for (id, delta) in &egui.textures_delta.set {
            self.egui_renderer
                .update_texture(&self.device, &self.queue, *id, delta);
        }
        let egui_cmds = self.egui_renderer.update_buffers(
            &self.device,
            &self.queue,
            &mut encoder,
            &egui.paint_jobs,
            &screen_desc,
        );

        // Pass 1 — canvas (clear), scissored to the central rect.
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
            if canvas_visible {
                pass.set_scissor_rect(sx, sy, sgw, sgh);
                pass.set_bind_group(0, &self.bind_group, &[]);
                if !tris.is_empty() {
                    pass.set_pipeline(&self.tri_pipeline);
                    pass.set_vertex_buffer(0, tri_buf.slice(..));
                    pass.draw(0..tris.len() as u32, 0..1);
                }
                // Module icons, on top of node bodies (camera-space, nearest-sampled atlas).
                if let (Some((bind, _)), Some(buf)) = (&self.icon_atlas, &icon_buf) {
                    pass.set_pipeline(&self.icon_pipeline);
                    pass.set_bind_group(0, &self.bind_group, &[]);
                    pass.set_bind_group(1, bind, &[]);
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..icon_verts.len() as u32, 0..1);
                }
                if !lines.is_empty() {
                    pass.set_pipeline(&self.line_pipeline);
                    pass.set_vertex_buffer(0, line_buf.slice(..));
                    pass.draw(0..lines.len() as u32, 0..1);
                }
            }
        }

        // Pass 2 — egui chrome (load, full surface) over the canvas.
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                })
                .forget_lifetime();
            self.egui_renderer
                .render(&mut pass, &egui.paint_jobs, &screen_desc);
        }

        self.queue
            .submit(egui_cmds.into_iter().chain(std::iter::once(encoder.finish())));
        frame.present();

        for id in &egui.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
    }
}

const SHADER: &str = r#"
struct U {
    surface: vec2<f32>,
    canvas_origin: vec2<f32>,
    canvas_size: vec2<f32>,
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

fn world_to_ndc(p: vec2<f32>) -> vec2<f32> {
    let screen = u.canvas_origin + u.canvas_size * 0.5 + (p - u.pan) * u.zoom;
    return vec2<f32>(screen.x / u.surface.x * 2.0 - 1.0, 1.0 - screen.y / u.surface.y * 2.0);
}

@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) c: vec4<f32>) -> VOut {
    var o: VOut;
    o.pos = vec4<f32>(world_to_ndc(p), 0.0, 1.0);
    o.color = c;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    return i.color;
}
"#;

const ICON_SHADER: &str = r#"
struct U {
    surface: vec2<f32>,
    canvas_origin: vec2<f32>,
    canvas_size: vec2<f32>,
    pan: vec2<f32>,
    zoom: f32,
    pad0: f32,
    pad1: f32,
    pad2: f32,
};
@group(0) @binding(0) var<uniform> u: U;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var samp: sampler;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@location(0) p: vec2<f32>, @location(1) uv: vec2<f32>) -> VOut {
    let screen = u.canvas_origin + u.canvas_size * 0.5 + (p - u.pan) * u.zoom;
    let ndc = vec2<f32>(screen.x / u.surface.x * 2.0 - 1.0, 1.0 - screen.y / u.surface.y * 2.0);
    var o: VOut;
    o.pos = vec4<f32>(ndc, 0.0, 1.0);
    o.uv = uv;
    return o;
}

@fragment
fn fs(i: VOut) -> @location(0) vec4<f32> {
    let a = textureSample(tex, samp, i.uv).r;
    return vec4<f32>(0.86, 0.89, 0.94, a);
}
"#;
