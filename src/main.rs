/*************************

# Quick summary of the initialization that’s required to do that. We will need:

1. A window to display our rendered image.
2. A backend instance to access the graphics API. This gives us access to:
  - A surface on which to render, and then present to the window.
  - An adapter which represents a physical device (like a graphics card).
3. One or more queue groups, which give us access to command queues.
4. A device, which is a logical device we obtain by configuring the adapter.
This will be used to create most of the rest of our resources, including:
  - A command pool for allocating command buffers to send instructions
  to the command queues.
  - A render pass which defines how different images are used.
  (For example, which to render to, which is a depth buffer, etc.)
  - A graphics pipeline which contains our shaders and specifies
  how exactly to render each triangle.
  - And finally, a fence and a semaphore for synchronizing our program.

*****************************/

use std::mem::ManuallyDrop;

use gfx_hal::{
    device::Device,
    window::{Extent2D, PresentationSurface, Surface},
    Instance,
};
use shaderc::ShaderKind;

// We defined the PushConstants struct in our shader where we were just asking it to interpret
// some of those bytes as a struct. We do the same in our Rust code to make it easier to send that data.

// The repr(C) attribute which tells the compiler to lay out this struct in memory the way C would.
// This is also (close to) how structs are laid out in shader code.
// By ensuring the layouts are the same, we can easily copy the Rust struct straight into push
// constants without worrying about individual fields.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct PushConstants {
    transform: [[f32; 4]; 4],
}

// In the repo is a mesh of the Utah teapot, serialized with the bincode crate.
// Really it’s just a Vec of vertices efficiently packed into a binary file.
// To deserialize it, we’ll need to define a compatible Vertex struct:
#[derive(serde::Deserialize)]
#[repr(C)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

fn main() {
    const APP_NAME: &'static str = "Part 3: Vertex buffers";
    const WINDOW_SIZE: [u32; 2] = [512, 512];

    let event_loop = winit::event_loop::EventLoop::new();

    // "High-DPI displays, to avoid having unusably small UI elements,
    // pretend to have a smaller size than they actually do. For example,
    // a screen 2048 physical pixels wide may report a logical size of 1024,
    // along with a scale factor of 2. This means that a 1024 pixel window
    // will fill the whole screen, because the OS will scale it up by 2 under
    // the hood to cover all 2048 pixels. It also means that on my other, more
    // ancient 1024 pixel monitor with a scale factor of just 1, the window
    // will appear to be the same size, without me having to configure the window differently."

    // So physical size represents real life pixels, and varies a lot across different devices,
    // while logical size is an abstraction representing a smaller size which is more consistent between devices.
    let (logical_window_size, physical_window_size) = {
        use winit::dpi::{LogicalSize, PhysicalSize};

        let scale_factor = match event_loop.primary_monitor() {
            Some(m) => m.scale_factor(),
            None => 2.0,
        };
        let dpi = scale_factor;
        let logical: LogicalSize<u32> = WINDOW_SIZE.into();
        let physical: PhysicalSize<u32> = logical.to_physical(dpi);

        (logical, physical)
    };

    let mut surface_extent = Extent2D {
        width: physical_window_size.width,
        height: physical_window_size.height,
    };

    let window = winit::window::WindowBuilder::new()
        .with_title(APP_NAME)
        .with_inner_size(logical_window_size)
        .build(&event_loop)
        .expect("Failed to create window");

    // Our very first call to gfx will be to create an Instance which serves
    // as an entrypoint to the backend graphics API. We use this only to acquire
    // a surface to draw on, and an adapter which represents a physical
    // graphics device (e.g. a graphics card):
    let (instance, surface, adapter) = {
        let instance = gfx_backend::Instance::create(APP_NAME, 1).expect("Backend not supported");

        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create surface for window")
        };

        let adapter = instance.enumerate_adapters().remove(0);

        (Some(instance), surface, adapter)
    };

    // Next we want to acquire a logical device which will allow us to create the rest
    // of our resources. You can think of a logical device as a particular
    // configuration of a physical device - with or without certain features enabled.

    // We also want a queue_group to give us access to command queues so we can later
    // give commands to the GPU. There are different families of queues with different capabilities.
    let (device, mut queue_group) = {
        use gfx_hal::queue::QueueFamily;

        let queue_family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .expect("No compatible queue family found");

        let mut gpu = unsafe {
            use gfx_hal::adapter::PhysicalDevice;

            adapter
                .physical_device
                .open(&[(queue_family, &[1.0])], gfx_hal::Features::empty())
                .expect("Failed to open device")
        };

        (gpu.device, gpu.queue_groups.pop().unwrap())
    };

    // In order to render anything, we have to send commands to the GPU via a command queue.
    // To do this efficiently, we batch those commands together in a structure called a command buffer.
    // These command buffers are allocated from a command pool.
    let (command_pool, mut command_buffer) = unsafe {
        use gfx_hal::command::Level;
        use gfx_hal::pool::{CommandPool, CommandPoolCreateFlags};

        let mut command_pool = device
            .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
            .expect("Out of memory");

        let command_buffer = command_pool.allocate_one(Level::Primary);

        (command_pool, command_buffer)
    };

    // --- In order to build a useful command buffer, we’ll need to create a render pass and a pipeline.

    // The first thing we need for the render pass is a color format.
    // We get a list of supported formats and try to pick the first one that supports SRGB
    // (so gamma correction is handled for us).
    let surface_color_format = {
        use gfx_hal::format::{ChannelType, Format};

        let supported_formats = surface
            .supported_formats(&adapter.physical_device)
            .unwrap_or(vec![]);

        let default_format = *supported_formats.get(0).unwrap_or(&Format::Rgba8Srgb);

        supported_formats
            .into_iter()
            .find(|format| format.base_format().1 == ChannelType::Srgb)
            .unwrap_or(default_format)
    };

    let render_pass = {
        use gfx_hal::image::Layout;
        use gfx_hal::pass::{
            Attachment, AttachmentLoadOp, AttachmentOps, AttachmentStoreOp, SubpassDesc,
        };

        let color_attachment = Attachment {
            format: Some(surface_color_format),
            samples: 1,
            ops: AttachmentOps::new(AttachmentLoadOp::Clear, AttachmentStoreOp::Store),
            stencil_ops: AttachmentOps::DONT_CARE,
            layouts: Layout::Undefined..Layout::Present,
        };

        let subpass = SubpassDesc {
            colors: &[(0, Layout::ColorAttachmentOptimal)],
            depth_stencil: None,
            inputs: &[],
            resolves: &[],
            preserves: &[],
        };

        unsafe {
            device
                .create_render_pass(&[color_attachment], &[subpass], &[])
                .expect("Out of memory")
        }
    };

    let pipeline_layout = unsafe {
        use gfx_hal::pso::ShaderStageFlags;

        // The GPU doesn’t know anything about structs - all it knows is that we’re going to give it some bytes.
        // What it wants to know is which of those bytes it should use for each specific shader stage.
        // In our case, we only care about the vertex shader, and the number of bytes is however big the struct is.
        let push_constant_bytes = std::mem::size_of::<PushConstants>() as u32;

        device
            .create_pipeline_layout(&[], &[(ShaderStageFlags::VERTEX, 0..push_constant_bytes)])
            .expect("Out of memory")
    };

    // Vertex and fragment shaders.
    // these shaders are written in GLSL - which gfx-hal doesn’t support directly.
    // To use them, we’ll have to first compile them to SPIR-V - a more efficient intermediate representation.
    let vertex_shader = include_str!("shaders/part-3.vert");
    let fragment_shader = include_str!("shaders/part-3.frag");

    fn compile_shader(glsl: &str, shader_kind: ShaderKind) -> Vec<u32> {
        let mut compiler = shaderc::Compiler::new().unwrap();

        let compiled_shader = compiler
            .compile_into_spirv(glsl, shader_kind, "unnamed", "main", None)
            .expect("Failed to compile shader");

        compiled_shader.as_binary().to_vec()
    }

    // Create a pipeline with the given layout and shaders.
    unsafe fn make_pipeline<B: gfx_hal::Backend>(
        device: &B::Device,
        render_pass: &B::RenderPass,
        pipeline_layout: &B::PipelineLayout,
        vertex_shader: &str,
        fragment_shader: &str,
    ) -> B::GraphicsPipeline {
        use gfx_hal::pass::Subpass;
        use gfx_hal::pso::{
            BlendState, ColorBlendDesc, ColorMask, EntryPoint, Face, GraphicsPipelineDesc,
            InputAssemblerDesc, Primitive, PrimitiveAssemblerDesc, Rasterizer, Specialization,
        };
        let vertex_shader_module = device
            .create_shader_module(&compile_shader(vertex_shader, ShaderKind::Vertex))
            .expect("Failed to create vertex shader module");

        let fragment_shader_module = device
            .create_shader_module(&compile_shader(fragment_shader, ShaderKind::Fragment))
            .expect("Failed to create fragment shader module");

        // EntryPoint defines how your shader begins executing
        let (vs_entry, fs_entry) = (
            EntryPoint {
                entry: "main",
                module: &vertex_shader_module,
                specialization: Specialization::default(),
            },
            EntryPoint {
                entry: "main",
                module: &fragment_shader_module,
                specialization: Specialization::default(),
            },
        );

        // This describes how our pipeline should take in vertices and output primitives (in our case, triangles).
        // It says we want to render our vertices as a list of triangles, and we pass the vertex shader
        // entry point we prepared.
        let primitive_assembler = {
            use gfx_hal::format::Format;
            use gfx_hal::pso::{AttributeDesc, Element, VertexBufferDesc, VertexInputRate};

            PrimitiveAssemblerDesc::Vertex {
                buffers: &[VertexBufferDesc {
                    binding: 0,
                    stride: std::mem::size_of::<Vertex>() as u32,
                    rate: VertexInputRate::Vertex,
                }],

                attributes: &[
                    AttributeDesc {
                        location: 0,
                        binding: 0,
                        element: Element {
                            format: Format::Rgb32Sfloat,
                            offset: 0,
                        },
                    },
                    AttributeDesc {
                        location: 1,
                        binding: 0,
                        element: Element {
                            format: Format::Rgb32Sfloat,
                            offset: 12,
                        },
                    },
                ],
                input_assembler: InputAssemblerDesc::new(Primitive::TriangleList),
                vertex: vs_entry,
                tessellation: None,
                geometry: None,
            }
        };

        let mut pipeline_desc = GraphicsPipelineDesc::new(
            primitive_assembler,
            Rasterizer {
                cull_face: Face::BACK,
                ..Rasterizer::FILL
            },
            Some(fs_entry),
            pipeline_layout,
            Subpass {
                index: 0,
                main_pass: render_pass,
            },
        );

        pipeline_desc.blender.targets.push(ColorBlendDesc {
            mask: ColorMask::ALL,
            blend: Some(BlendState::ALPHA),
        });

        let pipeline = device
            .create_graphics_pipeline(&pipeline_desc, None)
            .expect("Failed to create graphics pipeline");

        device.destroy_shader_module(vertex_shader_module);
        device.destroy_shader_module(fragment_shader_module);

        pipeline
    };

    let pipeline = unsafe {
        make_pipeline::<gfx_backend::Backend>(
            &device,
            &render_pass,
            &pipeline_layout,
            vertex_shader,
            fragment_shader,
        )
    };

    // --- Synchronization primitives

    // The GPU can execute in parallel to the CPU, so we need some way of ensuring
    // that they don’t interfere with each other.

    // A fence allows the CPU to wait for the GPU.
    // In our case, we’re going to use it to wait for the command buffer we submit to be available for writing again.
    let submission_complete_fence = device.create_fence(true).expect("Out of memory");
    // A semaphore allows you to synchronize different processes within the GPU.
    // In our case we’re going to use it to tell the GPU to wait until the frame has finished rendering before displaying it onscreen.
    let rendering_complete_semaphore = device.create_semaphore().expect("Out of memory");

    // ---------------

    // The `teapot_mesh.bin` is just a `Vec<Vertex>` that was serialized
    // using the `bincode` crate. So we can deserialize it directly.
    let binary_mesh_data = include_bytes!("./assets/teapot_mesh.bin");
    let mesh: Vec<Vertex> =
        bincode::deserialize(binary_mesh_data).expect("Failed to deserialize mesh");

    // Create an empty buffer with the given size and properties.
    unsafe fn make_buffer<B: gfx_hal::Backend>(
        device: &B::Device,
        physical_device: &B::PhysicalDevice,
        buffer_len: usize,
        usage: gfx_hal::buffer::Usage,
        properties: gfx_hal::memory::Properties,
    ) -> (B::Memory, B::Buffer) {
        use gfx_hal::{adapter::PhysicalDevice, MemoryTypeId};

        let mut buffer = device
            .create_buffer(buffer_len as u64, usage)
            .expect("Failed to create buffer");

        let req = device.get_buffer_requirements(&buffer);

        let memory_types = physical_device.memory_properties().memory_types;

        let memory_type = memory_types
            .iter()
            .enumerate()
            .find(|(id, mem_type)| {
                let type_supported = req.type_mask & (1_u32 << id) != 0;
                type_supported && mem_type.properties.contains(properties)
            })
            .map(|(id, _ty)| MemoryTypeId(id))
            .expect("No compatible memory type available");

        let buffer_memory = device
            .allocate_memory(memory_type, req.size)
            .expect("Failed to allocate buffer memory");

        device
            .bind_buffer_memory(&buffer_memory, 0, &mut buffer)
            .expect("Failed to bind buffer memory");

        (buffer_memory, buffer)
    }

    let vertex_buffer_len = mesh.len() * std::mem::size_of::<Vertex>();

    let (vertex_buffer_memory, vertex_buffer) = unsafe {
        use gfx_hal::buffer::Usage;
        use gfx_hal::memory::Properties;

        make_buffer::<gfx_backend::Backend>(
            &device,
            &adapter.physical_device,
            vertex_buffer_len,
            Usage::VERTEX,
            Properties::CPU_VISIBLE,
        )
    };

    // --- Memory management

    // here’s the part that sucks: we have to clean up after ourselves.
    // This wouldn’t be so bad if not for a specific intersection of two things.
    // Namely that winit takes ownership over our resources and drops them, but gfx
    // requires us to manually delete them (which we can’t do because they’ve been moved).

    // The solution is to wrap our resources in a struct with a Drop implementation to clean them up.
    struct Resources<B: gfx_hal::Backend> {
        instance: Option<B::Instance>,
        surface: B::Surface,
        device: B::Device,
        render_passes: Vec<B::RenderPass>,
        pipeline_layouts: Vec<B::PipelineLayout>,
        pipelines: Vec<B::GraphicsPipeline>,
        command_pool: B::CommandPool,
        submission_complete_fence: B::Fence,
        rendering_complete_semaphore: B::Semaphore,
        vertex_buffer_memory: B::Memory,
        vertex_buffer: B::Buffer,
    }

    // Unfortunately, we can’t implement Drop for this struct directly. This is because the signature
    // of drop takes a &mut self parameter, while the signatures of the destroy_<something> functions
    // take a self parameter (meaning that they want to take ownership of self).

    // So we need a way to move our resources out of a &mut reference. One way to do this is to put
    // our resources in a ManuallyDrop, and use the take method to pull out the contents:
    struct ResourceHolder<B: gfx_hal::Backend>(ManuallyDrop<Resources<B>>);

    impl<B: gfx_hal::Backend> Drop for ResourceHolder<B> {
        fn drop(&mut self) {
            unsafe {
                let Resources {
                    instance,
                    mut surface,
                    device,
                    command_pool,
                    render_passes,
                    pipeline_layouts,
                    pipelines,
                    submission_complete_fence,
                    rendering_complete_semaphore,
                    vertex_buffer_memory,
                    vertex_buffer,
                } = ManuallyDrop::take(&mut self.0);

                device.destroy_semaphore(rendering_complete_semaphore);
                device.destroy_fence(submission_complete_fence);
                for pipeline in pipelines {
                    device.destroy_graphics_pipeline(pipeline);
                }
                for pipeline_layout in pipeline_layouts {
                    device.destroy_pipeline_layout(pipeline_layout);
                }
                for render_pass in render_passes {
                    device.destroy_render_pass(render_pass);
                }
                device.destroy_command_pool(command_pool);
                device.free_memory(vertex_buffer_memory);
                device.destroy_buffer(vertex_buffer);
                surface.unconfigure_swapchain(&device);
                instance
                    .expect("Instance was not initialized.")
                    .destroy_surface(surface);
            }
        }
    }

    unsafe {
        use gfx_hal::memory::Segment;

        let mapped_memory = device
            .map_memory(&vertex_buffer_memory, Segment::ALL)
            .expect("Failed to map memory");

        std::ptr::copy_nonoverlapping(mesh.as_ptr() as *const u8, mapped_memory, vertex_buffer_len);

        device
            .flush_mapped_memory_ranges(vec![(&vertex_buffer_memory, Segment::ALL)])
            .expect("Out of memory");

        device.unmap_memory(&vertex_buffer_memory);
    }

    let mut resource_holder: ResourceHolder<gfx_backend::Backend> =
        ResourceHolder(ManuallyDrop::new(Resources {
            instance,
            surface,
            device,
            command_pool,
            render_passes: vec![render_pass],
            pipeline_layouts: vec![pipeline_layout],
            pipelines: vec![pipeline],
            submission_complete_fence,
            rendering_complete_semaphore,
            vertex_buffer_memory,
            vertex_buffer,
        }));

    // --- Rendering

    // This will be very important later! It must be initialized to `true` so
    // that we rebuild the swapchain on the first frame.
    let mut should_configure_swapchain = true;

    // We'll use the elapsed time to drive some animations later on.
    let start_time = std::time::Instant::now();

    // Create a matrix that positions, scales, and rotates.
    fn make_transform(translate: [f32; 3], angle: f32, scale: f32) -> [[f32; 4]; 4] {
        let c = angle.cos() * scale;
        let s = angle.sin() * scale;
        let [dx, dy, dz] = translate;

        [
            [c, 0., s, 0.],
            [0., scale, 0., 0.],
            [-s, 0., c, 0.],
            [dx, dy, dz, 1.],
        ]
    }

    // Note that this takes a `move` closure. This means it will take ownership
    // over any resources referenced within. It also means they will be dropped
    // only when the application is quit.
    event_loop.run(move |event, _, control_flow| {
        use winit::event::{Event, WindowEvent};
        use winit::event_loop::ControlFlow;

        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => *control_flow = ControlFlow::Exit,
                WindowEvent::Resized(dims) => {
                    surface_extent = Extent2D {
                        width: dims.width,
                        height: dims.height,
                    };
                    // Note the should_configure_swapchain variable. The swapchain
                    // is a chain of images for rendering onto. Each frame, one of those
                    // images is displayed onscreen. I’ll explain more about this
                    // later - for now just make sure you set this variable to true.
                    should_configure_swapchain = true;
                }
                WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                    surface_extent = Extent2D {
                        width: new_inner_size.width,
                        height: new_inner_size.height,
                    };
                    should_configure_swapchain = true;
                }
                _ => (),
            },
            Event::MainEventsCleared => window.request_redraw(),
            Event::RedrawRequested(_) => {
                // --- Rendering
                let res: &mut Resources<_> = &mut resource_holder.0;
                let render_pass = &res.render_passes[0];
                let pipeline_layout = &res.pipeline_layouts[0];
                let pipeline = &res.pipelines[0];

                // We’re about to reset our command buffer - which would be terrible if the commands
                // hadn’t been submitted to the GPU yet. So what we’ll do is wait for the fence
                // before we reset it, and later when we submit the command buffer, we’ll tell
                // it to signal the fence once it’s done.
                unsafe {
                    use gfx_hal::pool::CommandPool;

                    // We refuse to wait more than a second, to avoid hanging.
                    let render_timeout_ns = 1_000_000_000;

                    res.device
                        .wait_for_fence(&res.submission_complete_fence, render_timeout_ns)
                        .expect("Out of memory or device lost");

                    res.device
                        .reset_fence(&res.submission_complete_fence)
                        .expect("Out of memory");

                    res.command_pool.reset(false);
                }

                // -- Swapchain (https://gfx-rs.github.io/2019/10/01/update.html#new-swapchain-model)

                // It’s a chain of images that we can render onto and then present to our window.
                // While we’re showing one of them on screen, we can render to a different one.
                // Then once we’re done rendering, we can swap them.
                if should_configure_swapchain {
                    use gfx_hal::window::SwapchainConfig;

                    let caps = res.surface.capabilities(&adapter.physical_device);

                    let mut swapchain_config =
                        SwapchainConfig::from_caps(&caps, surface_color_format, surface_extent);

                    // This seems to fix some fullscreen slowdown on macOS.
                    if caps.image_count.contains(&3) {
                        swapchain_config.image_count = 3;
                    }

                    surface_extent = swapchain_config.extent;

                    unsafe {
                        res.surface
                            .configure_swapchain(&res.device, swapchain_config)
                            .expect("Failed to configure swapchain");
                    };

                    should_configure_swapchain = false;
                }

                // The swapchain is now ready. To start rendering, we’ll need to acquire an image from it.
                let surface_image = unsafe {
                    // We refuse to wait more than a second, to avoid hanging.
                    let acquire_timeout_ns = 1_000_000_000;

                    match res.surface.acquire_image(acquire_timeout_ns) {
                        Ok((image, _)) => image,
                        Err(_) => {
                            should_configure_swapchain = true;
                            return;
                        }
                    }
                };

                // Framebuffer is what actually connects images (like the one we got from our swapchain)
                // to attachments within the render pass (like the one color attachment we specified).
                let framebuffer = unsafe {
                    use std::borrow::Borrow;

                    use gfx_hal::image::Extent;

                    res.device
                        .create_framebuffer(
                            render_pass,
                            vec![surface_image.borrow()],
                            Extent {
                                width: surface_extent.width,
                                height: surface_extent.height,
                                depth: 1,
                            },
                        )
                        .unwrap()
                };

                // The very last thing to create before we start recording commands is the viewport.
                // This is just a structure defining an area of the window, which can be used
                // to clip (scissor) or scale (viewport) the output of your rendering.
                let viewport = {
                    use gfx_hal::pso::{Rect, Viewport};

                    Viewport {
                        rect: Rect {
                            x: 0,
                            y: 0,
                            w: surface_extent.width as i16,
                            h: surface_extent.height as i16,
                        },
                        depth: 0.0..1.0,
                    }
                };

                // --- Vertices on screen (forming teapot)

                let angle = start_time.elapsed().as_secs_f32();

                let teapots = &[PushConstants {
                    transform: make_transform([0., 0., 0.5], angle, 1.0),
                }];

                // Returns a view of a struct as a slice of `u32`s.
                //
                // Note that this assumes the struct divides evenly into
                // 4-byte chunks. If the contents are all `f32`s, which they
                // often are, then this will always be the case.
                unsafe fn push_constant_bytes<T>(push_constants: &T) -> &[u32] {
                    let size_in_bytes = std::mem::size_of::<T>();
                    let size_in_u32s = size_in_bytes / std::mem::size_of::<u32>();
                    let start_ptr = push_constants as *const T as *const u32;
                    std::slice::from_raw_parts(start_ptr, size_in_u32s)
                }

                // --- Graphics commands

                unsafe {
                    use gfx_hal::command::{
                        ClearColor, ClearValue, CommandBuffer, CommandBufferFlags, SubpassContents,
                    };

                    // A command buffer must always start with a begin command
                    command_buffer.begin_primary(CommandBufferFlags::ONE_TIME_SUBMIT);

                    command_buffer.set_viewports(0, &[viewport.clone()]);
                    command_buffer.set_scissors(0, &[viewport.rect]);

                    command_buffer.bind_vertex_buffers(
                        0,
                        vec![(&res.vertex_buffer, gfx_hal::buffer::SubRange::WHOLE)],
                    );

                    // Next we begin the render pass. We tell it to clear the color attachment to black before rendering
                    command_buffer.begin_render_pass(
                        render_pass,
                        &framebuffer,
                        viewport.rect,
                        &[ClearValue {
                            color: ClearColor {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        }],
                        SubpassContents::Inline,
                    );
                    // Next we bind our pipeline. Now any triangles we draw will be rendered
                    // with the settings and shaders of that pipeline
                    command_buffer.bind_graphics_pipeline(pipeline);
                    // Now the actual draw call itself. We’ve already bound everything we need.
                    for teapot in teapots {
                        use gfx_hal::pso::ShaderStageFlags;

                        command_buffer.push_graphics_constants(
                            pipeline_layout,
                            ShaderStageFlags::VERTEX,
                            0,
                            push_constant_bytes(teapot),
                        );

                        let vertex_count = mesh.len() as u32;
                        command_buffer.draw(0..vertex_count, 0..1);
                    }
                    // Then finally, we can end the render pass, and our command buffer
                    command_buffer.end_render_pass();
                    command_buffer.finish();
                }

                // --- Submission

                // Submission simply contains the command buffers to submit, as well as a list
                // of semaphores to signal once rendering is complete.
                unsafe {
                    use gfx_hal::queue::{CommandQueue, Submission};

                    let submission = Submission {
                        command_buffers: vec![&command_buffer],
                        wait_semaphores: None,
                        signal_semaphores: vec![&res.rendering_complete_semaphore],
                    };

                    queue_group.queues[0].submit(submission, Some(&res.submission_complete_fence));
                    // Finally we call present and pass our rendering_complete_semaphore.
                    // This will wait until the semaphore signals and then display the finished image on screen:
                    let result = queue_group.queues[0].present(
                        &mut res.surface,
                        surface_image,
                        Some(&res.rendering_complete_semaphore),
                    );

                    should_configure_swapchain |= result.is_err();

                    res.device.destroy_framebuffer(framebuffer);
                }
            }
            _ => (),
        }
    });
}
