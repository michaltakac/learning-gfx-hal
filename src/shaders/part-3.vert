// shaders/part-3.vert
#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout(push_constant) uniform PushConstants {
    mat4 transform;
} push_constants;

layout(location = 0) out vec4 vertex_color;

void main() {
    // We don’t need to do this, but it’s more interesting than a flat color.
    vertex_color = vec4(abs(normal), 1.0);
    gl_Position = push_constants.transform * vec4(position, 1.0);
}
