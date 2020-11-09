// shaders/part-1.frag
#version 450
#extension GL_ARB_separate_shader_objects : enable

// All it does is output a nice lilac color.

layout(location = 0) out vec4 fragment_color;

void main() {
    fragment_color = vec4(0.5, 0.5, 1.0, 1.0);
}