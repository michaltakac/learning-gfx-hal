// shaders/part-1.vert
#version 450
#extension GL_ARB_separate_shader_objects : enable

// It’s so simple that it doesn’t even have any inputs.
// Instead we hardcode the three vertices of the triangle,
// and use the gl_VertexIndex build-in to set the position
// based on which vertex we’re on.

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(-0.5, 0.5),
    vec2(0.5, 0.5)
);

void main() {
    vec2 pos = positions[gl_VertexIndex];
    gl_Position = vec4(pos, 0.0, 1.0);
}
