#version 450

layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragtex;
layout(binding = 1) uniform sampler2D texsampler;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = texture(texsampler,fragtex);
}