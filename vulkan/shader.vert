
#version 450


layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCord;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragtex;
 
layout(binding = 0)  uniform objeto {
 int obj;
}instancia;

layout(binding = 2) uniform UniformBufferObject{
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

void main() {

vec4 v= vec4(inPosition,1.0f) +vec4(instancia.obj,0.0f,3.0f,0.0f);
 gl_Position = ubo.proj * ubo.view * ubo.model * v;
    fragColor = inColor;
    fragtex=inTexCord;
}