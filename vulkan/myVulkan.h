









#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>


#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <map>
#include <optional>
#include <set>
#include <algorithm>
#include <fstream>
#include <array>
#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <unordered_map>
#include <cmath>

#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <assimp/Importer.hpp>
#include <thread>


#define IMGUI_IMPLEMENTATION

#include <imgui_single_file.h>
#include <imgui_impl_glfw.cpp>
#include <imgui_impl_vulkan.cpp>


template <class Clock, class Duration>
void
sleep_until(std::chrono::time_point<Clock, Duration> tp)
{
    using namespace std::chrono;
    std::this_thread::sleep_until(tp - 10us);
    while (tp >= Clock::now())
        ;
}


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 1;
struct Vertex {
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;


    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color && texCoord == other.texCoord;
    }

    static VkVertexInputBindingDescription getBindingDescription() {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        return attributeDescriptions;
    }
};

using namespace Assimp;
//elementos
struct UniformBufferObject {
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};


static int mouseY = 0;
static int mouseX = 0;
static int move = 0;
static int moveL = 0;
static int rotate = 0;

int Foco = 0;

int  objetoFoco(int foco) {

    int newFoco = Foco + foco;

    if (newFoco != Foco) {
        move = 0;
        moveL = 0;
        rotate = 0;
    }
    Foco = newFoco;

    return newFoco;
}


namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

struct instancia {
    int obj;
};





struct Objeto {


    VkBuffer instanciabuffer;
    VkDeviceMemory instanciamem;
    int instacia;
    void* pinstanciaBuffer;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    VkBuffer indexBuffer;
    VkDeviceMemory indexmem;
    void* indexdata;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    void* vertexdata;

    VkImageView textureImageView;
    VkSampler textureSampler;
    VkImage image;
    VkDeviceMemory imagemem;

    struct Transform {


        int move = 0;

        int moveL = 0;

        int rotate = 0;

    };

    Transform transform{};



    VkBuffer ubomatrix;
    VkDeviceMemory ubomem;


    void* pUboMatrix;



    void loadModel(const char* modelo) {



        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, modelo)) {
            throw std::runtime_error(warn + err);
        }

        std::cout << "verticies" << shapes[0].mesh.num_face_vertices.size() << std::endl;

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};



                vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
 attrib.texcoords[2 * index.texcoord_index + 0],
 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }



    }

    void setBuffers() {

        vertices = {

            {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {0.0f, 1.0f}},
                { {0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
             {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
             {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}}





        };

        indices = {
         0,1,2,2,3,0

        };



    }

};

double interpolate(double x1, double y1, double x2, double y2, double x) {
    return y1 + (y2 - y1) * ((x - x1) / (x2 - x1));
}




const std::string MODEL_PATH = "source/hall/hallway.obj";
const std::string TEXTURE_PATH = "source/imagem_programador.jpg";

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};
const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class vulkan {
public:

    void init() {
        initWindow();
        initVulkan();
    }
    void loop(VkCommandBuffer func(vulkan& app, int imageindex)) {

        mainLoop(func);
        cleanup();
    }

private:


    struct myListerne {
        int height = 0;
        int width = 0;

        std::vector<void(*)(void)> funcs;
        std::vector<Objeto*>objs;

        void addFunc(void (*func_ptr)(void)) { funcs.push_back(func_ptr); }

        void excluir(int index) {
            std::vector<void(*)(void)> Newfuncs;

            for (int i = 0; i < funcs.size(); i++) {

                if (i != index) {
                    Newfuncs.push_back(funcs[i]);
                }
            }
            funcs = Newfuncs;
        }

        void listen(Objeto& obj, GLFWwindow* w) {

            glfwGetFramebufferSize(w, &width, &height);
            int wordPosX = interpolate(-1, 0, 1, height, mouseX);
            int wordposY = interpolate(-1, 0, 1, width, mouseY);

            for (int i = 0; i < objs.size(); i++) {
                if (objs[i]->image) {
                    funcs[i]();
                }

            }







        }

        void flushFuncs() { funcs.clear(); }

    };

public:
    //inicialização da janela
    GLFWwindow* window = nullptr;
    uint32_t glfwExtensionCount;
    const char** glfwExtensions = nullptr;

    //inicialização do vulkan
    VkInstance instance = nullptr;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    uint32_t deviceCount = 0;
    VkPhysicalDeviceProperties deviceProperties = {};
    VkPhysicalDeviceFeatures deviceFeatures = {};
    VkDevice vkDevice = {};







    //feramentas de desenho
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    VkQueue presentQueue = {};
    VkQueue graphicsQueue = {};
    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;
    std::vector<VkImageView> swapChainImageViews;
    std::vector<VkFramebuffer> swapChainFramebuffers;


    std::vector<VkCommandBuffer> commandBuffers;
    std::vector<VkCommandBuffer> renderCommandBuffers;
    VkCommandBuffer cmdLimpeza;
    VkCommandPool commandPool;

    std::vector<VkSemaphore> sinalizadorDeLipezaFrameBuffer;
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    VkSemaphore renderGui;
    std::vector<VkFence> inFlightFences;

    bool framebufferResized = false;

    VkRenderPass renderPass;
    VkRenderPass renderPassLimpador;
    VkPipelineLayout pipelineLayout;

    VkPipeline graphicsPipeline;

    uint32_t currentFrame = 0;


    struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() {

            return graphicsFamily.has_value() && presentFamily.has_value();
        }
    };
    struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };




    VkBuffer indexBuffer;
    VkDeviceMemory indexmem;
    void* indexdata;

    VkBuffer vertexBuffer;
    VkDeviceMemory vertexBufferMemory;
    void* vertexdata;

    VkBuffer bufferImtermediario;
    VkDeviceMemory intermem{};

    VkBuffer ubomatrix;
    VkDeviceMemory matrixmem;
    void* pmatrix;

    VkImageView textureImageView;
    VkSampler textureSampler;
    VkImage image;
    VkDeviceMemory imagemem;


    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkBuffer Instancia;
    VkDeviceMemory instmem;





    //elementos
    struct UniformBufferObject {
        glm::mat4 model;
        glm::mat4 view;
        glm::mat4 proj;
    };

    struct

        std::chrono::high_resolution_clock::time_point inicio;
    int frameCount = 0;
    float time;

    // VkPipelineLayout pipelineLayoutM;
    VkDescriptorSetLayout setLayout;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;

    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

private:

    void loadModel() {

        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) {
            throw std::runtime_error(warn + err);
        }

        std::cout << "verticies" << shapes[0].mesh.num_face_vertices.size() << std::endl;

        std::unordered_map<Vertex, uint32_t> uniqueVertices{};

        for (const auto& shape : shapes) {
            for (const auto& index : shape.mesh.indices) {
                Vertex vertex{};



                vertex.pos = {
                attrib.vertices[3 * index.vertex_index + 0],
                attrib.vertices[3 * index.vertex_index + 1],
                attrib.vertices[3 * index.vertex_index + 2]
                };

                vertex.texCoord = {
 attrib.texcoords[2 * index.texcoord_index + 0],
 1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                };
                if (uniqueVertices.count(vertex) == 0) {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }

                indices.push_back(uniqueVertices[vertex]);
            }
        }



    }

    void createSurface() {

        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS)
        {
            throw std::runtime_error("falha ao criar a supefice");
        }
        else { std::cout << "sucesso ao criar a superfice " << std::endl; }


    }

    void initVulkan() {

        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        createCommandPoll();
        createDepthResources();
        createBuffer(sizeof(UniformBufferObject), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, ubomatrix, matrixmem);
        createBuffer(sizeof(instancia), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, Instancia, instmem);
        //  createTextureImage();//
         // createTextureImageView();//
         // createTextureSampler();//
         // loadModel();//
        //  createVertexBuffer();
         // createIndexBuffer();
         // createUboBuffer();
        createDescriptorPool();

        createCommandBuffer();
        createSyncObjects();
        inicio = std::chrono::high_resolution_clock::now();

    }


    void createDepthResources() {
        VkFormat depthFormat = findDepthFormat();

        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = createImageView(depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);



    }

    VkFormat findSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features) {

        for (VkFormat format : candidates) {
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(physicalDevice, format, &props);

            if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features) {
                return format;
            }
            else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }

        throw std::runtime_error("failed to find supported format!");
    }

    VkFormat findDepthFormat() {
        return findSupportedFormat(
            { VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT },
            VK_IMAGE_TILING_OPTIMAL,
            VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
        );
    }

    bool hasStencilComponent(VkFormat format) {
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    void createTextureSampler() {

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        //cordenadas
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        VkPhysicalDeviceProperties prop{};

        vkGetPhysicalDeviceProperties(physicalDevice, &prop);
        //filtragem
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = prop.limits.maxSamplerAnisotropy;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        //normalizada 1,0
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;


        if (vkCreateSampler(vkDevice, &samplerInfo, nullptr, &textureSampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

    }
    void createTextureImageView() {
        textureImageView = createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }





    void createImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) {
        VkImageCreateInfo imageInfo{};
        imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageInfo.imageType = VK_IMAGE_TYPE_2D;
        imageInfo.extent.width = width;
        imageInfo.extent.height = height;
        imageInfo.extent.depth = 1;
        imageInfo.mipLevels = 1;
        imageInfo.arrayLayers = 1;
        imageInfo.format = format;
        imageInfo.tiling = tiling;
        imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageInfo.usage = usage;
        imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateImage(vkDevice, &imageInfo, nullptr, &image) != VK_SUCCESS) {
            throw std::runtime_error("failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(vkDevice, image, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(vkDevice, &allocInfo, nullptr, &imageMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate image memory!");
        }

        vkBindImageMemory(vkDevice, image, imageMemory, 0);
    }

    VkImageView createImageView(VkImage image, VkFormat format, VkImageAspectFlags aspectFlags) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = image;
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = format;
        viewInfo.subresourceRange.aspectMask = aspectFlags;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if (vkCreateImageView(vkDevice, &viewInfo, nullptr, &imageView) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture image view!");
        }

        return imageView;
    }



    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;
        barrier.srcAccessMask = 0; // TODO
        barrier.dstAccessMask = 0; // TODO


        if (newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

            if (hasStencilComponent(format)) {
                barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
            }
        }

        VkPipelineStageFlags sourceStage;
        VkPipelineStageFlags destinationStage;

        if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourceStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        }
        else if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            sourceStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        }
        else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(
            commandBuffer,
            sourceStage /* TODO */, destinationStage /* TODO */,
            0,
            0, nullptr,
            0, nullptr,
            1, &barrier
        );


        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height) {
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;

        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;

        region.imageOffset = { 0, 0, 0 };
        region.imageExtent = {
            width,
            height,
            1
        };

        vkCmdCopyBufferToImage(
            commandBuffer,
            buffer,
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region
        );

        endSingleTimeCommands(commandBuffer);
    }



    void createTextureImage() {
        int texWidth, texHeight, texChannels = {};
        stbi_uc* pixels = stbi_load(TEXTURE_PATH.c_str(), &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        VkBuffer temp;
        VkDeviceMemory tempmem;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            , VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            , temp, tempmem);

        void* data;
        vkMapMemory(vkDevice, tempmem, 0, imageSize, 0, &data);
        memcpy(data, pixels, (size_t)imageSize);
        vkUnmapMemory(vkDevice, tempmem);
        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, image, imagemem);


        transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(temp, image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


        vkDestroyBuffer(vkDevice, temp, nullptr);
        vkFreeMemory(vkDevice, tempmem, nullptr);



    }

    void createDescriptorPool() {

        std::array<VkDescriptorPoolSize, 3> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(1);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[2].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[2].descriptorCount = static_cast<uint32_t>(13);

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if (vkCreateDescriptorPool(vkDevice, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create descriptor pool!");
        }

    }

    void createDescriptorSetsT() {






        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, setLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();


        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(vkDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");


        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {

            VkDescriptorBufferInfo bufferInfo{};
            //    bufferInfo.buffer =ubomatrix[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;


            std::array< VkWriteDescriptorSet, 2> descriptorWrite{};
            descriptorWrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[0].dstSet = descriptorSets[i];
            descriptorWrite[0].dstBinding = 0;
            descriptorWrite[0].dstArrayElement = 0;

            descriptorWrite[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite[0].descriptorCount = 1;
            descriptorWrite[0].pBufferInfo = &bufferInfo;
            descriptorWrite[0].pImageInfo = nullptr; // Optional
            descriptorWrite[0].pTexelBufferView = nullptr; // Optional


            descriptorWrite[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[1].dstSet = descriptorSets[i];
            descriptorWrite[1].dstBinding = 1;
            descriptorWrite[1].dstArrayElement = 0;

            descriptorWrite[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite[1].descriptorCount = 1;
            descriptorWrite[1].pImageInfo = &imageInfo; // Optional
            descriptorWrite[1].pTexelBufferView = nullptr; // Optional




            vkUpdateDescriptorSets(vkDevice, (uint32_t)descriptorWrite.size(), descriptorWrite.data(), 0, nullptr);
        }


    }

    void createDescriptorSetLayoutT() {

        VkDescriptorSetLayoutBinding lbind{};
        lbind.binding = 0;
        lbind.descriptorCount = 1;
        lbind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lbind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 2> bindings = { lbind, samplerLayoutBinding };
        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(vkDevice, &createInfo, nullptr, &setLayout) != VK_SUCCESS)
            throw std::runtime_error("falha ao criar o descritor");

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.pSetLayouts = &setLayout;
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        if (vkCreatePipelineLayout(vkDevice, &plinfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("falha ao criar o pipelineLayout");

    }

    void createDescriptorSetLayoutT(VkDescriptorType descT, VkShaderStageFlags shaderAcess, uint32_t bind, VkPipelineLayout* pipeline, VkDescriptorSetLayout* setLayout) {

        VkDescriptorSetLayoutBinding lbind{};
        lbind.binding = bind;
        lbind.descriptorCount = 1;
        lbind.descriptorType = descT;
        lbind.stageFlags = shaderAcess;


        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.bindingCount = 1;
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.pBindings = &lbind;

        if (vkCreateDescriptorSetLayout(vkDevice, &createInfo, nullptr, setLayout) != VK_SUCCESS)
            throw std::runtime_error("falha ao criar o descritor");

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.pSetLayouts = setLayout;
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        if (vkCreatePipelineLayout(vkDevice, &plinfo, nullptr, pipeline) != VK_SUCCESS)
            throw std::runtime_error("falha ao criar o pipelineLayout");

    }

    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("failed to find suitable memory type!");

    }

    //funcoes para criar objeto
    void createTextureImageView(VkImage image, VkImageView* textureimageview) {
        *textureimageview = createImageView(image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }
    void createTextureSampler(VkSampler* texturesampler) {

        VkSamplerCreateInfo samplerInfo{};
        samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;

        //cordenadas
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;

        VkPhysicalDeviceProperties prop{};

        vkGetPhysicalDeviceProperties(physicalDevice, &prop);
        //filtragem
        samplerInfo.anisotropyEnable = VK_TRUE;
        samplerInfo.maxAnisotropy = prop.limits.maxSamplerAnisotropy;

        samplerInfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        //normalizada 1,0
        samplerInfo.unnormalizedCoordinates = VK_FALSE;

        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
        samplerInfo.mipLodBias = 0.0f;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;


        if (vkCreateSampler(vkDevice, &samplerInfo, nullptr, texturesampler) != VK_SUCCESS) {
            throw std::runtime_error("failed to create texture sampler!");
        }

    }
    void createTextureImage(const char* textura, Objeto& obj) {
        int texWidth, texHeight, texChannels = {};
        stbi_uc* pixels = stbi_load(textura, &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        VkBuffer temp;
        VkDeviceMemory tempmem;

        createBuffer(imageSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            , VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            , temp, tempmem);

        void* data;
        vkMapMemory(vkDevice, tempmem, 0, imageSize, 0, &data);
        memcpy(data, pixels, (size_t)imageSize);
        vkUnmapMemory(vkDevice, tempmem);
        stbi_image_free(pixels);

        createImage(texWidth, texHeight, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj.image, obj.imagemem);


        transitionImageLayout(obj.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(temp, obj.image, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(obj.image, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);


        vkDestroyBuffer(vkDevice, temp, nullptr);
        vkFreeMemory(vkDevice, tempmem, nullptr);


        createTextureImageView(obj.image, &obj.textureImageView);
        createTextureSampler(&obj.textureSampler);



    }
    void createVertexBuffer(Objeto& obj) {
        VkDeviceSize bufferSize = sizeof(Vertex) * obj.vertices.size();

        VkBuffer stagingBuffer{};
        VkDeviceMemory stagingBufferMemory{};
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

        void* data;
        vkMapMemory(vkDevice, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, obj.vertices.data(), (size_t)bufferSize);
        vkUnmapMemory(vkDevice, stagingBufferMemory);

        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj.vertexBuffer, obj.vertexBufferMemory);

        copyBuffer(stagingBuffer, obj.vertexBuffer, bufferSize);

        vkFreeMemory(vkDevice, stagingBufferMemory, nullptr);
        vkDestroyBuffer(vkDevice, stagingBuffer, nullptr);



    }
    void createIndexBuffer(Objeto& obj) {

        VkBuffer temp{};
        VkDeviceMemory tempmem{};
        VkDeviceSize size = sizeof(uint32_t) * obj.indices.size();

        createBuffer(size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, temp, tempmem);

        void* indexdata;
        vkMapMemory(vkDevice, tempmem, 0, size, 0, &indexdata);
        memcpy(indexdata, obj.indices.data(), (size_t)size);
        vkUnmapMemory(vkDevice, tempmem);


        createBuffer(size, VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, obj.indexBuffer, obj.indexmem);

        copyBuffer(temp, obj.indexBuffer, size);


        vkDestroyBuffer(vkDevice, temp, nullptr);
        vkFreeMemory(vkDevice, tempmem, nullptr);

    }
    void createUboBuffer(Objeto& obj) {

        VkDeviceSize size = (VkDeviceSize)sizeof(UniformBufferObject);



        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, obj.ubomatrix, obj.ubomem);

        vkMapMemory(vkDevice, obj.ubomem, 0, size, 0, &obj.pUboMatrix);

    }
    void createInstanciaBuffer(Objeto& obj) {

        VkDeviceSize size = sizeof(instancia);

        createBuffer(size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, obj.instanciabuffer, obj.instanciamem);

        vkMapMemory(vkDevice, obj.instanciamem, 0, size, 0, &obj.pinstanciaBuffer);
    }



    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) {

        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void createSyncObjects() {

        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        sinalizadorDeLipezaFrameBuffer.resize(13);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (
                vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS || vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS
                || vkCreateFence(vkDevice, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
            }
        }

        vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &renderGui);

        for (size_t i = 0; i < 13; i++) {
            if (vkCreateSemaphore(vkDevice, &semaphoreInfo, nullptr, &sinalizadorDeLipezaFrameBuffer[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create semaphores!");
            }

        }

    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pInheritanceInfo = nullptr; // Optional
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;


        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("failed to begin recording command buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
        clearValues[1].depthStencil = { 1.0f, 0 };

        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(swapChainExtent.width);
        viewport.height = static_cast<float>(swapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;
        vkCmdSetScissor(commandBuffer, 0, 1, &scissor);

        VkBuffer buffers[] = { vertexBuffer };
        VkBuffer indexbuffers[] = { indexBuffer };
        VkDeviceSize ofsets[] = { 0 };



        vkCmdBindVertexBuffers(commandBuffer, 0, 1, buffers, ofsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

        vkCmdDrawIndexed(commandBuffer, (uint32_t)indices.size(), 1, 0, 0, 0);
        vkCmdEndRenderPass(commandBuffer);

        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
            throw std::runtime_error("falha ao gravar o buffer de comando");
        }
        std::cout << "buffer Criado" << std::endl;

    }

    void recordCommandBuffer(std::vector<VkCommandBuffer>& commandBuffer, uint32_t imageIndex, std::vector<Objeto>& obj) {



        //  std::cout << "inicio da renderizacao  " << std::endl;
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.pInheritanceInfo = nullptr; // Optional
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = swapChainFramebuffers[imageIndex];
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        //  std::cout << "inicio da renderizacao2  " << std::endl;
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };

        //  std::cout << "inicio da renderizacao3  " << std::endl;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        for (size_t i = 0; i < commandBuffer.size(); i++)
        {
            clearValues[1].depthStencil = { 1.0f + commandBuffer.size() * 1 / 100, 0 };
            vkResetCommandBuffer(commandBuffer[i], 0);

            if (vkBeginCommandBuffer(commandBuffer[i], &beginInfo) != VK_SUCCESS) {
                throw std::runtime_error("failed to begin recording command buffer!");
            }


            VkBufferCopy copyUbo{};
            copyUbo.dstOffset = 0;
            copyUbo.srcOffset = 0;
            copyUbo.size = (VkDeviceSize)sizeof(UniformBufferObject);

            VkBufferCopy copyInst{};
            copyInst.dstOffset = 0;
            copyInst.srcOffset = 0;
            copyInst.size = (VkDeviceSize)sizeof(instancia);


            vkCmdCopyBuffer(commandBuffer[i], obj[i].ubomatrix, ubomatrix, 1, &copyUbo);
            vkCmdCopyBuffer(commandBuffer[i], obj[i].instanciabuffer, Instancia, 1, &copyInst);


            if (i < commandBuffer.size() - 1) {
                vkCmdBeginRenderPass(commandBuffer[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            }
            else {
                renderPassInfo.renderPass = renderPassLimpador;
                vkCmdBeginRenderPass(commandBuffer[i], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            }

            vkCmdBindPipeline(commandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);

            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(swapChainExtent.width);
            viewport.height = static_cast<float>(swapChainExtent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(commandBuffer[i], 0, 1, &viewport);

            VkRect2D scissor{};
            scissor.offset = { 0, 0 };
            scissor.extent = swapChainExtent;
            vkCmdSetScissor(commandBuffer[i], 0, 1, &scissor);


            VkDeviceSize ofsets[] = { 0 };




            vkCmdBindDescriptorSets(commandBuffer[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

            //std::cout << "inicio do desenho do objeto: " << i << std::endl;

            vkCmdBindVertexBuffers(commandBuffer[i], 0, 1, &obj[i].vertexBuffer, ofsets);
            vkCmdBindIndexBuffer(commandBuffer[i], obj[i].indexBuffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(commandBuffer[i], (uint32_t)obj[i].indices.size(), 1, 0, 0, 0);
            //  std::cout << "fim do desenho do objeto: " << i << std::endl;


            vkCmdEndRenderPass(commandBuffer[i]);

            if (vkEndCommandBuffer(commandBuffer[i]) != VK_SUCCESS) {
                throw std::runtime_error("falha ao gravar o buffer de comando");
            }
            //  std::cout << "buffer Criado" << std::endl;



        }//fim do laço for

    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if (vkCreateBuffer(vkDevice, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
            throw std::runtime_error("failed to create buffer!");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(vkDevice, buffer, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

        if (vkAllocateMemory(vkDevice, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate buffer memory!");
        }

        vkBindBufferMemory(vkDevice, buffer, bufferMemory, 0);
    }

    void createCommandBuffer() {

        renderCommandBuffers.resize(13);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;


        allocInfo.commandBufferCount = (uint32_t)renderCommandBuffers.size();

        if (vkAllocateCommandBuffers(vkDevice, &allocInfo, renderCommandBuffers.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }
public:
    VkCommandBuffer beginSingleTimeCommands() {
        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandPool = commandPool;
        allocInfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(vkDevice, &allocInfo, &commandBuffer);

        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer) {
        vkEndCommandBuffer(commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(graphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(graphicsQueue);

        vkFreeCommandBuffers(vkDevice, commandPool, 1, &commandBuffer);
    }
private:

    void createCommandPoll() {

        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if (vkCreateCommandPool(vkDevice, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
            throw std::runtime_error("failed to create command pool!");
        }
    }

    void createFramebuffers() {

        swapChainFramebuffers.resize(swapChainImageViews.size());




        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<VkImageView, 2> attachments = {
               swapChainImageViews[i],
                 depthImageView
            };


            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(vkDevice, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createRenderPass() {

        { VkRenderPassCreateInfo createInfo{};


        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;


        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcAccessMask = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::vector<VkAttachmentDescription> desc{ colorAttachment, depthAttachment };

        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = 2;
        renderPassInfo.pAttachments = desc.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;

        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;



        if (vkCreateRenderPass(vkDevice, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) {
            throw std::runtime_error("failed to create render pass!");
        }
        }

        {


            VkAttachmentDescription depthAttachment{};
            depthAttachment.format = findDepthFormat();
            depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentDescription colorAttachment{};
            colorAttachment.format = swapChainImageFormat;
            colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
            colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

            VkAttachmentReference depthAttachmentRef{};
            depthAttachmentRef.attachment = 1;
            depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

            VkAttachmentReference colorAttachmentRef{};
            colorAttachmentRef.attachment = 0;
            colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &colorAttachmentRef;
            subpass.pDepthStencilAttachment = &depthAttachmentRef;


            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            dependency.srcAccessMask = 0;
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            std::vector<VkAttachmentDescription> desc{ colorAttachment, depthAttachment };

            VkRenderPassCreateInfo renderPassInfo2{};
            renderPassInfo2.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            renderPassInfo2.attachmentCount = 2;
            renderPassInfo2.pAttachments = desc.data();
            renderPassInfo2.subpassCount = 1;
            renderPassInfo2.pSubpasses = &subpass;

            renderPassInfo2.dependencyCount = 1;
            renderPassInfo2.pDependencies = &dependency;


            if (vkCreateRenderPass(vkDevice, &renderPassInfo2, nullptr, &renderPassLimpador) != VK_SUCCESS) {
                throw std::runtime_error("failed to create render pass!");
            }



        }
    }

    void createGraphicsPipeline() {
        auto verShaderCode = lerArquivos("shaders/vert.spv");
        auto fragShaderCode = lerArquivos("shaders/frag.spv");

        if (!verShaderCode.empty()) {
            std::cout << "arquivos carregados" << std::endl;
        }
        else {
            std::cout << "arquivos nao carregados" << std::endl;
        }

        VkShaderModule verShader = createShaderModule(verShaderCode);
        VkShaderModule fragShader = createShaderModule(fragShaderCode);


        VkPipelineShaderStageCreateInfo vpssi{};
        vpssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vpssi.module = verShader;
        vpssi.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vpssi.pName = "main";


        VkPipelineShaderStageCreateInfo fpssi{};
        fpssi.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fpssi.module = fragShader;
        fpssi.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fpssi.pName = "main";


        std::vector<VkDynamicState> dynamicStatesView = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicStateInfoV{};
        dynamicStateInfoV.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateInfoV.dynamicStateCount = static_cast<uint32_t>(dynamicStatesView.size());
        dynamicStateInfoV.pDynamicStates = dynamicStatesView.data();




        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};

        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();



        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.minDepthBounds = 0.0f; // Optional
        depthStencil.maxDepthBounds = 1.0f; // Optional
        depthStencil.stencilTestEnable = VK_FALSE;
        depthStencil.front = {}; // Optional
        depthStencil.back = {}; // Optional

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        std::vector<VkDynamicState> dynamicStates = {
    VK_DYNAMIC_STATE_VIEWPORT,
    VK_DYNAMIC_STATE_SCISSOR
        };

        VkPipelineDynamicStateCreateInfo dynamicStateInfo{};
        dynamicStateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicStateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicStateInfo.pDynamicStates = dynamicStates.data();

        std::vector<VkPipelineDynamicStateCreateInfo> dynamics{};
        dynamics.push_back(dynamicStateInfo);
        dynamics.push_back(dynamicStateInfoV);

        VkPipelineViewportStateCreateInfo viewportState{};

        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;


        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional







        VkPipelineShaderStageCreateInfo pssis[] = { vpssi,fpssi };




        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = pssis;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;

        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = dynamics.data();//posivel erro

        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPassLimpador;
        pipelineInfo.subpass = 0;

        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional



        if (vkCreateGraphicsPipelines(vkDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS) {
            throw std::runtime_error("failed to create graphics pipeline! possivel erro aqui linha 328");
        }









        vkDestroyShaderModule(vkDevice, fragShader, nullptr);
        vkDestroyShaderModule(vkDevice, verShader, nullptr);

    }

    VkShaderModule createShaderModule(const std::vector<char>& code) {

        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(vkDevice, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("falha ao criar o modulo do sombreador");
        }

        return shaderModule;
    }

    static std::vector<char> lerArquivos(const std::string& filename) {

        std::ifstream file(filename, std::ios::ate | std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("failed to open file!");

        }
        size_t tamBuffer = (size_t)file.tellg();
        std::vector<char>buffer;
        buffer.resize(tamBuffer);

        file.seekg(0);
        file.read(buffer.data(), tamBuffer);

        file.close();
        return buffer;
    }

    void createImageViews() {
        swapChainImageViews.resize(swapChainImages.size());

        for (uint32_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);


        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;

        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;

        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(vkDevice, &createInfo, nullptr, &swapChain) != VK_SUCCESS) {
            throw std::runtime_error("failed to create swap chain!");
        }


        vkGetSwapchainImagesKHR(vkDevice, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(vkDevice, swapChain, &imageCount, swapChainImages.data());


        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;

    }

    void createLogicalDevice() {

        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);

        std::vector<VkDeviceQueueCreateInfo>  queueInfos;
        std::set<uint32_t>  uniqueQueueFamilies{ indices.graphicsFamily.value(),indices.presentFamily.value() };
        float queuePriorit = 1.0f;

        for (uint32_t queueFamily : uniqueQueueFamilies) {
            VkDeviceQueueCreateInfo queueInf{};
            queueInf.queueFamilyIndex = queueFamily;
            queueInf.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueInf.queueCount = 1;
            queueInf.pQueuePriorities = &queuePriorit;
            queueInfos.push_back(queueInf);
        }
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.sparseBinding = VK_TRUE;
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        deviceFeatures.sparseResidencyAliased = VK_TRUE;


        VkDeviceCreateInfo CreateInfo = {};
        CreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        CreateInfo.pQueueCreateInfos = queueInfos.data();
        CreateInfo.pEnabledFeatures = &deviceFeatures;
        CreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
        CreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        CreateInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) {
            CreateInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            CreateInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            CreateInfo.enabledLayerCount = 0;
        }

        if (physicalDevice) { std::cout << "nao vazio F" << std::endl; }
        if (vkDevice) {
            std::cout << "nao vazio d" << std::endl;
        }


        if (vkCreateDevice(physicalDevice, &CreateInfo, nullptr, &vkDevice) == VK_SUCCESS) {}
        else { throw std::runtime_error("falha a o criar dispositivo logico"); }

        vkGetDeviceQueue(vkDevice, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(vkDevice, indices.presentFamily.value(), 0, &presentQueue);



    }


    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
        QueueFamilyIndices indices = {};
        // Logic to find queue family indices to populate struct with

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

        int i = 0;

        for (const auto& queueFamily : queueFamilies) {
            if (queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                indices.graphicsFamily = i;
                std::cout << "suport a desenho na fila" << i << std::endl;
            }
            VkBool32 presentSupport = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);

            if (presentSupport) {
                indices.presentFamily = i;
                std::cout << "suport a apresentacao na fila" << i << std::endl;
            }

            i++;
        }

        return indices;
    }


    void pickPhysicalDevice() {




        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

        if (deviceCount == 0) {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());


        // Use an ordered map to automatically sort candidates by increasing score
        std::multimap<int, VkPhysicalDevice> candidates;

        for (const auto& device : devices) {
            int score = rateDeviceSuitability(device);
            candidates.insert(std::make_pair(score, device));
        }

        // Check if the best candidate is suitable at all
        if (candidates.rbegin()->first > 0) {
            physicalDevice = candidates.rbegin()->second;
        }
        else {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    int rateDeviceSuitability(VkPhysicalDevice device) {
        vkGetPhysicalDeviceProperties(device, &deviceProperties);

        vkGetPhysicalDeviceFeatures(device, &deviceFeatures);

        int score = 0;

        // Discrete GPUs have a significant performance advantage
        std::cout << deviceProperties.deviceType << std::endl;
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {

            score += 1000;
        }

        // Maximum possible size of textures affects graphics quality
        score += deviceProperties.limits.maxImageDimension2D;

        // Application can't function without geometry shaders
        if (!deviceFeatures.geometryShader) {
            return 0;
        }

        std::cout << "score:" << score << std::endl;

        return score;
    }

    bool isDeviceSuitable(VkPhysicalDevice device) {
        QueueFamilyIndices indices = findQueueFamilies(device);

        bool extensionsSupported = checkDeviceExtensionSupport(device);


        bool swapChainAdequate = false;

        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            std::cout << "max imagem count" << swapChainSupport.capabilities.maxImageCount << std::endl;
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }
        return indices.isComplete() && extensionsSupported && swapChainAdequate;
    }

    void inline createInstance() {


        if (enableValidationLayers && !checkValidationLayerSupport()) {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;
        appInfo.pNext = nullptr;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;

        if (enableValidationLayers) {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else {
            createInfo.enabledLayerCount = 0;
        }


        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cout << "falha ao criar a INSTANCIA VULKAN";
        }
        else {

        }
    }

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
        glfwSetKeyCallback(window, key_callback);




    }

    static void key_callback(GLFWwindow* janela, int  chave, int código, int ação, int mods)
    {
        if (chave == GLFW_KEY_W && ação == GLFW_REPEAT) {
            move += 1;
        }

        if (chave == GLFW_KEY_S && ação == GLFW_REPEAT) {
            move -= 1;
        }

        if (chave == GLFW_KEY_D && ação == GLFW_REPEAT) {
            moveL -= 1;
        }

        if (chave == GLFW_KEY_A && ação == GLFW_REPEAT) {
            moveL += 1;
        }

        if (chave == GLFW_KEY_Q && ação == GLFW_REPEAT) {
            rotate += 1;
        }

        if (chave == GLFW_KEY_E && ação == GLFW_REPEAT) {
            rotate -= 1;
        }

        if (chave == GLFW_KEY_RIGHT && ação == GLFW_RELEASE) {
            objetoFoco(1);
        }

        if (chave == GLFW_KEY_LEFT && ação == GLFW_RELEASE) {
            objetoFoco(-1);
        }

    }

    static void framebufferResizeCallback(GLFWwindow* window, int width, int height) {
        auto app = reinterpret_cast<vulkan*>(glfwGetWindowUserPointer(window));
        app->framebufferResized = true;
    }

    void mainLoop(VkCommandBuffer func(vulkan& app, int imageindex)) {



        bool open;
        float col;
        std::vector<Objeto> objetos;
        objetos.resize(13);

        for (int i = 0; i < objetos.size(); i++) {
            // objetos[i].loadModel(MODEL_PATH.c_str());
            objetos[i].setBuffers();
            criarObjeto(TEXTURE_PATH.c_str(), objetos[i]);

        }

        createDescriptorSets(objetos);





        using framerate = std::chrono::duration<int, std::ratio<1, 24>>;
        auto prev = std::chrono::system_clock::now();
        auto next = prev + framerate{ 1 };
        int N = 0;
        std::chrono::system_clock::duration sum{ 0 };


        while (!glfwWindowShouldClose(window)) {



            // Create a window called "My First Tool", with a menu bar
            glfwPollEvents();
            ::sleep_until(next);
            next += framerate{ 1 };

            int image = drawFrame(objetos, func);

            auto now = std::chrono::system_clock::now();
            sum += now - prev;
            ++N;
            //   std::cout << "This frame: " << std::chrono::round< std::chrono::milliseconds>(now - prev)
             //      << "  average: " << std::chrono::round< std::chrono::milliseconds>(sum / N) << '\n';
            prev = now;


            //metodo antigo de atualizar os verticies
          /*   if (vertices[0].pos.x<1.0f) {
                vertices[0].pos = vertices[0].pos + glm::vec2(0.001f, 0.001f);
                vertices[1].pos = vertices[1].pos + glm::vec2(0.001f, 0.001f);
                vertices[2].pos = vertices[2].pos + glm::vec2(0.001f, 0.001f);

                vertices[0].color = vertices[0].color + glm::vec3(0.01f, 0, 0);
                vertices[1].color = vertices[1].color + glm::vec3(0.001f, 0.01f, 0);
                vertices[2].color = vertices[2].color + glm::vec3(0.001f, 0, 0.01f);
             }
             else {

                 vertices[0].pos = vertices[0].pos + glm::vec2(-2.501f, -2.501f);
                 vertices[1].pos = vertices[1].pos + glm::vec2(-2.501f,- 2.501f);
                 vertices[2].pos = vertices[2].pos + glm::vec2(-2.501f,- 2.501f);

                 vertices[0].color = vertices[0].color + glm::vec3(-0.01f, 0, 0);
                 vertices[1].color = vertices[1].color + glm::vec3(0.001f,- 0.01f, 0);
                 vertices[2].color = vertices[2].color + glm::vec3(0.001f, 0, -0.01f);
      }*/

      //   vkMapMemory(vkDevice, intermem, 0, vertices.size()*sizeof(Vertex), 0, &vertexdata);
       //  memcpy(vertexdata, vertices.data(), vertices.size() * sizeof(Vertex));
       //    vkUnmapMemory(vkDevice, intermem);
       //    copyBuffer(bufferImtermediario,vertexBuffer,(VkDeviceSize)sizeof(Vertex)*vertices.size());



        }

        vkDeviceWaitIdle(vkDevice);
    }

    int drawFrame(std::vector<Objeto>& obj, VkCommandBuffer func(vulkan& app, int img)) {



        vkWaitForFences(vkDevice, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        vkResetFences(vkDevice, 1, &inFlightFences[currentFrame]);


        updateUboBuffer(obj[objetoFoco(0)], objetoFoco(0));


        uint32_t imageIndex = 0;

        std::vector<uint32_t> indexs;
        indexs.push_back(imageIndex);

        VkResult result = vkAcquireNextImageKHR(vkDevice, swapChain, 100000000, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, indexs.data());

        // comandos do Gui

        VkCommandBuffer cmdGui = func(*this, imageIndex);
        //end

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }

        else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("failed to acquire swap chain image!");
        }



        std::cout << "frame atual:" << currentFrame << "cerca:" << inFlightFences[currentFrame] << std::endl;


        recordCommandBuffer(renderCommandBuffers, indexs[0], obj);

        std::cout << "passo do record:" << std::endl;



        VkSemaphore semaphores[] = { imageAvailableSemaphores[currentFrame] };

        VkSemaphore semaphoresR[] = { renderFinishedSemaphores[currentFrame] };
        std::vector<VkPipelineStageFlags> stages = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        stages.resize(13);

        for (size_t i = 0; i < stages.size(); i++)
        {
            stages[i] = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

        }


        std::vector< VkSubmitInfo> subinf{};
        subinf.resize(2);

        subinf[0].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;





        subinf[0].commandBufferCount = 1;
        subinf[0].pCommandBuffers = &renderCommandBuffers[renderCommandBuffers.size() - 1];
        subinf[0].pWaitSemaphores = semaphores;
        subinf[0].waitSemaphoreCount = 1;
        subinf[0].signalSemaphoreCount = sinalizadorDeLipezaFrameBuffer.size();
        subinf[0].pSignalSemaphores = sinalizadorDeLipezaFrameBuffer.data();
        subinf[0].pWaitDstStageMask = stages.data();

        if (vkQueueSubmit(graphicsQueue, 1, &subinf[0], inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }







        subinf[0].pCommandBuffers = renderCommandBuffers.data();
        subinf[0].commandBufferCount = renderCommandBuffers.size() - 1;
        subinf[0].pWaitSemaphores = sinalizadorDeLipezaFrameBuffer.data();
        subinf[0].waitSemaphoreCount = sinalizadorDeLipezaFrameBuffer.size();
        subinf[0].signalSemaphoreCount = 1;
        subinf[0].pSignalSemaphores = &renderGui;
        subinf[0].pWaitDstStageMask = &stages[0];

        subinf[1].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        subinf[1].commandBufferCount = 1;
        subinf[1].pCommandBuffers = &cmdGui;
        subinf[1].pWaitSemaphores = &renderGui;
        subinf[1].waitSemaphoreCount = 1;
        subinf[1].signalSemaphoreCount = 1;
        subinf[1].pSignalSemaphores = semaphoresR;
        subinf[1].pWaitDstStageMask = &stages[0];

        if (vkQueueSubmit(graphicsQueue, 2, subinf.data(), nullptr) != VK_SUCCESS) {
            throw std::runtime_error("failed to submit draw command buffer!");
        }
        std::cout << "cmd: " << 1 << " sumbmetido" << std::endl;



        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = semaphoresR;

        VkSwapchainKHR swapChains[] = { swapChain };

        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        presentInfo.pResults = nullptr; // Optional


        result = vkQueuePresentKHR(graphicsQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebufferResized) {
            framebufferResized = false;
            recreateSwapChain();
        }
        else if (result != VK_SUCCESS) {
            throw std::runtime_error("failed to present swap chain image!");
        }




        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        vkFreeCommandBuffers(vkDevice, commandPool, 1, &cmdGui);

        return imageIndex;
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0) {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(vkDevice);
        cleanupSwapChain();
        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
    }

    void cleanupSwapChain() {
        for (size_t i = 0; i < swapChainFramebuffers.size(); i++) {
            vkDestroyFramebuffer(vkDevice, swapChainFramebuffers[i], nullptr);
        }

        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            vkDestroyImageView(vkDevice, swapChainImageViews[i], nullptr);
        }

        vkDestroySwapchainKHR(vkDevice, swapChain, nullptr);
    }

    void cleanup()
    {


        cleanupSwapChain();
        vkQueueWaitIdle(graphicsQueue);
        vkDeviceWaitIdle(vkDevice);

        vkDestroySampler(vkDevice, textureSampler, nullptr);
        vkDestroyImageView(vkDevice, textureImageView, nullptr);


        vkDestroyImageView(vkDevice, depthImageView, nullptr);
        vkDestroyImage(vkDevice, depthImage, nullptr);
        vkFreeMemory(vkDevice, depthImageMemory, nullptr);

        vkDestroyImage(vkDevice, image, nullptr);
        vkFreeMemory(vkDevice, imagemem, nullptr);

        vkDestroyDescriptorPool(vkDevice, descriptorPool, nullptr);

        vkDestroyDescriptorSetLayout(vkDevice, setLayout, nullptr);

        vkDestroyBuffer(vkDevice, indexBuffer, nullptr);
        vkFreeMemory(vkDevice, indexmem, nullptr);

        vkUnmapMemory(vkDevice, instmem);
        vkDestroyBuffer(vkDevice, Instancia, nullptr);
        vkFreeMemory(vkDevice, instmem, nullptr);

        vkUnmapMemory(vkDevice, matrixmem);
        vkDestroyBuffer(vkDevice, ubomatrix, nullptr);
        vkFreeMemory(vkDevice, matrixmem, nullptr);

        vkQueueWaitIdle(graphicsQueue);
        vkDeviceWaitIdle(vkDevice);
        vkDestroyBuffer(vkDevice, bufferImtermediario, nullptr);
        vkFreeMemory(vkDevice, intermem, nullptr);

        vkDestroyBuffer(vkDevice, vertexBuffer, nullptr);
        vkFreeMemory(vkDevice, vertexBufferMemory, nullptr);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            vkDestroySemaphore(vkDevice, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(vkDevice, imageAvailableSemaphores[i], nullptr);
            vkDestroySemaphore(vkDevice, sinalizadorDeLipezaFrameBuffer[i], nullptr);
            vkDestroyFence(vkDevice, inFlightFences[i], nullptr);
        }
        vkDestroyCommandPool(vkDevice, commandPool, nullptr);
        vkDestroyPipeline(vkDevice, graphicsPipeline, nullptr);
        vkDestroyPipelineLayout(vkDevice, pipelineLayout, nullptr);
        vkDestroyRenderPass(vkDevice, renderPass, nullptr);
        vkDestroyRenderPass(vkDevice, renderPassLimpador, nullptr);

        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(vkDevice, nullptr);
        vkDestroyInstance(instance, nullptr);


        glfwDestroyWindow(window);
        glfwTerminate();

    }

    bool checkValidationLayerSupport() {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) {
                if (strcmp(layerName, layerProperties.layerName) == 0) {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) {
                return false;
            }
        }

        return true;
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {

        SwapChainSupportDetails details;

        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);


        uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }

        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }

        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {

        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                return availableFormat;
            }
        }

        return availableFormats[0];

    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR) {
                return availablePresentMode;
            }
        }

        return VK_PRESENT_MODE_FIFO_KHR;
    }


    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
            return capabilities.currentExtent;
        }
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

            return actualExtent;
        }

    }

public:
    void criarObjeto(const char* textura, Objeto& obj) {

        createTextureImage(textura, obj);
        createVertexBuffer(obj);
        createIndexBuffer(obj);
        createUboBuffer(obj);
        createInstanciaBuffer(obj);


    }

    //funcoes para os descritores
    void createDescriptorSetLayout() {


        std::array<VkDescriptorSetLayoutBinding, 3> bindings{};


        VkDescriptorSetLayoutBinding lbind{};
        lbind.binding = 0;
        lbind.descriptorCount = 1;
        lbind.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lbind.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
        bindings[0] = lbind;

        VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        bindings[1] = samplerLayoutBinding;

        VkDescriptorSetLayoutBinding lbind2{};
        lbind2.binding = 2;
        lbind2.descriptorCount = 1;
        lbind2.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lbind2.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        bindings[2] = lbind2;



        //std::vector<VkDescriptorSetLayoutBinding> bindings = { lbind, samplerLayoutBinding };

        VkDescriptorSetLayoutCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        createInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        createInfo.pBindings = bindings.data();

        if (vkCreateDescriptorSetLayout(vkDevice, &createInfo, nullptr, &setLayout) != VK_SUCCESS)
            throw std::runtime_error("falha ao criar o descritor");

        VkPipelineLayoutCreateInfo plinfo{};
        plinfo.pSetLayouts = &setLayout;
        plinfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        plinfo.setLayoutCount = 1;
        if (vkCreatePipelineLayout(vkDevice, &plinfo, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("falha ao criar o pipelineLayout");

    }

    void createDescriptorSets(std::vector<Objeto>& obj) {






        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, setLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();


        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(vkDevice, &allocInfo, descriptorSets.data()) != VK_SUCCESS) {
            throw std::runtime_error("failed to allocate descriptor sets!");


        }



        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {






            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = Instancia;
            bufferInfo.offset = 0;
            bufferInfo.range = (VkDeviceSize)sizeof(instancia);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = obj[0].textureImageView;
            imageInfo.sampler = obj[0].textureSampler;



            std::vector< VkWriteDescriptorSet> descriptorWrite{};
            descriptorWrite.resize(3);

            descriptorWrite[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[0].dstSet = descriptorSets[i];
            descriptorWrite[0].dstBinding = 0;
            descriptorWrite[0].dstArrayElement = 0;
            descriptorWrite[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite[0].descriptorCount = 1;
            descriptorWrite[0].pBufferInfo = &bufferInfo;
            descriptorWrite[0].pImageInfo = nullptr; // Optional
            descriptorWrite[0].pTexelBufferView = nullptr; // Optional

            descriptorWrite[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[1].dstSet = descriptorSets[i];
            descriptorWrite[1].dstBinding = 1;
            descriptorWrite[1].dstArrayElement = 0;
            descriptorWrite[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrite[1].descriptorCount = 1;
            descriptorWrite[1].pImageInfo = &imageInfo; // Optional
            descriptorWrite[1].pTexelBufferView = nullptr; // Optional

            std::cout << "problema aqui?" << std::endl;
            VkDescriptorBufferInfo uboInfo{};
            uboInfo.buffer = ubomatrix;
            uboInfo.offset = 0;
            uboInfo.range = (VkDeviceSize)sizeof(UniformBufferObject);


            descriptorWrite[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrite[2].dstSet = descriptorSets[i];
            descriptorWrite[2].dstBinding = 2;
            descriptorWrite[2].dstArrayElement = 0;
            descriptorWrite[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrite[2].descriptorCount = 1;
            descriptorWrite[2].pBufferInfo = &uboInfo;

            vkMapMemory(vkDevice, matrixmem, 0, (VkDeviceSize)sizeof(UniformBufferObject), 0, &pmatrix);



            vkUpdateDescriptorSets(vkDevice, (uint32_t)descriptorWrite.size(), descriptorWrite.data(), 0, nullptr);
        }


    }

    //funcoes para atualizar o objeto
    void updateUboBuffer(Objeto& obj, float instanci) {



        auto tempoAtual = std::chrono::high_resolution_clock::now();

        time = std::chrono::duration<float, std::chrono::seconds::period>(tempoAtual - inicio).count();


        UniformBufferObject ubo{};
        // ubo.model = glm::rotate(glm::mat4(1.0f), glm::radians(rotate * 15.0f), glm::vec3(0.0f, 1.0f, 0.0f));


        ubo.model += glm::translate(glm::mat4(1.0f), glm::vec3((float)moveL, move + instanci * 5, 1.0f));

        ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 0.5f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.view += glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -10.0f));

        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 100.0f);

        // ubo.proj[1][1] *= -1;

        instancia inst{};
        inst.obj = instanci;


        memcpy(obj.pUboMatrix, &ubo, sizeof(ubo));
        memcpy(obj.pinstanciaBuffer, &inst, sizeof(inst));


    }
};


