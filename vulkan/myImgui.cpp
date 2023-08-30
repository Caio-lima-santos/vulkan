
#include "server2.cpp"
#include <omp.h>
#define GLFW_INCLUDE_VULKAN

#include <GLFW/glfw3.h>
#include <string>

using namespace ImGui;

char buff[DEFAULT_BUFLEN];
char* ptr_buff = &buff[0];
int bits;

aiString mensagemAtual;

char buffEntre[DEFAULT_BUFLEN];

server Server;

const char* nome = "cliente";

std::vector< aiString>mensagens2;




class GUI {



    struct input {
        const char* nome;
        float* valor;
    };

    struct Janela {

        const char* nome;
        int altura;
        int largura;
        bool aberto;


    };


    std::vector<void*>alocs;
    std::vector<input*> inputs;





public:

    void initImGui(vulkan& app) {
        mensagens2.resize(120);

        mensagens2[0].Set("Bate-papo");

        ZeroMemory(buffEntre,DEFAULT_BUFLEN);
       Server.initS();



        //  bits = initSint();



        VkDescriptorPool descpoll{};

        VkDescriptorPoolSize pool_sizes[] =
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };


        {


            VkDescriptorPoolCreateInfo descinfo{};

            descinfo.flags = 0;
            descinfo.maxSets = 50;
            descinfo.pPoolSizes = pool_sizes;
            descinfo.poolSizeCount = 11;
            descinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;

            if (vkCreateDescriptorPool(app.vkDevice, &descinfo, nullptr, &descpoll) != VK_SUCCESS)
                throw std::runtime_error("erro no desc poll");




        }

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        ImGui_ImplGlfw_InitForVulkan(app.window, true);

        ImGui_ImplVulkan_InitInfo info{};
        info.Allocator = nullptr;
        info.CheckVkResultFn = nullptr;
        info.ColorAttachmentFormat = app.swapChainImageFormat;
        info.DescriptorPool = descpoll;
        info.Device = app.vkDevice;
        info.ImageCount = 2;
        info.Instance = app.instance;
        info.MinImageCount = 2;
        info.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
        info.PhysicalDevice = app.physicalDevice;
        info.PipelineCache = nullptr;
        info.Queue = app.graphicsQueue;
        info.QueueFamily = 0;
        info.Subpass = 0;
        info.UseDynamicRendering = false;


        ImGui_ImplVulkan_Init(&info, app.renderPass);




        VkCommandBuffer command_buffer = app.beginSingleTimeCommands();
        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        app.endSingleTimeCommands(command_buffer);



    }

    input* pegarPorNome(const char* nome) {

        for (int i = 0; i < inputs.size(); i++) {
            if (strcmp(nome, inputs[i]->nome)) {
                return inputs[i];
            }
        }

    }

    void receberFloat(const char* nome, float* flutuante) {

        input* in = new input();
        alocs.push_back(in);
        inputs.push_back(in);

        in->nome = nome;
        in->valor = flutuante;

        InputFloat(nome, flutuante);

    }

    void clear() {
        for (int i = 0; i < alocs.size(); i++) {
            free(alocs[i]);
        };

    }


};


bool open = true;
const char* texto = "texto";
ImGuiID windosID;
ImGuiWindow* window;
bool atualizar = false;
bool conectar = false;


VkCommandBuffer imGuiL(vulkan& app, int imageindex) {

    VkCommandBuffer cmd = app.beginSingleTimeCommands();
    char temp[DEFAULT_BUFLEN];



#pragma omp parallel 
    {




#pragma omp single nowait
        {
            
            
            glfwPollEvents();

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            Begin("Debug do Servidor", &open);

            windosID = ImGui::GetID("janelinha");
            window = ImGui::FindWindowByID(windosID);

            BeginChild("scrolling");

            InputText("digite uma mensagem", buffEntre, DEFAULT_BUFLEN);
            if (Button("enviar")) {
                if (strlen(buffEntre) > 0) {
                    //retira!!
                    PAUSE_TREAD= true;
                    filaEO.inserirE(buffEntre);
                }
            }
            



            for (int i = 0; i < mensagensS.size(); i++) {
                Text(mensagensS[i].C_Str());
            }

          



            EndChild();

            if (Button("connectar")) {
                conectar = true;



            }
            if (Button("Desconectar")) {

                Server.desconectarTodosEencerar();

            }


            End();




            Begin("janelinha##", &open, ImGuiWindowFlags_MenuBar);
            //ImGui::BeginMenu("menu");
            ImGui::BeginMenuBar();
         
            ImGui::MenuItem("item", texto, &open, true);

            ImGui::EndMenuBar();

            float valor = 0.0f;
            ImGui::InputFloat("valor", &valor);
            Text(std::to_string(0).c_str());


            Text("bate-papo");
            BeginChild("Scrolling");

          


            for (int i = 0; i < filaEO.filaDeRecebimento.size(); i++) {
             
                Text(filaEO.filaDeRecebimento[i].C_Str());

            }
            InputText("escreva a mensagen", mensagemAtual.data, MAXLEN);
            if (Button("enviar")) {
                PAUSE_TREAD = true;
                filaEO.inserirE(mensagemAtual.data);

            }
            EndChild();

          

            Text(std::to_string(valor).c_str());

            //  ImGui::EndMenu();
            ImGui::End();









            ImGui::ShowDemoWindow();
            ImGui::EndFrame();
            ImGui::Render();










            VkRenderPassBeginInfo renderPassInfo{};
            renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            renderPassInfo.renderPass = app.renderPass;
            renderPassInfo.framebuffer = app.swapChainFramebuffers[imageindex];
            renderPassInfo.renderArea.offset = { 0, 0 };
            renderPassInfo.renderArea.extent = app.swapChainExtent;

            std::array<VkClearValue, 2> clearValues{};
            clearValues[0].color = { {0.0f, 0.0f, 0.0f, 1.0f} };
            clearValues[1].depthStencil = { 1.0f, 0 };


            renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
            renderPassInfo.pClearValues = clearValues.data();



            vkCmdBeginRenderPass(cmd, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);

            vkCmdEndRenderPass(cmd);


            vkEndCommandBuffer(cmd);

        }

#pragma omp single nowait

        {





        }          }






    return cmd;


}


void end() {




}

void body() {

}
