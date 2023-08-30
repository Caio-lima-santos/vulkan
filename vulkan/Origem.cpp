


#include "myImGui.cpp"









int main() {
    vulkan app;
    GUI Gui;





    app.init();
    Gui.initImGui(app);


    omp_set_num_threads(2);
   


#pragma omp parallel shared(connectado,conectLoop,mensagensS,filaEO)       
    {


        debugServer(std::to_string(omp_get_num_threads()));


#pragma omp single nowait
        {
            //   app.loop(imGui);
            app.loop(imGuiL);
           


        }

#pragma omp single nowait
        {

            if(!Server.conectarEXTERN("cliente"))
                Server.conectar("cliente");

           

        }







    }



    return 0;

}

