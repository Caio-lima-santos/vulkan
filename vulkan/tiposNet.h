

#define NET_CALLBACK(int,data)
#define BUFF_LARGURA 512
#define CB_TAMANHO 4
#define  mk_par_CONN( x,y) std::map<char*,data_action>::value_type(x,y)


#include <assimp/types.h>
#include <string>
#include <map>
#include <vector>
struct Netfila{

    bool em_uso_para_escrita = false;
    bool em_uso_para_leitura = false;
    std::vector<aiString>filaDeEnvio;
    std::vector<aiString>filaDeRecebimento;

    Netfila() {


    }

    bool inserirR(char* data) {

        if (em_uso_para_escrita == false) {
            em_uso_para_escrita = true;
            this->filaDeRecebimento.push_back(aiString(data));
            em_uso_para_escrita = false;
            return true;
        }
        else {
            return false;
        }

    }

    aiString retiraR() {
        aiString temp;
        em_uso_para_leitura = true;
        temp = this->filaDeRecebimento[this->filaDeRecebimento.size() - 1];
        filaDeRecebimento.pop_back();
        em_uso_para_leitura = false;
        return temp;
    }

    aiString retiraE() {
        aiString temp;
        em_uso_para_leitura = true;
        temp = this->filaDeEnvio[this->filaDeEnvio.size() - 1];
        this->filaDeEnvio.pop_back();
        em_uso_para_leitura = false;
        return temp;
    }

    bool inserirE(char* data) {

        if (em_uso_para_escrita == false) {
            em_uso_para_escrita = true;
            this->filaDeEnvio.push_back(aiString(data));
            em_uso_para_escrita = false;
            return true;
        }
        else {
            return false;
        }

    }
};




struct CABEÇALHO {

   char começo[4];
    int tamanho;

};

struct data_action {

    CABEÇALHO cb;
    void* action;

};


struct CONN_CL {

        std::map<char*, data_action> chamadas;


      Netfila* filaEO;
     CONN_CL(Netfila *netfila) {
        this->filaEO = netfila;
    }

    void chamar_quando_receber(char *id,data_action ac) {
       
        chamadas.insert(mk_par_CONN(id,ac));

    }

    void chamar_quando_enviar(void funcao(aiString& data),CABEÇALHO cb) {
       
    }

    void processar_eventos() {
        char data_temp[BUFF_LARGURA];
        CABEÇALHO cb_temp{};
        //CONTINUAR 

        while (!filaEO->filaDeRecebimento.empty()) {

            strcpy(data_temp, filaEO->filaDeRecebimento[filaEO->filaDeRecebimento.size()].C_Str());

            for (int i = 0; i < CB_TAMANHO; i++) {


            }


        }

    }
    
    
    




  };

struct CONN_SV {
    Netfila* filaEO;


};
