#undef UNICODE

#define WIN32_LEAN_AND_MEAN
#include "myVulkan.h"
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <string>
#include <map>
#include <omp.h>
#include "tiposNet.h"

// Need to link with Ws2_32.lib
#pragma comment (lib, "Ws2_32.lib")
#pragma comment (lib, "Mswsock.lib")
#pragma comment (lib, "AdvApi32.lib")

#define DEFAULT_BUFLEN 512
#define DEFAULT_PORT "27015"





typedef std::map<const char*, SOCKET>::value_type sock;

#define  mk_par( x,y) std::map<const char*, SOCKET>::value_type(x,y)

#define   debugServer(x) mensagensS.push_back(aiString(x));

std::vector<aiString>mensagensS;

std::vector<aiString>filadeEnvio;



Netfila filaEO{};
bool connectado = false;
bool conectLoop = false;
bool listenb = false;
bool PAUSE_TREAD=false;

class server {


    WSADATA wsaData;
    int iResult;
    const char* cli_name = "cliente";
    const char* sendbuf = "caio";

    std::map<const char*, SOCKET> clientes;



  

    SOCKET ListenSocket = INVALID_SOCKET;
    SOCKET ClientSocket = INVALID_SOCKET;
    SOCKET ConnectSocket = INVALID_SOCKET;

    struct addrinfo* result = NULL, * ptr = NULL;
    struct addrinfo hints;

    struct addrinfo* result2 = NULL, * ptr2 = NULL;
    struct addrinfo hints2;

    int iSendResult;
    char recvbuf[DEFAULT_BUFLEN];
    int recvbuflen = DEFAULT_BUFLEN;

    
       
    int bytes = 0;

    fd_set fd_r;
    fd_set fd_w;
   
  

    timeval tm;
   

    u_long mode = 0;

    char recvBuff[DEFAULT_BUFLEN];
    using framerate = std::chrono::duration<int, std::ratio<1, 10>>;
    using framerate2 = std::chrono::duration<int, std::ratio<1, 200>>;
   
    int N = 0;
    std::chrono::system_clock::duration sum{ 0 };






public:

    int __cdecl initS(const char* endereço = ("127.0.168.1"), int af = (AF_INET), int tipo = (SOCK_STREAM), int protocolo = (IPPROTO_TCP))
    {


        return true;

    }


    bool conectar(const char* nome) {



        // Initialize Winsock
        iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (iResult != 0) {
            printf("WSAStartup failed with error: %d\n", iResult);
            return 1;
        }
        ZeroMemory(&result, sizeof(result));
        ZeroMemory(&hints, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;
        hints.ai_flags = AI_PASSIVE;

        // Resolve the server address and port
        iResult = getaddrinfo(NULL, DEFAULT_PORT, &hints, &result);
        if (iResult != 0) {
            printf("getaddrinfo failed with error: %d\n", iResult);
            WSACleanup();
            return false;
        }

        // Create a SOCKET for the server to listen for client connections.

        ListenSocket = socket(result->ai_family, result->ai_socktype, result->ai_protocol);
        if (ListenSocket == INVALID_SOCKET) {
            printf("socket failed with error: %ld\n", WSAGetLastError());
            freeaddrinfo(result);
            WSACleanup();
            return false;
        }


        // Setup the TCP listening socket
        iResult = bind(ListenSocket, result->ai_addr, (int)result->ai_addrlen);
        if (iResult == SOCKET_ERROR) {
            printf("bind failed with error: %d\n", WSAGetLastError());
            freeaddrinfo(result);
            closesocket(ListenSocket);
            WSACleanup();
            return false;
        }

        freeaddrinfo(result);

        iResult = listen(ListenSocket, SOMAXCONN);
        if (iResult == SOCKET_ERROR) {
            printf("listen failed with error: %d\n", WSAGetLastError());
            closesocket(ListenSocket);
            WSACleanup();
            return false;
        }

        debugServer("modo escuta...")
            ClientSocket = accept(ListenSocket, NULL, NULL);
        if (ClientSocket == INVALID_SOCKET) {
            debugServer("falha na aceitação")
                printf("accept failed with error: %d\n", WSAGetLastError());
            closesocket(ListenSocket);
            WSACleanup();
            return false;
        }
     
        
        clientes.insert(mk_par(nome, ClientSocket));
        debugServer("sucesso na aceitacao")
            closesocket(ListenSocket);
        connectado = true;
        listenb = true;
        ConnectLoop();
        return true;

    }


    bool conectarEXTERN(const char* nome) {

      
        auto prev = std::chrono::system_clock::now();
        auto next = prev + framerate{ 1 };
       
        WSACleanup();
        WSADATA wsaData;
        SOCKET ConnectSocket = INVALID_SOCKET;
        struct addrinfo* result = NULL,
            * ptr = NULL,
            hints;

        char recvbuf[DEFAULT_BUFLEN];
        int iResult;
        int recvbuflen = DEFAULT_BUFLEN;

        // Validate the parameters


        // Initialize Winsock
        iResult = WSAStartup(MAKEWORD(2, 2), &wsaData);
        if (iResult != 0) {
            printf("WSAStartup failed with error: %d\n", iResult);
            return false;
        }

        ZeroMemory(&hints, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        // Resolve the server address and port
        iResult = getaddrinfo("127.0.0.1", DEFAULT_PORT, &hints, &result);
        if (iResult != 0) {
            debugServer("addr info falho")
                WSACleanup();
            return false;
        }

        // Attempt to connect to an address until one succeeds
        for (ptr = result; ptr != NULL; ptr = ptr->ai_next) {

            // Create a SOCKET for connecting to server
            ConnectSocket = socket(ptr->ai_family, ptr->ai_socktype,
                ptr->ai_protocol);
            if (ConnectSocket == INVALID_SOCKET) {
                debugServer("socket falho")
                    WSACleanup();
                return false;
            }

            // Connect to server.
            iResult = connect(ConnectSocket, ptr->ai_addr, (int)ptr->ai_addrlen);
            if (iResult == SOCKET_ERROR) {
                debugServer("conexao falhou")
                    closesocket(ConnectSocket);
                ConnectSocket = INVALID_SOCKET;
                return false;
            }
            break;
        }

        FD_ZERO(&fd_r);
        FD_ZERO(&fd_w);
        FD_SET(ClientSocket, &fd_r);
        FD_SET(ConnectSocket, &fd_r);

        FD_SET(ClientSocket, &fd_w);
        FD_SET(ConnectSocket, &fd_w);
        freeaddrinfo(result);

        if (ConnectSocket == INVALID_SOCKET) {
            debugServer("socket invalido")
                WSACleanup();
            return false;
        }

        // Send an initial buffer
        

    

        // Receive until the peer closes the connection


        // cleanup

        listenb = false;
        connectado = true;

        tm.tv_sec = 5;
        tm.tv_usec = 2;
        mode = 1;
        ioctlsocket(ConnectSocket, FIONBIO, (u_long*)&mode);

        while (connectado) {

            FD_ZERO(&fd_r);
            FD_ZERO(&fd_w);
            FD_SET(ConnectSocket, &fd_r);
            FD_SET(ConnectSocket, &fd_w);
           
            const char* b = "tesdasdasdas";

         
            mode = 1;
              //  if (FD_ISSET(ConnectSocket, &fd_w) && PAUSE_TREAD) {
            if (select(0, &fd_r, nullptr, nullptr, &tm) > 0) {

                if (FD_ISSET(ConnectSocket, &fd_r) && !PAUSE_TREAD) {
                    mode = 1;
                    debugServer("evento de recebimento")
                        ZeroMemory(recvBuff,sizeof(recvBuff));
                        iResult = recv(ConnectSocket, recvBuff, DEFAULT_BUFLEN, 0);
                    if (iResult > 0) {


                        {
                            std::vector<char>i;
                            int count = 0;

                            recvBuff[iResult] = NULL;
                            while (recvBuff[count] != NULL) {
                                i.push_back(recvBuff[count]);
                                count++;
                            }
                            i.push_back(recvBuff[count + 1]);
                            filaEO.inserirR(i.data());
                        }





                      debugServer("BYTES recebidos")
                          debugServer(std::to_string(iResult))
                           debugServer("mensagem:")
                         
                       debugServer(recvBuff)
                            PAUSE_TREAD = !PAUSE_TREAD;


                    }
                  
                }
            }
                 
                    if (filaEO.filaDeEnvio.size()>0 && PAUSE_TREAD) {
                        debugServer("eventos de envio")
                        aiString temp(filaEO.retiraE());
                        if (send(ConnectSocket,temp.C_Str(),temp.length,0) > 0)
                        debugServer("enviado");
                        //retirar!!
                      
                        PAUSE_TREAD = !PAUSE_TREAD;
                    }
                   
                        //    }
                     
                    //send else

               

            
         /*   else {
                debugServer("sem eventos de envio. fazendo send as cegas :/")
                if (filaEO.filaDeEnvio.size() > 0) {
                    aiString temp = filaEO.retiraE();
                    if (send(ConnectSocket, temp.C_Str(), temp.length, 0) > 0)
                        debugServer("enviado")
                    PAUSE_TREAD = !PAUSE_TREAD;
                }}*/

            

          
            ::sleep_until(next);
            next += framerate{ 1 };
            
        }



        conectLoop = false;
        debugServer("conexao encerada")



        return true;

    }


    void ConnectLoop() {
        conectLoop = true;
        int bytes = 0, iResult = 0;

        fd_set fd_r;
        fd_set fd_w;
        FD_ZERO(&fd_r);
        FD_ZERO(&fd_w);
        FD_SET(ClientSocket, &fd_r);
        FD_SET(ConnectSocket, &fd_r);

        FD_SET(ClientSocket, &fd_w);
        FD_SET(ConnectSocket, &fd_w);

        timeval tm;
        tm.tv_sec = 10;
        tm.tv_usec = 0;

        char recvBuff[DEFAULT_BUFLEN];
        using framerate = std::chrono::duration<int, std::ratio<1, 200>>;
        auto prev = std::chrono::system_clock::now();
        auto next = prev + framerate{ 1 };
        int N = 0;
        std::chrono::system_clock::duration sum{ 0 };




        if (listenb) {
            do {
                int Sn = 0;
                if (Sn = select(0, &fd_r, &fd_w, nullptr, NULL) > 0) {
                    // debugServer("select funciono HIV")
                    // debugServer(std::to_string(Sn))


                    if (FD_ISSET(ClientSocket, &fd_w) && !PAUSE_TREAD) {

                        debugServer("cliente esta enviando")
                            iResult = recv(ClientSocket, recvBuff, DEFAULT_BUFLEN, 0);
                        if (iResult > 0) {
                            std::vector<char>i;
                            int count = 0;
                            recvBuff[iResult] = (char)NULL;
                            while (recvBuff[count] != (char)NULL) {
                                i.push_back(recvBuff[count]);
                                count++;
                            }
                            i.push_back((char)NULL);
                            filaEO.inserirR(i.data());
                            debugServer("BYTES recebidos")
                                debugServer(std::to_string(iResult))
                                debugServer("Mensagen:")
                                debugServer(i.data())


                                recvBuff[iResult + 1] = (char)"\0";
                            PAUSE_TREAD = !PAUSE_TREAD;

                        }
                        else if (iResult == 0) {
                            debugServer("recv nao recebeu nada")
                        }
                        else {
                            debugServer("erro no revc")
                                break;
                        }

                    }if (FD_ISSET(ClientSocket, &fd_r)) {
                        debugServer("cliente esta recebendo")
                            send(ClientSocket, sendbuf, DEFAULT_BUFLEN, 0);
                        PAUSE_TREAD = !PAUSE_TREAD;

                    }
                }
                FD_ZERO(&fd_r);
                FD_ZERO(&fd_w);
                FD_SET(ClientSocket, &fd_r);
                FD_SET(ClientSocket, &fd_w);

                ::sleep_until(next);
                next += framerate{ 1 };

            } while (connectado);
        }
        else {

            do {
                int Sn = 0;
                if (Sn = select(NULL, &fd_r, &fd_w, nullptr, &tm) > 0) {
                    debugServer("select funciono HIV")
                        debugServer(std::to_string(Sn).c_str())
                }

                if (FD_ISSET(ConnectSocket, &fd_r)) {
                    iResult = recv(ConnectSocket, recvBuff, strlen(recvbuf), 0);
                    if (iResult > 0) {
                        debugServer("BYTES recebidos")
                            debugServer(std::to_string(iResult))
                            recvBuff[strlen(recvBuff)] = (char)"\0";

                        ZeroMemory(recvBuff, sizeof(recvBuff));
                    }
                    else if (iResult == 0)
                        printf("Connection closed\n");
                }
                if (FD_ISSET(ConnectSocket, &fd_w)) {

                    send(ConnectSocket, sendbuf, DEFAULT_BUFLEN, 0);

                }
                FD_ZERO(&fd_r);
                FD_ZERO(&fd_w);
                FD_SET(ConnectSocket, &fd_r);
                FD_SET(ConnectSocket, &fd_w);

                ::sleep_until(next);
                next += framerate{ 1 };

            } while (connectado);

        }



        conectLoop = false;
        debugServer("conexao encerada")



    }




    bool desconectarCli(const char* nome) {

        if (!clientes.empty())
            closesocket(clientes.at(nome));
        return true;


    }


    bool desconectarTodosEencerar() {
        connectado = false;

       

            closesocket(ListenSocket);
            closesocket(ConnectSocket);
        
        return WSACleanup();

    }

    //possivel err
    bool receberDe(const char* nome, char* data, int* tamanho = ((int*)0), UINT64 maxTenta = (20000)) {
        int tm = 0;
        int tentativas = 0;
        char inf[DEFAULT_BUFLEN] = "falha2";
        char* ptr_inf = &inf[0];

        while (tm <= 0 && tentativas < maxTenta) {

            tm = recv(clientes.at(nome), inf, DEFAULT_BUFLEN, 0);
            tentativas++;
        }

        if (tentativas >= maxTenta) {
            data = (char*)"falha ao receber";
            *tamanho = strlen(data);
            return false;
        }
        *tamanho = strlen(inf);
        inf[strlen(inf)] = '\0';
        strcpy(data, ptr_inf);
        return true;

    }

    bool enviarPara(const char* nome, char* data, size_t tamanho = ((size_t)0)) {

        send(clientes.at(nome), data, tamanho, 0);

        return true;

    }

};