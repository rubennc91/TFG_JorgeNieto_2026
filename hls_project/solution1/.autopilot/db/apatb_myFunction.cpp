#include <systemc>
#include <iostream>
#include <cstdlib>
#include <cstddef>
#include <stdint.h>
#include "SysCFileHandler.h"
#include "ap_int.h"
#include "ap_fixed.h"
#include <complex>
#include <stdbool.h>
#include "autopilot_cbe.h"
#include "hls_stream.h"
#include "hls_half.h"
#include "hls_signal_handler.h"

using namespace std;
using namespace sc_core;
using namespace sc_dt;

// wrapc file define:
#define AUTOTB_TVIN_x_ini_0 "../tv/cdatafile/c.myFunction.autotvin_x_ini_0.dat"
#define AUTOTB_TVOUT_x_ini_0 "../tv/cdatafile/c.myFunction.autotvout_x_ini_0.dat"
// wrapc file define:
#define AUTOTB_TVIN_x_ini_1 "../tv/cdatafile/c.myFunction.autotvin_x_ini_1.dat"
#define AUTOTB_TVOUT_x_ini_1 "../tv/cdatafile/c.myFunction.autotvout_x_ini_1.dat"
// wrapc file define:
#define AUTOTB_TVIN_x_ini_2 "../tv/cdatafile/c.myFunction.autotvin_x_ini_2.dat"
#define AUTOTB_TVOUT_x_ini_2 "../tv/cdatafile/c.myFunction.autotvout_x_ini_2.dat"
// wrapc file define:
#define AUTOTB_TVIN_Vsd "../tv/cdatafile/c.myFunction.autotvin_Vsd.dat"
#define AUTOTB_TVOUT_Vsd "../tv/cdatafile/c.myFunction.autotvout_Vsd.dat"
// wrapc file define:
#define AUTOTB_TVIN_Vsq "../tv/cdatafile/c.myFunction.autotvin_Vsq.dat"
#define AUTOTB_TVOUT_Vsq "../tv/cdatafile/c.myFunction.autotvout_Vsq.dat"
// wrapc file define:
#define AUTOTB_TVIN_iL "../tv/cdatafile/c.myFunction.autotvin_iL.dat"
#define AUTOTB_TVOUT_iL "../tv/cdatafile/c.myFunction.autotvout_iL.dat"
// wrapc file define:
#define AUTOTB_TVIN_u00_0 "../tv/cdatafile/c.myFunction.autotvin_u00_0.dat"
#define AUTOTB_TVOUT_u00_0 "../tv/cdatafile/c.myFunction.autotvout_u00_0.dat"
// wrapc file define:
#define AUTOTB_TVIN_u00_1 "../tv/cdatafile/c.myFunction.autotvin_u00_1.dat"
#define AUTOTB_TVOUT_u00_1 "../tv/cdatafile/c.myFunction.autotvout_u00_1.dat"
// wrapc file define:
#define AUTOTB_TVIN_outputVector_0 "../tv/cdatafile/c.myFunction.autotvin_outputVector_0.dat"
#define AUTOTB_TVOUT_outputVector_0 "../tv/cdatafile/c.myFunction.autotvout_outputVector_0.dat"
// wrapc file define:
#define AUTOTB_TVIN_outputVector_1 "../tv/cdatafile/c.myFunction.autotvin_outputVector_1.dat"
#define AUTOTB_TVOUT_outputVector_1 "../tv/cdatafile/c.myFunction.autotvout_outputVector_1.dat"

#define INTER_TCL "../tv/cdatafile/ref.tcl"

// tvout file define:
#define AUTOTB_TVOUT_PC_x_ini_0 "../tv/rtldatafile/rtl.myFunction.autotvout_x_ini_0.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_x_ini_1 "../tv/rtldatafile/rtl.myFunction.autotvout_x_ini_1.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_x_ini_2 "../tv/rtldatafile/rtl.myFunction.autotvout_x_ini_2.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_Vsd "../tv/rtldatafile/rtl.myFunction.autotvout_Vsd.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_Vsq "../tv/rtldatafile/rtl.myFunction.autotvout_Vsq.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_iL "../tv/rtldatafile/rtl.myFunction.autotvout_iL.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_u00_0 "../tv/rtldatafile/rtl.myFunction.autotvout_u00_0.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_u00_1 "../tv/rtldatafile/rtl.myFunction.autotvout_u00_1.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_outputVector_0 "../tv/rtldatafile/rtl.myFunction.autotvout_outputVector_0.dat"
// tvout file define:
#define AUTOTB_TVOUT_PC_outputVector_1 "../tv/rtldatafile/rtl.myFunction.autotvout_outputVector_1.dat"

class INTER_TCL_FILE {
  public:
INTER_TCL_FILE(const char* name) {
  mName = name; 
  x_ini_0_depth = 0;
  x_ini_1_depth = 0;
  x_ini_2_depth = 0;
  Vsd_depth = 0;
  Vsq_depth = 0;
  iL_depth = 0;
  u00_0_depth = 0;
  u00_1_depth = 0;
  outputVector_0_depth = 0;
  outputVector_1_depth = 0;
  trans_num =0;
}
~INTER_TCL_FILE() {
  mFile.open(mName);
  if (!mFile.good()) {
    cout << "Failed to open file ref.tcl" << endl;
    exit (1); 
  }
  string total_list = get_depth_list();
  mFile << "set depth_list {\n";
  mFile << total_list;
  mFile << "}\n";
  mFile << "set trans_num "<<trans_num<<endl;
  mFile.close();
}
string get_depth_list () {
  stringstream total_list;
  total_list << "{x_ini_0 " << x_ini_0_depth << "}\n";
  total_list << "{x_ini_1 " << x_ini_1_depth << "}\n";
  total_list << "{x_ini_2 " << x_ini_2_depth << "}\n";
  total_list << "{Vsd " << Vsd_depth << "}\n";
  total_list << "{Vsq " << Vsq_depth << "}\n";
  total_list << "{iL " << iL_depth << "}\n";
  total_list << "{u00_0 " << u00_0_depth << "}\n";
  total_list << "{u00_1 " << u00_1_depth << "}\n";
  total_list << "{outputVector_0 " << outputVector_0_depth << "}\n";
  total_list << "{outputVector_1 " << outputVector_1_depth << "}\n";
  return total_list.str();
}
void set_num (int num , int* class_num) {
  (*class_num) = (*class_num) > num ? (*class_num) : num;
}
void set_string(std::string list, std::string* class_list) {
  (*class_list) = list;
}
  public:
    int x_ini_0_depth;
    int x_ini_1_depth;
    int x_ini_2_depth;
    int Vsd_depth;
    int Vsq_depth;
    int iL_depth;
    int u00_0_depth;
    int u00_1_depth;
    int outputVector_0_depth;
    int outputVector_1_depth;
    int trans_num;
  private:
    ofstream mFile;
    const char* mName;
};

static void RTLOutputCheckAndReplacement(std::string &AESL_token, std::string PortName) {
  bool no_x = false;
  bool err = false;

  no_x = false;
  // search and replace 'X' with '0' from the 3rd char of token
  while (!no_x) {
    size_t x_found = AESL_token.find('X', 0);
    if (x_found != string::npos) {
      if (!err) { 
        cerr << "WARNING: [SIM 212-201] RTL produces unknown value 'X' on port" 
             << PortName << ", possible cause: There are uninitialized variables in the C design."
             << endl; 
        err = true;
      }
      AESL_token.replace(x_found, 1, "0");
    } else
      no_x = true;
  }
  no_x = false;
  // search and replace 'x' with '0' from the 3rd char of token
  while (!no_x) {
    size_t x_found = AESL_token.find('x', 2);
    if (x_found != string::npos) {
      if (!err) { 
        cerr << "WARNING: [SIM 212-201] RTL produces unknown value 'x' on port" 
             << PortName << ", possible cause: There are uninitialized variables in the C design."
             << endl; 
        err = true;
      }
      AESL_token.replace(x_found, 1, "0");
    } else
      no_x = true;
  }
}
extern "C" void myFunction_hw_stub_wrapper(volatile void *, volatile void *, volatile void *, float, float, float, volatile void *, volatile void *, volatile void *, volatile void *);

extern "C" void apatb_myFunction_hw(volatile void * __xlx_apatb_param_x_ini_0, volatile void * __xlx_apatb_param_x_ini_1, volatile void * __xlx_apatb_param_x_ini_2, float __xlx_apatb_param_Vsd, float __xlx_apatb_param_Vsq, float __xlx_apatb_param_iL, volatile void * __xlx_apatb_param_u00_0, volatile void * __xlx_apatb_param_u00_1, volatile void * __xlx_apatb_param_outputVector_0, volatile void * __xlx_apatb_param_outputVector_1) {
  refine_signal_handler();
  fstream wrapc_switch_file_token;
  wrapc_switch_file_token.open(".hls_cosim_wrapc_switch.log");
  int AESL_i;
  if (wrapc_switch_file_token.good())
  {

    CodeState = ENTER_WRAPC_PC;
    static unsigned AESL_transaction_pc = 0;
    string AESL_token;
    string AESL_num;{
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(AUTOTB_TVOUT_PC_outputVector_0);
        if (rtl_tv_out_file.good()) {
          rtl_tv_out_file >> AESL_token;
          if (AESL_token != "[[[runtime]]]")
            exit(1);
        }
      }
  
      if (rtl_tv_out_file.good()) {
        rtl_tv_out_file >> AESL_token; 
        rtl_tv_out_file >> AESL_num;  // transaction number
        if (AESL_token != "[[transaction]]") {
          cerr << "Unexpected token: " << AESL_token << endl;
          exit(1);
        }
        if (atoi(AESL_num.c_str()) == AESL_transaction_pc) {
          std::vector<sc_bv<32> > outputVector_0_pc_buffer(1);
          int i = 0;

          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){

            RTLOutputCheckAndReplacement(AESL_token, "outputVector_0");
  
            // push token into output port buffer
            if (AESL_token != "") {
              outputVector_0_pc_buffer[i] = AESL_token.c_str();;
              i++;
            }
  
            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
          if (i > 0) {
            ((int*)__xlx_apatb_param_outputVector_0)[0] = outputVector_0_pc_buffer[0].to_int64();
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  {
      static ifstream rtl_tv_out_file;
      if (!rtl_tv_out_file.is_open()) {
        rtl_tv_out_file.open(AUTOTB_TVOUT_PC_outputVector_1);
        if (rtl_tv_out_file.good()) {
          rtl_tv_out_file >> AESL_token;
          if (AESL_token != "[[[runtime]]]")
            exit(1);
        }
      }
  
      if (rtl_tv_out_file.good()) {
        rtl_tv_out_file >> AESL_token; 
        rtl_tv_out_file >> AESL_num;  // transaction number
        if (AESL_token != "[[transaction]]") {
          cerr << "Unexpected token: " << AESL_token << endl;
          exit(1);
        }
        if (atoi(AESL_num.c_str()) == AESL_transaction_pc) {
          std::vector<sc_bv<32> > outputVector_1_pc_buffer(1);
          int i = 0;

          rtl_tv_out_file >> AESL_token; //data
          while (AESL_token != "[[/transaction]]"){

            RTLOutputCheckAndReplacement(AESL_token, "outputVector_1");
  
            // push token into output port buffer
            if (AESL_token != "") {
              outputVector_1_pc_buffer[i] = AESL_token.c_str();;
              i++;
            }
  
            rtl_tv_out_file >> AESL_token; //data or [[/transaction]]
            if (AESL_token == "[[[/runtime]]]" || rtl_tv_out_file.eof())
              exit(1);
          }
          if (i > 0) {
            ((int*)__xlx_apatb_param_outputVector_1)[0] = outputVector_1_pc_buffer[0].to_int64();
          }
        } // end transaction
      } // end file is good
    } // end post check logic bolck
  
    AESL_transaction_pc++;
    return ;
  }
static unsigned AESL_transaction;
static AESL_FILE_HANDLER aesl_fh;
static INTER_TCL_FILE tcl_file(INTER_TCL);
std::vector<char> __xlx_sprintf_buffer(1024);
CodeState = ENTER_WRAPC;
//x_ini_0
aesl_fh.touch(AUTOTB_TVIN_x_ini_0);
aesl_fh.touch(AUTOTB_TVOUT_x_ini_0);
//x_ini_1
aesl_fh.touch(AUTOTB_TVIN_x_ini_1);
aesl_fh.touch(AUTOTB_TVOUT_x_ini_1);
//x_ini_2
aesl_fh.touch(AUTOTB_TVIN_x_ini_2);
aesl_fh.touch(AUTOTB_TVOUT_x_ini_2);
//Vsd
aesl_fh.touch(AUTOTB_TVIN_Vsd);
aesl_fh.touch(AUTOTB_TVOUT_Vsd);
//Vsq
aesl_fh.touch(AUTOTB_TVIN_Vsq);
aesl_fh.touch(AUTOTB_TVOUT_Vsq);
//iL
aesl_fh.touch(AUTOTB_TVIN_iL);
aesl_fh.touch(AUTOTB_TVOUT_iL);
//u00_0
aesl_fh.touch(AUTOTB_TVIN_u00_0);
aesl_fh.touch(AUTOTB_TVOUT_u00_0);
//u00_1
aesl_fh.touch(AUTOTB_TVIN_u00_1);
aesl_fh.touch(AUTOTB_TVOUT_u00_1);
//outputVector_0
aesl_fh.touch(AUTOTB_TVIN_outputVector_0);
aesl_fh.touch(AUTOTB_TVOUT_outputVector_0);
//outputVector_1
aesl_fh.touch(AUTOTB_TVIN_outputVector_1);
aesl_fh.touch(AUTOTB_TVOUT_outputVector_1);
CodeState = DUMP_INPUTS;
// print x_ini_0 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_x_ini_0, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_x_ini_0);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_x_ini_0, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.x_ini_0_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_x_ini_0, __xlx_sprintf_buffer.data());
}
// print x_ini_1 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_x_ini_1, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_x_ini_1);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_x_ini_1, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.x_ini_1_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_x_ini_1, __xlx_sprintf_buffer.data());
}
// print x_ini_2 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_x_ini_2, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_x_ini_2);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_x_ini_2, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.x_ini_2_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_x_ini_2, __xlx_sprintf_buffer.data());
}
// print Vsd Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_Vsd, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)&__xlx_apatb_param_Vsd);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_Vsd, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.Vsd_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_Vsd, __xlx_sprintf_buffer.data());
}
// print Vsq Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_Vsq, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)&__xlx_apatb_param_Vsq);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_Vsq, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.Vsq_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_Vsq, __xlx_sprintf_buffer.data());
}
// print iL Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_iL, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)&__xlx_apatb_param_iL);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_iL, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.iL_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_iL, __xlx_sprintf_buffer.data());
}
// print u00_0 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_u00_0, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_u00_0);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_u00_0, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.u00_0_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_u00_0, __xlx_sprintf_buffer.data());
}
// print u00_1 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_u00_1, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_u00_1);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_u00_1, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.u00_1_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_u00_1, __xlx_sprintf_buffer.data());
}
// print outputVector_0 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_outputVector_0, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_outputVector_0);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_outputVector_0, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.outputVector_0_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_outputVector_0, __xlx_sprintf_buffer.data());
}
// print outputVector_1 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVIN_outputVector_1, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_outputVector_1);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVIN_outputVector_1, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.outputVector_1_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVIN_outputVector_1, __xlx_sprintf_buffer.data());
}
CodeState = CALL_C_DUT;
myFunction_hw_stub_wrapper(__xlx_apatb_param_x_ini_0, __xlx_apatb_param_x_ini_1, __xlx_apatb_param_x_ini_2, __xlx_apatb_param_Vsd, __xlx_apatb_param_Vsq, __xlx_apatb_param_iL, __xlx_apatb_param_u00_0, __xlx_apatb_param_u00_1, __xlx_apatb_param_outputVector_0, __xlx_apatb_param_outputVector_1);
CodeState = DUMP_OUTPUTS;
// print outputVector_0 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVOUT_outputVector_0, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_outputVector_0);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVOUT_outputVector_0, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.outputVector_0_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVOUT_outputVector_0, __xlx_sprintf_buffer.data());
}
// print outputVector_1 Transactions
{
  sprintf(__xlx_sprintf_buffer.data(), "[[transaction]] %d\n", AESL_transaction);
  aesl_fh.write(AUTOTB_TVOUT_outputVector_1, __xlx_sprintf_buffer.data());
  {
    sc_bv<32> __xlx_tmp_lv = *((int*)__xlx_apatb_param_outputVector_1);

    sprintf(__xlx_sprintf_buffer.data(), "%s\n", __xlx_tmp_lv.to_string(SC_HEX).c_str());
    aesl_fh.write(AUTOTB_TVOUT_outputVector_1, __xlx_sprintf_buffer.data()); 
  }
  tcl_file.set_num(1, &tcl_file.outputVector_1_depth);
  sprintf(__xlx_sprintf_buffer.data(), "[[/transaction]] \n");
  aesl_fh.write(AUTOTB_TVOUT_outputVector_1, __xlx_sprintf_buffer.data());
}
CodeState = DELETE_CHAR_BUFFERS;
AESL_transaction++;
tcl_file.set_num(AESL_transaction , &tcl_file.trans_num);
}
