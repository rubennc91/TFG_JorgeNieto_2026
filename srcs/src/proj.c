#include "proj.h"
#include "workspace.h" // Acceso a ldata, udata, data.m

void project(c_float *z) {
  c_int i;

  // Usamos data.m global
  for (i = 0; i < data.m; i++) {
    // Usamos ldata y udata globales en vez de work->data->l
    z[i] = c_min(c_max(z[i], ldata[i]), udata[i]);
  }
}
