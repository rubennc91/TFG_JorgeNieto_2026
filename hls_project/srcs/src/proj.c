#include "proj.h"
#include "workspace.h"

void project(c_float *z) {
  c_int i;
  for (i = 0; i < data.m; i++) {
    // Usamos ldata y udata globales en vez de work->data->l/u
    z[i] = c_min(c_max(z[i], ldata[i]), udata[i]);
  }
}
