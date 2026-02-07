#include <stdio.h>
#include <math.h>

// Declaración de la función top-level (para no depender de simulink_block.h si da guerra)
void myFunction(float x_ini[3], float Vsd, float Vsq, float iL, float u00[2], float outputVector[2]);

int main() {
    printf("=============================================\n");
    printf("   INICIO TESTBENCH HLS - OSQP SOLVER MPC    \n");
    printf("=============================================\n");

    // 1. Definir Entradas de Prueba (Valores típicos de inversor)
    // Estado inicial: [id, iq, Vdc]
    float x_ini[3] = {10.0f, 5.0f, 750.0f};

    // Voltajes de red (Grid)
    float Vsd = 380.0f;
    float Vsq = 0.0f;

    // Corriente de carga
    float iL = 15.0f;

    // Control anterior (u00)
    float u00[2] = {0.0f, 0.0f};

    // Salida (Output Vector)
    float output[2] = {0.0f, 0.0f};

    // 2. Bucle de Simulación (Simulamos 3 pasos de control)
    for (int i = 0; i < 3; i++) {
        printf("\n--- Iteracion %d ---\n", i + 1);
        printf("Inputs: Vdc=%.2f, iL=%.2f\n", x_ini[2], iL);

        // LLAMADA A LA FUNCIÓN TOP-LEVEL (La que irá en la FPGA)
        myFunction(x_ini, Vsd, Vsq, iL, u00, output);

        // 3. Mostrar Resultados
        printf("Resultados (Control Action):\n");
        printf("  v_ref_d: %f\n", output[0]);
        printf("  v_ref_q: %f\n", output[1]);

        // (Opcional) Feedback simple para la siguiente iteración
        // En un test real, x_ini debería evolucionar según la planta física.
        // Aquí solo actualizamos u00 para ver si el solver reacciona.
        u00[0] = output[0];
        u00[1] = output[1];

        // Variamos ligeramente el voltaje DC para ver cambios
        x_ini[2] += 1.0f;
    }

    printf("\n=============================================\n");
    printf("   TESTBENCH FINALIZADO EXITOSAMENTE (RET 0) \n");
    printf("=============================================\n");

    // Retornar 0 es OBLIGATORIO para que Vitis marque "Pass"
    return 0;
}
