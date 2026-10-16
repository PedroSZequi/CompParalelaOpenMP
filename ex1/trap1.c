/*
 * Aluno: Pedro de Souza Zequi
 * Lab OpenMP - Exercicio 1
 * Versao 1: Regra do Trapezio com #pragma omp critical dentro de Trap()
 * (slides 24-25)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x) { return x * x; }   /* integral de x^2 de 0 a 3 = 9 */

void Trap(double a, double b, int n, double* global_result_p);

int main(int argc, char* argv[]) {
   double global_result = 0.0;   /* Resultado guardado em global_result */
   double a, b;                  /* Extremos esquerdo e direito         */
   int    n;                     /* Numero total de trapezios           */
   int    thread_count;

   if (argc != 2) {
      fprintf(stderr, "Uso: %s <num_threads>\n", argv[0]);
      exit(1);
   }
   thread_count = strtol(argv[1], NULL, 10);
   printf("Enter a, b, and n\n");
   if (scanf("%lf %lf %d", &a, &b, &n) != 3) return 1;
   if (n % thread_count != 0) {
      fprintf(stderr, "n deve ser divisivel pelo numero de threads\n");
      exit(1);
   }

#  pragma omp parallel num_threads(thread_count)
   Trap(a, b, n, &global_result);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n", a, b, global_result);
   return 0;
}

void Trap(double a, double b, int n, double* global_result_p) {
   double h, x, my_result;
   double local_a, local_b;
   int    i, local_n;
   int    my_rank = omp_get_thread_num();
   int    thread_count = omp_get_num_threads();

   h = (b - a) / n;
   local_n = n / thread_count;
   local_a = a + my_rank * local_n * h;
   local_b = local_a + local_n * h;
   my_result = (f(local_a) + f(local_b)) / 2.0;
   for (i = 1; i <= local_n - 1; i++) {
      x = local_a + i * h;
      my_result += f(x);
   }
   my_result = my_result * h;

#  pragma omp critical
   *global_result_p += my_result;
}