/*
 * Aluno: Pedro de Souza Zequi
 * Lab OpenMP - Exercicio 1
 * Versao 3: clausula reduction(+: global_result) na diretiva parallel
 * (slide 35)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x) { return x * x; }

double Local_trap(double a, double b, int n);

int main(int argc, char* argv[]) {
   double global_result;
   double a, b;
   int    n, thread_count;

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

   global_result = 0.0;
#  pragma omp parallel num_threads(thread_count) \
      reduction(+: global_result)
   global_result += Local_trap(a, b, n);

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n", a, b, global_result);
   return 0;
}

double Local_trap(double a, double b, int n) {
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
   return my_result * h;
}