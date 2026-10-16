/*
 * Aluno: Pedro de Souza Zequi
 * Lab OpenMP - Exercicio 1
 * Versao 4: diretiva parallel for com reduction(+: approx)
 * (slide 38)
 */
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

double f(double x) { return x * x; }

int main(int argc, char* argv[]) {
   double a, b, h, approx;
   int    n, i, thread_count;

   if (argc != 2) {
      fprintf(stderr, "Uso: %s <num_threads>\n", argv[0]);
      exit(1);
   }
   thread_count = strtol(argv[1], NULL, 10);
   printf("Enter a, b, and n\n");
   if (scanf("%lf %lf %d", &a, &b, &n) != 3) return 1;

   h = (b - a) / n;
   approx = (f(a) + f(b)) / 2.0;
#  pragma omp parallel for num_threads(thread_count) \
      reduction(+: approx)
   for (i = 1; i <= n - 1; i++)
      approx += f(a + i * h);
   approx = h * approx;

   printf("With n = %d trapezoids, our estimate\n", n);
   printf("of the integral from %f to %f = %.14e\n", a, b, approx);
   return 0;
}