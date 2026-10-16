/*
 * Aluno: Pedro de Souza Zequi
 * Lab OpenMP - Exercicio 3
 * Versao SEQUENCIAL: multiplicacao de matrizes C = A x B
 *
 * Otimizacoes para uso de cache (L1/L2):
 *  1) Ordem de lacos i-k-j: o laco mais interno percorre B e C por linha,
 *     acessando a memoria de forma contigua (stride 1).
 *  2) Blocagem (tiling): a matriz e processada em blocos BS x BS, de modo
 *     que os blocos de A, B e C usados no momento caibam na cache.
 *
 * Uso: ./mm_sequencial <N>       (matrizes N x N de double)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define BS          64      /* tamanho do bloco: 3 blocos de 64x64 doubles = 96 KB (cabe na L2) */
#define REPETICOES  3

static inline int min(int a, int b) { return a < b ? a : b; }

void multiplica(const double *A, const double *B, double *C, int n) {
   memset(C, 0, (size_t)n * n * sizeof(double));
   for (int ii = 0; ii < n; ii += BS)
      for (int kk = 0; kk < n; kk += BS)
         for (int jj = 0; jj < n; jj += BS)
            for (int i = ii; i < min(ii + BS, n); i++)
               for (int k = kk; k < min(kk + BS, n); k++) {
                  double a = A[(size_t)i * n + k];
                  for (int j = jj; j < min(jj + BS, n); j++)
                     C[(size_t)i * n + j] += a * B[(size_t)k * n + j];
               }
}

/* soma de todos os elementos de C calculada de outro jeito, para conferir:
   soma(C) = soma_k ( soma_i A[i][k] ) * ( soma_j B[k][j] )                */
double esperado(const double *A, const double *B, int n) {
   double total = 0.0;
   for (int k = 0; k < n; k++) {
      double colA = 0.0, linB = 0.0;
      for (int i = 0; i < n; i++) colA += A[(size_t)i * n + k];
      for (int j = 0; j < n; j++) linB += B[(size_t)k * n + j];
      total += colA * linB;
   }
   return total;
}

int main(int argc, char *argv[]) {
   if (argc != 2) { fprintf(stderr, "Uso: %s <N>\n", argv[0]); return 1; }
   int n = atoi(argv[1]);
   size_t tam = (size_t)n * n;
   double *A = malloc(tam * sizeof(double));
   double *B = malloc(tam * sizeof(double));
   double *C = malloc(tam * sizeof(double));
   if (!A || !B || !C) { fprintf(stderr, "Sem memoria\n"); return 1; }

   /* valores sempre iguais (inteiros pequenos -> resultado exato em double) */
   for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++) {
         A[(size_t)i * n + j] = (double)((i + j) % 7 + 1);
         B[(size_t)i * n + j] = (double)((i * j) % 5 + 1);
      }

   FILE *csv = fopen("resultado_sequencial.csv", "w");
   fprintf(csv, "versao;threads;N;t1;t2;t3;media;checksum\n");

   printf("Versao: sequencial | N = %d | bloco = %d | %d repeticoes\n", n, BS, REPETICOES);
   double tempos[REPETICOES], soma_t = 0.0, checksum = 0.0;
   for (int r = 0; r < REPETICOES; r++) {
      double ini = omp_get_wtime();
      multiplica(A, B, C, n);
      tempos[r] = omp_get_wtime() - ini;
      soma_t += tempos[r];
   }
    for (int r = 0; r < REPETICOES; r++)       /* imprime fora da medicao */
      printf("  execucao %d: %.4f s\n", r + 1, tempos[r]);
   for (size_t x = 0; x < tam; x++) checksum += C[x];
   double media = soma_t / REPETICOES;
   printf("Media: %.4f s | checksum = %.0f | confere? %s\n", media, checksum,
          checksum == esperado(A, B, n) ? "OK" : "ERRO");

   fprintf(csv, "sequencial;1;%d;%.6f;%.6f;%.6f;%.6f;%.0f\n",
           n, tempos[0], tempos[1], tempos[2], media, checksum);
   fclose(csv);
   printf("CSV gravado em resultado_sequencial.csv\n");
   free(A); free(B); free(C);
   return 0;
}