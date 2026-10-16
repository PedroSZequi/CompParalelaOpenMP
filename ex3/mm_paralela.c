/*
 * Aluno: Pedro de Souza Zequi
 * Lab OpenMP - Exercicio 3
 * Versao PARALELA (OpenMP): multiplicacao de matrizes C = A x B
 *
 * Otimizacoes para uso de cache (L1/L2):
 *  1) Ordem de lacos i-k-j: o laco mais interno percorre B e C por linha,
 *     acessando a memoria de forma contigua (stride 1).
 *  2) Blocagem (tiling): a matriz e processada em blocos BS x BS, de modo
 *     que os blocos de A, B e C usados no momento caibam na cache.
 *
 *  3) Paralelizacao: #pragma omp parallel for no laco dos blocos de linhas
 *     (ii). Cada thread calcula faixas de linhas de C diferentes, entao
 *     nao ha condicao de corrida e nao e preciso critical nem reduction.
 *     schedule(dynamic) equilibra a carga quando o numero de blocos nao
 *     e multiplo do numero de threads.
 *
 * Uso: ./mm_paralela <N>   (matrizes N x N de double, threads 1..6)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define BS          64      /* tamanho do bloco: 3 blocos de 64x64 doubles = 96 KB (cabe na L2) */
#define REPETICOES  3
#define MAX_THREADS 6

static inline int min(int a, int b) { return a < b ? a : b; }

void multiplica(const double *A, const double *B, double *C, int n, int num_threads) {
   memset(C, 0, (size_t)n * n * sizeof(double));
#  pragma omp parallel for num_threads(num_threads) schedule(dynamic)
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

   double ref = esperado(A, B, n);
   FILE *csv = fopen("resultado_paralela.csv", "w");
   fprintf(csv, "versao;threads;N;t1;t2;t3;media;speedup;checksum\n");

   printf("Versao: paralela | N = %d | bloco = %d | %d repeticoes | nproc=%d\n",
          n, BS, REPETICOES, omp_get_num_procs());
   printf("%-8s %-10s %-10s %-10s %-10s %-9s %-14s %s\n",
          "threads", "t1(s)", "t2(s)", "t3(s)", "media(s)", "speedup", "checksum", "confere?");
   double media1 = 0.0;
   for (int p = 1; p <= MAX_THREADS; p++) {
      double tempos[REPETICOES], soma_t = 0.0;
      for (int r = 0; r < REPETICOES; r++) {
         double ini = omp_get_wtime();
         multiplica(A, B, C, n, p);
         tempos[r] = omp_get_wtime() - ini;
         soma_t += tempos[r];
      }
      double checksum = 0.0;
      for (size_t x = 0; x < tam; x++) checksum += C[x];
      double media = soma_t / REPETICOES;
      if (p == 1) media1 = media;
      double speedup = media1 / media;
      printf("%-8d %-10.4f %-10.4f %-10.4f %-10.4f %-9.3f %-14.0f %s\n", p,
             tempos[0], tempos[1], tempos[2], media, speedup, checksum,
             checksum == ref ? "OK" : "ERRO");
      fprintf(csv, "paralela;%d;%d;%.6f;%.6f;%.6f;%.6f;%.4f;%.0f\n",
              p, n, tempos[0], tempos[1], tempos[2], media, speedup, checksum);
   }
   fclose(csv);
   printf("CSV gravado em resultado_paralela.csv\n");
   free(A); free(B); free(C);
   return 0;
}