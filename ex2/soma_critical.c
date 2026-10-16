/*
 * Aluno: Pedro de Souza Zequi
 * Lab OpenMP - Exercicio 2
 * Versao A: somatorio de vetor com #pragma omp for (somas parciais)
 *           + variavel compartilhada protegida por #pragma omp critical
 *
 * Uso:
 *   ./soma_critical forte <expoente>  -> N fixo = 2^expoente, threads 1..6
 *   ./soma_critical fraca <expoente>  -> N cresce com as threads:
 *                                        N(p) = p * (2^expoente / 6),
 *                                        com 6 threads N ~ 2^expoente
 * Cada configuracao e executada 10 vezes; mostra a media e o speedup
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define MAX_THREADS 6
#define REPETICOES  10

/* ---- somatorio paralelo: versao critical ---- */
long long soma_vetor(const int *v, long long n, int num_threads) {
   long long total = 0;                      /* compartilhada */
#  pragma omp parallel num_threads(num_threads)
   {
      long long parcial = 0;                 /* privada de cada thread */
#     pragma omp for
      for (long long i = 0; i < n; i++)
         parcial += v[i];
#     pragma omp critical
      total += parcial;                      /* secao critica */
   }
   return total;
}

int main(int argc, char *argv[]) {
   const char *versao = "critical";
   if (argc != 3 || (strcmp(argv[1], "forte") && strcmp(argv[1], "fraca"))) {
      fprintf(stderr, "Uso: %s forte|fraca <expoente>\n", argv[0]);
      return 1;
   }
   int forte = (strcmp(argv[1], "forte") == 0);
   int expoente = atoi(argv[2]);
   long long n_max = 1LL << expoente;
   long long n_base = forte ? n_max : n_max / MAX_THREADS;
   long long n_alocar = forte ? n_max : n_base * MAX_THREADS;

   /* vetor sempre com os mesmos valores: v[i] = (i % 100) + 1 */
   int *v = malloc(n_alocar * sizeof(int));
   if (!v) { fprintf(stderr, "Sem memoria para %lld inteiros\n", n_alocar); return 1; }
   for (long long i = 0; i < n_alocar; i++)
      v[i] = (int)(i % 100) + 1;

   char nome_csv[64];
   snprintf(nome_csv, sizeof nome_csv, "resultado_%s_%s.csv", versao, argv[1]);
   FILE *csv = fopen(nome_csv, "w");
   fprintf(csv, "versao;escalabilidade;threads;N");
   for (int r = 1; r <= REPETICOES; r++) fprintf(csv, ";t%d", r);
   fprintf(csv, ";media;speedup;soma\n");

   printf("Versao: %s | Escalabilidade: %s | 2^%d | %d repeticoes | nproc=%d\n",
          versao, argv[1], expoente, REPETICOES, omp_get_num_procs());
   printf("%-8s %-12s %-12s %-10s %-14s %s\n",
          "threads", "N", "media(s)", "speedup", "soma", "confere?");

   double media1 = 0.0;
   for (int p = 1; p <= MAX_THREADS; p++) {
      long long n = forte ? n_base : n_base * p;
      long long esperado = 0;
      for (long long i = 0; i < n; i++) esperado += v[i];   /* referencia serial */

      double tempos[REPETICOES], soma_t = 0.0;
      long long soma = 0;
      int ok = 1;
      soma_vetor(v, n, p);   /* aquecimento (nao entra na media) */
      for (int r = 0; r < REPETICOES; r++) {
         double ini = omp_get_wtime();
         soma = soma_vetor(v, n, p);
         tempos[r] = omp_get_wtime() - ini;
         soma_t += tempos[r];
         if (soma != esperado) ok = 0;
      }
      double media = soma_t / REPETICOES;
      if (p == 1) media1 = media;
      /* forte: S = T1/Tp | fraca (scaled speedup): S = p * T1/Tp */
      double speedup = forte ? media1 / media : p * media1 / media;

      printf("%-8d %-12lld %-12.6f %-10.3f %-14lld %s\n",
             p, n, media, speedup, soma, ok ? "OK" : "ERRO");
      fprintf(csv, "%s;%s;%d;%lld", versao, argv[1], p, n);
      for (int r = 0; r < REPETICOES; r++) fprintf(csv, ";%.6f", tempos[r]);
      fprintf(csv, ";%.6f;%.4f;%lld\n", media, speedup, soma);
   }
   fclose(csv);
   free(v);
   printf("CSV gravado em %s\n", nome_csv);
   return 0;
}