/**
 * wave_stats.c — Modul Analisis Statistik Gelombang
 * UTS Praktikum HPC — Bagian 1: OpenMP
 *
 * Kompilasi:
 *   gcc -O2 -fopenmp wave_stats.c -o wave_stats -lm
 *
 * Jalankan:
 *   OMP_NUM_THREADS=4 ./wave_stats
 *
 * PERHATIAN: Kode ini mengandung bug yang DISENGAJA untuk tujuan ujian.
 *            Tugas Anda adalah mengidentifikasi dan memperbaikinya.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

/* Kurangi N jika VM tidak cukup RAM: bisa pakai 50000000 */
#define N 500000000L

/* ============================================================
 * BUG AREA 1: Variabel global yang diakses oleh banyak thread
 * ============================================================ */
double sum     = 0.0;
double max_val = 0.0;

/**
 * compute_stats — Menghitung jumlah dan nilai maksimum dari data gelombang.
 *
 * BUGS:
 *   1. `sum`     diakses tanpa proteksi -> Race Condition
 *   2. `max_val` diakses tanpa proteksi -> Race Condition
 *   3. schedule(static) tidak cocok untuk beban heterogen
 */
void compute_stats(float *data, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        sum += data[i];           /* BUG 1: Race Condition */

        if (data[i] > max_val) {
            max_val = data[i];    /* BUG 2: Race Condition */
        }
    }
}

/* ============================================================
 * BUG AREA 2: False Sharing pada array partial sums
 * ============================================================
 * Array `partial` berukuran MAX_THREADS double (masing-masing 8 byte).
 * Satu cache line = 64 byte = 8 double.
 * Jika ada 4 thread, semua elemen partial[0..3] masuk dalam SATU cache line.
 * Ketika thread 0 mengupdate partial[0], cache line tersebut di-invalidate
 * untuk semua core lain, meskipun thread 1, 2, 3 mengakses elemen berbeda.
 */
#define MAX_THREADS 64

void update_partial_sums(float *data, long n) {
    int nthreads = omp_get_max_threads();
    double partial[MAX_THREADS];   /* BUG 3: False Sharing! */

    for (int t = 0; t < nthreads; t++) partial[t] = 0.0;

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        partial[tid] += data[i];  /* False sharing terjadi di sini */
    }

    /* Reduce partial sums */
    double total = 0.0;
    for (int t = 0; t < nthreads; t++) total += partial[t];
    printf("[partial_sums] Total: %.4f\n", total);
}

/**
 * generate_wave_data — Membangkitkan data sintetis sensor gelombang.
 * Nilai bervariasi antara 0.0 dan 10.0 meter (ketinggian gelombang).
 */
void generate_wave_data(float *data, long n, unsigned int seed) {
    srand(seed);
    for (long i = 0; i < n; i++) {
        data[i] = (float)(rand() % 1000) / 100.0f;
    }
}

int main(int argc, char *argv[]) {
    printf("=== Modul Analisis Statistik Gelombang ===\n");
    printf("Jumlah titik sensor : %ld\n", N);
    printf("Jumlah thread       : %d\n", omp_get_max_threads());
    printf("==========================================\n\n");

    /* Alokasi memori */
    float *wave_data = (float *)malloc(N * sizeof(float));
    if (!wave_data) {
        fprintf(stderr, "GAGAL: Tidak cukup memori untuk %ld float\n", N);
        return 1;
    }

    printf("Membangkitkan data sensor...\n");
    generate_wave_data(wave_data, N, 42);

    /* --- Benchmark compute_stats --- */
    sum = 0.0; max_val = 0.0;  /* reset */
    double t_start = omp_get_wtime();
    compute_stats(wave_data, N);
    double t_end   = omp_get_wtime();

    printf("\n--- Hasil compute_stats ---\n");
    printf("Rata-rata ketinggian : %.6f meter\n", sum / (double)N);
    printf("Ketinggian maksimum  : %.6f meter\n", max_val);
    printf("Waktu eksekusi       : %.4f detik\n", t_end - t_start);

    /* --- Benchmark update_partial_sums --- */
    printf("\n--- Benchmark update_partial_sums ---\n");
    t_start = omp_get_wtime();
    update_partial_sums(wave_data, N);
    t_end   = omp_get_wtime();
    printf("Waktu eksekusi : %.4f detik\n", t_end - t_start);

    free(wave_data);
    return 0;
}
