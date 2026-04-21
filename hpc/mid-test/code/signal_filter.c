/**
 * signal_filter.c — Modul Filter Sinyal Sensor Hidroakustik
 * UTS Praktikum HPC — Bagian 2: SIMD & Vektorisasi
 *
 * Kompilasi (baseline, tanpa vektorisasi):
 *   gcc -O2 -fopt-info-vec-missed -fopt-info-vec-optimized \
 *       signal_filter.c -o signal_filter -lm
 *
 * Kompilasi (dengan AVX2):
 *   gcc -O3 -mavx2 -fopt-info-vec-optimized \
 *       signal_filter.c -o signal_filter -lm
 *
 * PERHATIAN: Kode ini mengandung bug/kekurangan yang DISENGAJA untuk ujian.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define N 100000000L   /* 100 juta sampel */

/* ============================================================
 * BUG 1: Pointer Aliasing — kompiler tidak bisa vektorisasi
 * ============================================================
 * Kompiler harus mengasumsikan bahwa `output`, `input`, dan `bias`
 * mungkin menunjuk ke region memori yang sama (overlap).
 * Oleh karena itu, kompiler tidak berani menggunakan instruksi SIMD
 * yang memproses beberapa elemen sekaligus, karena urutan operasi
 * bisa menghasilkan hasil yang berbeda jika ada overlap.
 */
void apply_filter(float *output, float *input, float *bias,
                  float alpha, float beta, float gamma_val, long n) {
    for (long i = 0; i < n; i++) {
        output[i] = alpha * input[i] + beta * bias[i] + gamma_val;
    }
}

/* ============================================================
 * BUG 2: Unaligned Memory Access
 * ============================================================
 * malloc() tidak menjamin alignment khusus (biasanya 8 atau 16 byte).
 * Instruksi AVX2 (ymm registers) bekerja optimal dengan data yang
 * sejajar pada batas 32-byte. Akses ke pointer+1 hampir pasti
 * menghasilkan unaligned access untuk semua elemen berikutnya.
 */
void apply_filter_unaligned(float *output, float *input, float *bias,
                             float alpha, float beta, float gamma_val, long n) {
    /* Simulasi unaligned: mulai dari offset 1 byte dari pointer asli */
    float *out_unaligned = output + 1;
    float *in_unaligned  = input  + 1;
    float *bias_unaligned= bias   + 1;
    long   n_adj         = n - 1;

    for (long i = 0; i < n_adj; i++) {
        out_unaligned[i] = alpha * in_unaligned[i]
                         + beta  * bias_unaligned[i]
                         + gamma_val;
    }
}

/* ============================================================
 * Helper: checksum untuk memverifikasi kebenaran hasil
 * ============================================================ */
double checksum(float *arr, long n) {
    double s = 0.0;
    for (long i = 0; i < n; i += 1000) s += arr[i];
    return s;
}

/* ============================================================
 * Helper: ukur waktu dalam detik
 * ============================================================ */
static double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(void) {
    printf("=== Modul Filter Sinyal Sensor ===\n");
    printf("Jumlah sampel : %ld\n", N);
    printf("==================================\n\n");

    /* Alokasi memori (malloc standar, tidak guaranteed aligned) */
    float *input  = (float *)malloc((N + 2) * sizeof(float));
    float *output = (float *)malloc((N + 2) * sizeof(float));
    float *bias   = (float *)malloc((N + 2) * sizeof(float));

    if (!input || !output || !bias) {
        fprintf(stderr, "GAGAL: Tidak cukup memori\n");
        return 1;
    }

    /* Inisialisasi data */
    printf("Menginisialisasi data...\n");
    for (long i = 0; i < N + 2; i++) {
        input[i] = (float)i * 0.001f;
        bias[i]  = (float)(i % 100) * 0.01f;
    }

    float alpha     =  1.5f;
    float beta      =  0.3f;
    float gamma_val = -0.5f;

    /* --- Benchmark 1: apply_filter (pointer aliasing) --- */
    printf("\n--- Benchmark 1: apply_filter (baseline) ---\n");
    memset(output, 0, (N + 2) * sizeof(float));
    double t0 = get_time();
    apply_filter(output, input, bias, alpha, beta, gamma_val, N);
    double t1 = get_time();
    printf("Checksum : %.4f\n", checksum(output, N));
    printf("Waktu    : %.4f detik\n", t1 - t0);

    /* --- Benchmark 2: apply_filter_unaligned --- */
    printf("\n--- Benchmark 2: apply_filter_unaligned ---\n");
    memset(output, 0, (N + 2) * sizeof(float));
    t0 = get_time();
    apply_filter_unaligned(output, input, bias, alpha, beta, gamma_val, N);
    t1 = get_time();
    printf("Checksum : %.4f\n", checksum(output, N - 1));
    printf("Waktu    : %.4f detik\n", t1 - t0);

    printf("\n[TUGAS] Tambahkan benchmark untuk versi yang sudah Anda perbaiki.\n");

    free(input);
    free(output);
    free(bias);
    return 0;
}
