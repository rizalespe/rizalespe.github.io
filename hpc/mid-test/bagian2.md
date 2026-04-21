---
layout: hpc
title: Bagian 2 — Modul Filter Sinyal (SIMD & Vektorisasi)
---

[← Kembali ke Halaman Utama](./)

# Bagian 2: Modul Filter Sinyal
## Topik: SIMD & Auto-Vectorization (30 poin)

---

## Konteks

Sinyal mentah dari sensor hidroakustik mengandung banyak noise. Sebelum dianalisis, setiap sinyal harus difilter menggunakan **transformasi linear** — sebuah operasi sederhana namun harus dijalankan pada ratusan juta sampel per detik.

Operasi filternya adalah:

$$\text{output}[i] = \alpha \cdot \text{input}[i] + \beta \cdot \text{bias}[i] + \gamma$$

dimana $\alpha$, $\beta$, dan $\gamma$ adalah konstanta kalibrasi sensor.

Tim sebelumnya melaporkan bahwa loop ini berjalan **lebih lambat dari yang seharusnya** — padahal operasinya sangat sederhana. Kompiler tidak berhasil melakukan vektorisasi otomatis, sehingga CPU hanya memproses satu elemen per siklus alih-alih 8 elemen sekaligus (dengan SSE/AVX).

---

## Kode Awal (Bermasalah)

File: [`signal_filter.c`](code/signal_filter.c)

```c
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 100000000  // 100 juta sampel

// BUG 1: Tanpa __restrict__, kompiler tidak bisa vektorisasi
//        karena khawatir `output` dan `input` atau `bias` tumpang tindih
void apply_filter(float *output, float *input, float *bias,
                  float alpha, float beta, float gamma_val, long n) {
    for (long i = 0; i < n; i++) {
        output[i] = alpha * input[i] + beta * bias[i] + gamma_val;
    }
}

// BUG 2: Akses memori tidak sejajar (unaligned)
//        malloc tidak menjamin alignment 32-byte untuk AVX
void apply_filter_v2(float *output, float *input, float *bias,
                     float alpha, float beta, float gamma_val, long n) {
    // Simulasi unaligned: mulai dari offset 1
    output = output + 1;
    input  = input  + 1;
    bias   = bias   + 1;
    long n_adj = n - 1;

    for (long i = 0; i < n_adj; i++) {
        output[i] = alpha * input[i] + beta * bias[i] + gamma_val;
    }
}

int main() {
    float *input  = (float *)malloc(N * sizeof(float));
    float *output = (float *)malloc(N * sizeof(float));
    float *bias   = (float *)malloc(N * sizeof(float));

    // Inisialisasi data
    for (long i = 0; i < N; i++) {
        input[i] = (float)i * 0.001f;
        bias[i]  = (float)(i % 100) * 0.01f;
    }

    float alpha = 1.5f, beta = 0.3f, gamma_val = -0.5f;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    apply_filter(output, input, bias, alpha, beta, gamma_val, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_nsec - t0.tv_nsec) * 1e-9;

    printf("Checksum : %.2f\n", output[N/2]);
    printf("Waktu    : %.4f detik\n", elapsed);

    free(input); free(output); free(bias);
    return 0;
}
```

**Cara kompilasi dengan laporan vektorisasi:**
```bash
# Kompilasi tanpa optimisasi vektorisasi
gcc -O2 signal_filter.c -o signal_filter -lm

# Kompilasi dengan laporan vektorisasi GCC
gcc -O2 -fopt-info-vec-optimized -fopt-info-vec-missed \
    signal_filter.c -o signal_filter -lm

# Kompilasi dengan AVX2 (untuk server yang support)
gcc -O3 -mavx2 -fopt-info-vec-optimized \
    signal_filter.c -o signal_filter -lm
```

---

## Pertanyaan

### Soal 2.1 — Diagnosis Vektorisasi (10 poin)

**a)** Kompilasi kode dengan flag `-fopt-info-vec-missed` dan `-fopt-info-vec-optimized`. Salin output lengkap laporan kompiler tersebut ke dalam laporan Anda. Apakah loop di fungsi `apply_filter` berhasil divektorisasi? Apa pesan yang diberikan kompiler?

**b)** Jelaskan apa yang dimaksud dengan **Pointer Aliasing**. Mengapa kompiler tidak berani memvektorisasi loop yang melibatkan dua pointer berbeda (`input` dan `output`) tanpa jaminan bahwa keduanya tidak tumpang tindih?

Berikan contoh skenario konkret di mana `input` dan `output` *bisa saja* tumpang tindih, sehingga vektorisasi akan menghasilkan jawaban yang **salah**.

**c)** Jelaskan konsep **Memory Alignment** dalam konteks instruksi SIMD. Mengapa akses memori yang tidak sejajar (*unaligned*) bisa lebih lambat dibandingkan akses yang sejajar (*aligned*), khususnya pada instruksi AVX yang memproses 256 bit (8 float) sekaligus?

---

### Soal 2.2 — Perbaikan dengan `__restrict__` dan `#pragma` (12 poin)

**a)** Tambahkan kata kunci `__restrict__` pada semua parameter pointer di fungsi `apply_filter`. Kompilasi ulang dengan flag `-fopt-info-vec-optimized` dan tunjukkan bahwa loop sekarang **berhasil divektorisasi** (sertakan output kompiler sebagai bukti).

```c
// Versi yang harus Anda lengkapi:
void apply_filter_fixed(float * __restrict__ output,
                        float * __restrict__ input,
                        float * __restrict__ bias,
                        float alpha, float beta, float gamma_val, long n) {
    // ... kode Anda di sini
}
```

**b)** Sebagai alternatif, gunakan `#pragma omp simd` untuk memaksa vektorisasi **tanpa** mengubah signature fungsi:

```c
void apply_filter_pragma(float *output, float *input, float *bias,
                         float alpha, float beta, float gamma_val, long n) {
    #pragma omp simd  // Tambahkan ini
    for (long i = 0; i < n; i++) {
        output[i] = alpha * input[i] + beta * bias[i] + gamma_val;
    }
}
```

Kompilasi dengan `-fopenmp-simd` dan ukur waktunya. Bandingkan pendekatan ini dengan pendekatan `__restrict__`.

**c)** Perbaiki masalah *unaligned memory access* pada `apply_filter_v2` menggunakan `posix_memalign` atau `aligned_alloc` untuk mengalokasikan memori dengan alignment 32-byte (AVX2):

```c
// Alokasikan dengan alignment 32-byte
float *input_aligned;
posix_memalign((void **)&input_aligned, 32, N * sizeof(float));
```

Ukur perbedaan waktu antara versi *unaligned* dan *aligned* dan sajikan dalam tabel.

---

### Soal 2.3 — Benchmarking & Analisis (8 poin)

Lakukan eksperimen komprehensif dan isi tabel berikut dengan data nyata dari VM Anda:

| Versi | Flag Kompilasi | Divektorisasi? | Waktu (detik) | Speedup |
|-------|----------------|----------------|---------------|---------|
| Baseline | `-O2` | Tidak | | 1.0× |
| + `__restrict__` | `-O2` | | | |
| + AVX2 | `-O3 -mavx2` | | | |
| + Aligned + AVX2 | `-O3 -mavx2` | | | |
| + `#pragma omp simd` | `-O2 -fopenmp-simd` | | | |

**a)** Berdasarkan data Anda, berapa speedup maksimum yang berhasil dicapai? Apakah angka ini masuk akal secara teoritis? (SSE memproses 4 float, AVX2 memproses 8 float — berapa speedup teoritis maksimumnya?)

**b)** Jika speedup aktual lebih rendah dari teoritis, jelaskan **tiga faktor** yang mungkin menyebabkan gap tersebut (petunjuk: pikirkan tentang *memory bandwidth*, *pipeline latency*, dan *loop overhead*).

---

## Output yang Harus Dilaporkan

1. Output lengkap laporan vektorisasi kompiler (sebelum dan sesudah perbaikan)
2. Kode lengkap semua versi perbaikan
3. Tabel benchmarking dengan data nyata
4. Penjelasan mengapa speedup aktual ≠ speedup teoritis
5. Grafik perbandingan waktu eksekusi (opsional, +2 poin bonus)

---

[← Bagian 1: OpenMP](bagian1) | [Bagian 3 → CUDA](bagian3)
