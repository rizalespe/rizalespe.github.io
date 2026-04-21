---
layout: hpc
title: Bagian 1 — Modul Analisis Statistik (OpenMP)
---

[← Kembali ke Halaman Utama](./)

# Bagian 1: Modul Analisis Statistik
## Topik: Shared Memory & OpenMP (35 poin)

---

## Konteks

Modul pertama dalam sistem pemantauan gelombang bertugas menghitung **statistik deskriptif** (rata-rata, varians, dan nilai maksimum) dari data ketinggian gelombang yang dikumpulkan oleh jutaan titik sensor yang tersebar di perairan Indonesia.

Data yang diproses mencapai **500 juta titik pengukuran** per siklus pemantauan. Tim pengembang sebelumnya sudah mencoba mempercepat proses ini dengan OpenMP, namun hasilnya **tidak konsisten** — terkadang nilai rata-rata berubah setiap kali program dijalankan — dan speedup yang diperoleh jauh di bawah ekspektasi.

---

## Kode Awal (Bermasalah)

File: [`wave_stats.c`](code/wave_stats.c)

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 500000000  // 500 juta titik sensor

double sum = 0.0;        // variabel global
double max_val = 0.0;    // variabel global

void compute_stats(float *data, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        sum += data[i];              // BUG 1: Race condition!

        if (data[i] > max_val) {
            max_val = data[i];       // BUG 2: Race condition!
        }
    }
}

// False sharing simulation
void update_partial_sums(float *data, double *partial, int nthreads, long n) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < n; i++) {
        int tid = omp_get_thread_num();
        partial[tid] += data[i];    // BUG 3: False sharing!
    }
}

int main() {
    float *wave_data = (float *)malloc(N * sizeof(float));

    // Generate synthetic sensor data
    srand(42);
    for (long i = 0; i < N; i++) {
        wave_data[i] = (float)(rand() % 1000) / 100.0f;
    }

    double t_start = omp_get_wtime();
    compute_stats(wave_data, N);
    double t_end = omp_get_wtime();

    printf("Rata-rata: %.4f\n", sum / N);
    printf("Maksimum : %.4f\n", max_val);
    printf("Waktu    : %.3f detik\n", t_end - t_start);

    free(wave_data);
    return 0;
}
```

**Cara kompilasi:**
```bash
gcc -O2 -fopenmp wave_stats.c -o wave_stats
OMP_NUM_THREADS=4 ./wave_stats
```

---

## Pertanyaan

### Soal 1.1 — Identifikasi Bug (10 poin)

Jalankan program di atas **minimal 5 kali** dengan `OMP_NUM_THREADS=4`. Catat setiap output yang dihasilkan.

**a)** Apakah nilai "Rata-rata" dan "Maksimum" konsisten di setiap run? Tunjukkan output terminal Anda sebagai bukti.

**b)** Jelaskan secara teknis **mengapa Race Condition terjadi** pada variabel `sum` dan `max_val`. Gunakan konsasi *critical section* dan *memory model* OpenMP dalam penjelasan Anda.

**c)** Jelaskan apa yang dimaksud dengan **False Sharing** pada fungsi `update_partial_sums`. Mengapa penempatan array `partial[]` seperti pada kode di atas bisa memperlambat performa meskipun tidak ada race condition yang "terlihat"?

> **Petunjuk:** Perhatikan ukuran *cache line* pada arsitektur x86-64 (64 byte). Hitung berapa elemen `double` yang muat dalam satu cache line.

---

### Soal 1.2 — Perbaikan Kode (15 poin)

**a)** Perbaiki fungsi `compute_stats` menggunakan **klausa `reduction`** OpenMP untuk menghilangkan race condition pada `sum`. Sertakan kode lengkap yang sudah diperbaiki.

**b)** Perbaiki race condition pada `max_val` menggunakan salah satu dari pendekatan berikut (pilih satu dan jelaskan pilihan Anda):
  - `#pragma omp critical`
  - `#pragma omp atomic` (jika berlaku)
  - Pendekatan *reduction* manual dengan variabel lokal per-thread

**c)** Perbaiki masalah **False Sharing** pada `update_partial_sums`. Gunakan teknik *padding* agar setiap elemen `partial[tid]` berada di cache line yang berbeda. Implementasikan dan ukur perbedaan waktunya.

---

### Soal 1.3 — Eksperimen Scheduling (10 poin)

Anggap distribusi data sensor tidak merata — beberapa area laut memerlukan komputasi 10× lebih berat (misalnya area sekitar pusat gempa memerlukan validasi silang tambahan). Simulasikan kondisi ini dengan menambahkan delay artifisial:

```c
// Tambahkan kode ini di dalam loop compute_stats (versi perbaikan Anda)
if (i % 1000 == 0) {
    // Simulasi beban berat: hitung sqrt 100 kali
    double dummy = data[i];
    for (int k = 0; k < 100; k++) dummy = sqrt(dummy + k);
    sum += dummy * 1e-10;  // agar tidak di-optimize away
}
```

Lakukan eksperimen dengan **semua strategi scheduling** berikut dan catat waktu eksekusi:

| Strategi | Waktu (detik) | Speedup vs Serial |
|----------|---------------|-------------------|
| Serial (tanpa OpenMP) | | — |
| `schedule(static)` | | |
| `schedule(dynamic, 1)` | | |
| `schedule(dynamic, 1000)` | | |
| `schedule(guided)` | | |

**a)** Berdasarkan data Anda, strategi mana yang terbaik untuk beban kerja *heterogen* seperti ini? Jelaskan mengapa.

**b)** Hitung **Speedup** dan **Efisiensi Parallel** untuk konfigurasi terbaik Anda:

$$S = \frac{T_{serial}}{T_{parallel}}$$

$$E = \frac{S}{p} \times 100\%$$

dimana $p$ adalah jumlah thread yang digunakan.

---

## Output yang Harus Dilaporkan

1. Screenshot/output terminal menunjukkan inkonsistensi pada kode awal
2. Kode lengkap versi perbaikan dengan komentar penjelasan
3. Tabel hasil eksperimen scheduling (dengan data nyata dari komputer Anda)
4. Grafik perbandingan waktu eksekusi (bar chart atau line chart)
5. Analisis singkat trade-off antara overhead scheduling dan load balancing

---

[← Kembali ke Halaman Utama](./) | [Bagian 2 → SIMD](bagian2)
