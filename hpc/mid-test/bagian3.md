---
layout: hpc
title: Bagian 3 — Modul Simulasi Perambatan Gelombang (CUDA)
---

[← Kembali ke Halaman Utama](./)

# Bagian 3: Modul Simulasi Perambatan Gelombang
## Topik: GPU Programming dengan CUDA (35 poin)

---

## Konteks

Modul paling kritis dalam sistem ini adalah **simulasi numerik perambatan gelombang tsunami** menggunakan model *Shallow Water Equations* yang disederhanakan. Komputasinya sangat masif — setiap titik grid harus diupdate berdasarkan nilai tetangganya di setiap langkah waktu.

Tim sebelumnya sudah memindahkan komputasi ke GPU, namun hasilnya mengecewakan:
- **Beberapa sel grid tidak pernah diupdate** (ada yang diproses dua kali, ada yang terlewat)
- **Performa GPU hanya 3× lebih cepat** dari CPU, padahal seharusnya bisa 50×+
- **Transfer data CPU↔GPU** memakan 80% dari total waktu eksekusi

---

## Kode Awal (Bermasalah)

File: [`wave_propagation.cu`](code/wave_propagation.cu)

```cuda
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define GRID_X 4096
#define GRID_Y 4096
#define TIMESTEPS 1000
#define C 0.5f  // Konstanta kecepatan gelombang

// ============================================================
// BUG 1: Global Thread Index yang salah
// Harusnya: int idx = blockIdx.x * blockDim.x + threadIdx.x;
//           int idy = blockIdx.y * blockDim.y + threadIdx.y;
// ============================================================
__global__ void update_wave(float *grid_new, float *grid_old,
                            int width, int height, float c) {
    // BUG: menggunakan threadIdx saja, mengabaikan blockIdx!
    int idx = threadIdx.x;   // <-- SALAH
    int idy = threadIdx.y;   // <-- SALAH

    if (idx >= width || idy >= height) return;

    int center = idy * width + idx;
    int left   = idy * width + (idx - 1);
    int right  = idy * width + (idx + 1);
    int up     = (idy - 1) * width + idx;
    int down   = (idy + 1) * width + idx;

    // Boundary check
    if (idx == 0 || idx == width-1 || idy == 0 || idy == height-1) {
        grid_new[center] = 0.0f;
        return;
    }

    // Finite difference update (simplified wave equation)
    grid_new[center] = grid_old[center] +
                       c * (grid_old[left] + grid_old[right] +
                            grid_old[up]   + grid_old[down]  -
                            4.0f * grid_old[center]);
}

// ============================================================
// BUG 2: Warp Divergence — percabangan kompleks di dalam kernel
// ============================================================
__global__ void classify_wave_height(float *grid, int *category,
                                     int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    int i = idy * width + idx;
    float h = grid[i];

    // BUG: Setiap thread dalam satu warp bisa masuk ke cabang yang berbeda
    //      menyebabkan warp harus mengeksekusi SEMUA cabang secara serial
    if (h < 0.1f) {
        category[i] = 0;       // Normal
    } else if (h < 0.5f) {
        category[i] = 1;       // Waspada
        // Simulasi operasi berat hanya untuk kategori ini
        for (int k = 0; k < 50; k++) h = sqrtf(h + k);
        category[i] += (int)(h * 0.0001f);
    } else if (h < 1.0f) {
        category[i] = 2;       // Siaga
        for (int k = 0; k < 100; k++) h = sqrtf(h + k);
        category[i] += (int)(h * 0.0001f);
    } else if (h < 2.0f) {
        category[i] = 3;       // Awas
        for (int k = 0; k < 200; k++) h = sqrtf(h + k);
        category[i] += (int)(h * 0.0001f);
    } else {
        category[i] = 4;       // Kritis
        for (int k = 0; k < 500; k++) h = sqrtf(h + k);
        category[i] += (int)(h * 0.0001f);
    }
}

// ============================================================
// BUG 3: Transfer data H2D/D2H terjadi di setiap timestep!
// ============================================================
void simulate_with_excessive_transfer(float *h_grid, int width, int height) {
    float *d_grid_old, *d_grid_new;
    size_t size = width * height * sizeof(float);

    dim3 blockDim(16, 16);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    for (int t = 0; t < TIMESTEPS; t++) {
        // BUG: Alokasi dan transfer DI DALAM LOOP!
        cudaMalloc(&d_grid_old, size);
        cudaMalloc(&d_grid_new, size);

        cudaMemcpy(d_grid_old, h_grid, size, cudaMemcpyHostToDevice);  // H2D

        update_wave<<<gridDim, blockDim>>>(d_grid_new, d_grid_old,
                                           width, height, C);

        cudaMemcpy(h_grid, d_grid_new, size, cudaMemcpyDeviceToHost);  // D2H

        cudaFree(d_grid_old);
        cudaFree(d_grid_new);
        // Ini terjadi 1000 kali! Sangat boros!
    }
}

int main() {
    size_t size = GRID_X * GRID_Y * sizeof(float);
    float *h_grid = (float *)malloc(size);

    // Inisialisasi: gelombang di tengah grid
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            float dx = j - GRID_X/2.0f;
            float dy = i - GRID_Y/2.0f;
            float r  = sqrtf(dx*dx + dy*dy);
            h_grid[i * GRID_X + j] = expf(-r*r / 1000.0f);
        }
    }

    printf("Memulai simulasi...\n");
    simulate_with_excessive_transfer(h_grid, GRID_X, GRID_Y);
    printf("Selesai.\n");

    free(h_grid);
    return 0;
}
```

**Cara kompilasi & profiling:**
```bash
nvcc -O2 wave_propagation.cu -o wave_propagation
./wave_propagation

# Profiling dengan nvprof
nvprof ./wave_propagation

# Profiling dengan Nsight (jika tersedia)
ncu --set full ./wave_propagation
```

---

## Pertanyaan

### Soal 3.1 — Perbaikan Global Thread Index (10 poin)

**a)** Jalankan kode awal dengan konfigurasi berikut:
```bash
# Periksa output: apakah semua sel grid diupdate?
# Coba dengan GRID_X = GRID_Y = 64 (untuk observasi lebih mudah)
```

Jelaskan secara matematis **mengapa hanya sebagian kecil grid yang diproses**. Jika `blockDim = (16, 16)` dan `gridDim = (256, 256)`, berapa sel yang seharusnya diproses? Berapa yang aktualnya diproses dengan kode yang salah?

**b)** Perbaiki fungsi `update_wave` dengan menggunakan formula global thread index yang benar:

```cuda
int idx = blockIdx.x * blockDim.x + threadIdx.x;
int idy = blockIdx.y * blockDim.y + threadIdx.y;
```

Jelaskan apa peran `blockIdx`, `blockDim`, dan `threadIdx` masing-masing, dan mengapa ketiga komponen ini harus dikombinasikan.

**c)** Tambahkan **bounds checking** yang tepat setelah memperbaiki indeks, dan jelaskan mengapa hal ini penting khususnya ketika ukuran grid tidak habis dibagi ukuran block.

---

### Soal 3.2 — Meminimalkan Warp Divergence (12 poin)

**a)** Jelaskan konsep **Warp** dalam arsitektur GPU NVIDIA. Berapa thread dalam satu warp? Apa yang terjadi ketika thread-thread dalam satu warp mengeksekusi cabang `if-else` yang berbeda?

**b)** Analisis fungsi `classify_wave_height`. Dalam skenario realistis di mana nilai `h` terdistribusi acak, berapa persen dari waktu eksekusi yang "terbuang" akibat *warp divergence*? (Estimasi, tidak perlu presisi 100%)

**c)** Lakukan **refaktorisasi** untuk mengurangi divergence. Salah satu pendekatannya adalah memisahkan kalkulasi berat dari klasifikasi, atau menggunakan teknik *predication*:

```cuda
// Pendekatan: hitung untuk semua, terapkan kondisi dengan masking
__global__ void classify_wave_no_divergence(float *grid, int *category,
                                             int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx >= width || idy >= height) return;

    int i = idy * width + idx;
    float h = grid[i];

    // Gunakan ekspresi aritmetika alih-alih if-else bercabang
    // Petunjuk: (kondisi) * nilai akan menghasilkan nilai jika kondisi benar,
    //           dan 0 jika kondisi salah — tanpa divergence!
    // ... implementasikan pendekatan Anda di sini
}
```

Tunjukkan kode hasil refaktorisasi Anda dan ukur perbedaan performa menggunakan `nvprof`.

---

### Soal 3.3 — Optimisasi Transfer Data Host-Device (13 poin)

**a)** Jalankan `nvprof ./wave_propagation` pada kode awal. Dari output profiler, berapa persentase total waktu yang dihabiskan untuk:
- Eksekusi kernel GPU
- Transfer `cudaMemcpy` (H2D + D2H)
- Alokasi/dealokasi memori (`cudaMalloc`/`cudaFree`)

**b)** Tulis ulang fungsi `simulate_with_excessive_transfer` menjadi versi yang optimal. Prinsipnya: **alokasikan sekali, transfer hanya jika perlu**:

```cuda
void simulate_optimized(float *h_grid, int width, int height) {
    float *d_grid_old, *d_grid_new;
    size_t size = width * height * sizeof(float);

    // TODO: Alokasikan d_grid_old dan d_grid_new di LUAR loop

    // TODO: Transfer h_grid -> d_grid_old SEKALI di awal

    dim3 blockDim(16, 16);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    for (int t = 0; t < TIMESTEPS; t++) {
        // TODO: Panggil kernel
        // TODO: Swap pointer d_grid_old <-> d_grid_new (tidak perlu copy!)
    }

    // TODO: Transfer d_grid_old -> h_grid SEKALI di akhir

    // TODO: Bebaskan memori GPU
}
```

Implementasikan fungsi ini dan bandingkan timeline eksekusi dengan `nvprof`.

**c)** Buat tabel perbandingan performa antara versi bermasalah dan versi optimal:

| Metrik | Versi Awal | Versi Optimal | Improvement |
|--------|-----------|---------------|-------------|
| Total waktu eksekusi (s) | | | |
| Waktu kernel GPU (s) | | | |
| Waktu transfer data (s) | | | |
| Jumlah panggilan `cudaMalloc` | 2000× | | |
| Throughput efektif (GB/s) | | | |

**d) (Bonus, 5 poin):** Implementasikan penggunaan **CUDA Streams** untuk melakukan overlap antara komputasi kernel dengan transfer data asinkron. Jelaskan kapan teknik ini bermanfaat dan kapan tidak.

---

## Output yang Harus Dilaporkan

1. Output `nvprof` dari kode awal (tunjukkan bottleneck transfer data)
2. Penjelasan matematis kesalahan global thread index
3. Kode lengkap semua versi perbaikan (index fix + warp divergence fix + transfer optimization)
4. Output `nvprof` dari kode optimal (tunjukkan improvement)
5. Tabel perbandingan performa dengan data nyata dari CUDA server
6. Analisis: apakah bottleneck berhasil dieliminasi? Apa bottleneck berikutnya?

---

[← Bagian 2: SIMD](bagian2) | [Template Laporan →](template_laporan)
