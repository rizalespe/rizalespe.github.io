/**
 * wave_propagation.cu — Modul Simulasi Perambatan Gelombang Tsunami
 * UTS Praktikum HPC — Bagian 3: CUDA
 *
 * Kompilasi:
 *   nvcc -O2 wave_propagation.cu -o wave_propagation -lm
 *
 * Jalankan:
 *   ./wave_propagation
 *
 * Profiling:
 *   nvprof ./wave_propagation
 *   ncu --set full ./wave_propagation
 *
 * PERHATIAN: Kode ini mengandung tiga bug yang DISENGAJA untuk ujian.
 *            Identifikasi, perbaiki, dan ukur dampaknya.
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

/* Untuk observasi lebih mudah, gunakan grid kecil dulu:
 * Ganti GRID_X/GRID_Y ke 64 untuk debugging, lalu kembali ke 4096 */
#define GRID_X    4096
#define GRID_Y    4096
#define TIMESTEPS 1000
#define DT        0.01f   /* Time step */
#define DX        1.0f    /* Spatial resolution (meter) */
#define C_WAVE    0.5f    /* Kecepatan gelombang (normalized) */

/* ============================================================
 * Helper: CUDA error checking
 * ============================================================ */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ============================================================
 * KERNEL 1: update_wave — Finite Difference Wave Equation
 *
 * BUG 1: Global Thread Index yang SALAH
 *   Kode menggunakan threadIdx saja, mengabaikan blockIdx.
 *   Akibatnya, hanya 16×16 = 256 sel yang diproses dari
 *   total 4096×4096 = 16.777.216 sel.
 * ============================================================ */
__global__ void update_wave(float *grid_new, float *grid_old,
                            int width, int height, float c) {
    /* BUG: Harusnya menggunakan blockIdx untuk global index */
    int idx = threadIdx.x;   /* <- SALAH: harusnya blockIdx.x * blockDim.x + threadIdx.x */
    int idy = threadIdx.y;   /* <- SALAH: harusnya blockIdx.y * blockDim.y + threadIdx.y */

    /* Boundary check */
    if (idx >= width || idy >= height) return;

    int center = idy * width + idx;

    /* Boundary cells: set to zero (absorbing boundary) */
    if (idx == 0 || idx == width - 1 || idy == 0 || idy == height - 1) {
        grid_new[center] = 0.0f;
        return;
    }

    int left  = idy * width + (idx - 1);
    int right = idy * width + (idx + 1);
    int up    = (idy - 1) * width + idx;
    int down  = (idy + 1) * width + idx;

    /* 2D Wave Equation: u_tt = c^2 * (u_xx + u_yy)
     * Simplified explicit finite difference update */
    grid_new[center] = grid_old[center]
                     + c * (grid_old[left]  + grid_old[right]
                           + grid_old[up]   + grid_old[down]
                           - 4.0f * grid_old[center]);
}

/* ============================================================
 * KERNEL 2: classify_wave_height — Kategorisasi Tingkat Bahaya
 *
 * BUG 2: Warp Divergence yang parah
 *   Setiap thread mengeksekusi cabang if-else yang berbeda,
 *   menyebabkan seluruh warp (32 thread) harus menunggu
 *   semua cabang selesai dieksekusi secara serial.
 *   Loop heavy computation (sqrtf) makin memperparah masalah.
 * ============================================================ */
__global__ void classify_wave_height(float *grid, int *category,
                                     int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= width || idy >= height) return;

    int i = idy * width + idx;
    float h = grid[i];

    /* BUG: Tiap thread bisa masuk cabang berbeda -> Warp Divergence */
    if (h < 0.1f) {
        category[i] = 0;                          /* Normal */

    } else if (h < 0.5f) {
        category[i] = 1;                          /* Waspada */
        float tmp = h;
        for (int k = 0; k < 50; k++) tmp = sqrtf(tmp + (float)k);
        category[i] += (int)(tmp * 0.0001f);

    } else if (h < 1.0f) {
        category[i] = 2;                          /* Siaga */
        float tmp = h;
        for (int k = 0; k < 100; k++) tmp = sqrtf(tmp + (float)k);
        category[i] += (int)(tmp * 0.0001f);

    } else if (h < 2.0f) {
        category[i] = 3;                          /* Awas */
        float tmp = h;
        for (int k = 0; k < 200; k++) tmp = sqrtf(tmp + (float)k);
        category[i] += (int)(tmp * 0.0001f);

    } else {
        category[i] = 4;                          /* Kritis */
        float tmp = h;
        for (int k = 0; k < 500; k++) tmp = sqrtf(tmp + (float)k);
        category[i] += (int)(tmp * 0.0001f);
    }
}

/* ============================================================
 * BUG 3: Transfer data Host-to-Device/Device-to-Host di setiap timestep
 *
 * cudaMalloc dan cudaMemcpy dipanggil TIMESTEPS kali!
 * Dengan TIMESTEPS=1000 dan grid 4096×4096×4 byte = 64 MB:
 *   - 1000 × cudaMalloc (sangat lambat, bisa ratusan microsecond tiap call)
 *   - 1000 × cudaMemcpy H2D (64 MB × 1000 = 64 GB transfer PCIe!)
 *   - 1000 × cudaMemcpy D2H (64 GB transfer lagi!)
 *   - 1000 × cudaFree
 * Total PCIe transfer: ~128 GB untuk sesuatu yang seharusnya 128 MB.
 * ============================================================ */
void simulate_buggy(float *h_grid, int width, int height) {
    size_t size = (size_t)width * height * sizeof(float);

    dim3 blockDim(16, 16);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    printf("[simulate_buggy] Memulai %d timestep...\n", TIMESTEPS);

    for (int t = 0; t < TIMESTEPS; t++) {
        float *d_old, *d_new;

        /* BUG: Alokasi dan transfer di DALAM loop — sangat boros! */
        CUDA_CHECK(cudaMalloc(&d_old, size));
        CUDA_CHECK(cudaMalloc(&d_new, size));
        CUDA_CHECK(cudaMemcpy(d_old, h_grid, size, cudaMemcpyHostToDevice));

        update_wave<<<gridDim, blockDim>>>(d_new, d_old, width, height, C_WAVE);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemcpy(h_grid, d_new, size, cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaFree(d_old));
        CUDA_CHECK(cudaFree(d_new));

        if (t % 100 == 0)
            printf("  Timestep %d/%d selesai\n", t, TIMESTEPS);
    }
}

/* ============================================================
 * [AREA UNTUK ANDA LENGKAPI]
 * simulate_optimized — Versi tanpa excessive transfer
 *
 * Prinsip:
 *   1. Alokasi d_old dan d_new DI LUAR loop (sekali saja)
 *   2. Transfer h_grid -> d_old SEKALI di awal
 *   3. Di setiap timestep: jalankan kernel, lalu swap pointer
 *   4. Transfer d_old -> h_grid SEKALI di akhir
 *   5. Bebaskan memori GPU
 * ============================================================ */
void simulate_optimized(float *h_grid, int width, int height) {
    /* TODO: Implementasikan di sini */
    printf("[simulate_optimized] Belum diimplementasikan — ini tugas Anda!\n");
}

/* ============================================================
 * Inisialisasi grid: gelombang Gaussian di tengah
 * ============================================================ */
void init_grid(float *grid, int width, int height) {
    float cx = width  / 2.0f;
    float cy = height / 2.0f;
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            float dx = (float)i - cx;
            float dy = (float)j - cy;
            float r2 = dx * dx + dy * dy;
            grid[j * width + i] = expf(-r2 / 1000.0f);
        }
    }
}

int main(void) {
    printf("=== Modul Simulasi Perambatan Gelombang Tsunami ===\n");
    printf("Grid     : %d × %d\n", GRID_X, GRID_Y);
    printf("Timestep : %d\n", TIMESTEPS);
    printf("===================================================\n\n");

    size_t size = (size_t)GRID_X * GRID_Y * sizeof(float);
    float *h_grid = (float *)malloc(size);
    if (!h_grid) {
        fprintf(stderr, "GAGAL: Tidak cukup memori host\n");
        return 1;
    }

    /* Print GPU info */
    int dev;
    cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop, dev);
    printf("GPU      : %s\n", prop.name);
    printf("VRAM     : %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("\n");

    /* Inisialisasi */
    init_grid(h_grid, GRID_X, GRID_Y);

    /* Jalankan versi bermasalah */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    simulate_buggy(h_grid, GRID_X, GRID_Y);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_buggy;
    CUDA_CHECK(cudaEventElapsedTime(&ms_buggy, start, stop));
    printf("\n[simulate_buggy]    Total waktu : %.2f detik\n", ms_buggy / 1000.0f);

    /* Reset dan jalankan versi optimal */
    init_grid(h_grid, GRID_X, GRID_Y);
    CUDA_CHECK(cudaEventRecord(start));
    simulate_optimized(h_grid, GRID_X, GRID_Y);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_opt;
    CUDA_CHECK(cudaEventElapsedTime(&ms_opt, start, stop));
    printf("[simulate_optimized] Total waktu: %.2f detik\n", ms_opt / 1000.0f);

    if (ms_opt > 0.0f)
        printf("Speedup  : %.2f×\n", ms_buggy / ms_opt);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_grid);
    return 0;
}
