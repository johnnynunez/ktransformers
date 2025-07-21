#if defined(KTRANSFORMERS_USE_NPU) && KTRANSFORMERS_USE_NPU
    // 使用 x86 版本
    #include "tinyblas_cpu_sgemm_arm.inc"
#else
    // 使用 ARM 版本
    #include "tinyblas_cpu_sgemm_x86.inc"
#endif