#if defined(KTRANSFORMERS_USE_NPU) && KTRANSFORMERS_USE_NPU
    // 使用 x86 版本
    #include "iqk_mul_mat_arm.inc"
#else
    // 使用 ARM 版本
    #include "iqk_mul_mat_x86.inc"
#endif