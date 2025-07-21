#if defined(KTRANSFORMERS_USE_NPU) && KTRANSFORMERS_USE_NPU
    // 使用 x86 版本
    #include "sgemm_arm.cpp"
#else
    // 使用 ARM 版本
    #include "sgemm_x86.cpp"
#endif