python setup.py build_ext --inplace
nv-nsight-cu-cli -o mec_conv_profile_custom_kernel --force-overwrite --target-processes all -k "fused_im2col_gemm_kernel_v2" python test.py