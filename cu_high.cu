extern "C" __global__
void cu_high(float* final_img, float* edge_img, float* strong_edge_mask,
                        float t_high, int img_height, int img_width)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (img_height * img_width)) {
        // apply high threshold
        if (edge_img[idx] > t_high) {
            strong_edge_mask[idx] = 1;
            final_img[idx] = 1;
        } else {
            strong_edge_mask[idx] = 0;
            final_img[idx] = 0;
        }
    }
}