extern "C" __global__
void cu_hysteresis_low(float* final_img, float* edge_img, float* strong_edge_mask,
                       float* weak_edge_mask, float t_low, int img_height, int img_width)
{
    int n, s, e, w;
    int nw, ne, sw, se;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (weak_edge_mask[idx] > 0)
    {
        n = idx - img_width;
        nw = n - 1;
        ne = n + 1;
        s = idx + img_width;
        sw = s - 1;
        se = s + 1;
        w = idx - 1;
        e = idx + 1;
        if (strong_edge_mask[nw] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[n] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[ne] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[w] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[e] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[sw] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[s] > 0) {
            final_img[idx] = 1;
        }
        if (strong_edge_mask[se] > 0) {
            final_img[idx] = 1;
        }
    }
    // if ((idx > img_width)                               /* skip first row */
    //     && (idx < (img_height * img_width) - img_width) /* skip last row */
    //     && ((idx % img_width) < (img_width - 1))        /* skip last column */
    //     && ((idx % img_width) > 0))                     /* skip first column */
    // {
    //     if (strong_edge_mask[idx]>0) { /* if this pixel was previously found to be a strong edge */
    //         // get indices
    //         n = idx - img_width;
    //         nw = n - 1;
    //         ne = n + 1;
    //         s = idx + img_width;
    //         sw = s - 1;
    //         se = s + 1;
    //         w = idx - 1;
    //         e = idx + 1;

    //         if (edge_img[nw] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[n] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[ne] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[w] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[e] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[sw] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[s] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //         if (edge_img[se] >= t_low) {
    //             final_img[idx] = 1;
    //         }
    //     }//end if(1 == strong_edge_mask[idx])
    // }//end if ((idx > img_width)
}