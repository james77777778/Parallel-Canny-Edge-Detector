extern "C" __global__
void cu_suppress_non_max(float* mag, float* deltaX, float* deltaY, float* nms,
                         long parser_length, long offset)
{
    const int SUPPRESSED = 0;
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 0 && idx < parser_length * offset)
    {
        float alpha;
        float mag1, mag2;
        // put zero all boundaries of image
        // TOP edge line of the image
        if((idx >= 0) && (idx <offset))
            nms[idx] = SUPPRESSED;
        // BOTTOM edge line of image
        else if((idx >= (parser_length-1)*offset) && (idx < (offset * parser_length)))
            nms[idx] = SUPPRESSED;
        // LEFT & RIGHT edge line
        else if(((idx % offset)==0) || ((idx % offset)==(offset - 1)))
        {
            nms[idx] = SUPPRESSED;
        }
        else // not the boundaries
        {
            // if magnitude = 0, no edge
            if(mag[idx] == 0)
                nms[idx] = SUPPRESSED;
            else{
                if(deltaX[idx] >= 0)
                {
                    if(deltaY[idx] >= 0)  // dx >= 0, dy >= 0
                    {
                        if((deltaX[idx] - deltaY[idx]) >= 0)       // direction 1 (SEE, South-East-East)
                        {
                            alpha = (float)deltaY[idx] / deltaX[idx];
                            mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                            mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                        }
                        else                                // direction 2 (SSE)
                        {
                            alpha = (float)deltaX[idx] / deltaY[idx];
                            mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                            mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                        }
                    }
                    else  // dx >= 0, dy < 0
                    {
                        if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 8 (NEE)
                        {
                            alpha = (float)-deltaY[idx] / deltaX[idx];
                            mag1 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                            mag2 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                        }
                        else                                // direction 7 (NNE)
                        {
                            alpha = (float)deltaX[idx] / -deltaY[idx];
                            mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                            mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                        }
                    }
                }
                else
                {
                    if(deltaY[idx] >= 0) // dx < 0, dy >= 0
                    {
                        if((deltaX[idx] + deltaY[idx]) >= 0)    // direction 3 (SSW)
                        {
                            alpha = (float)-deltaX[idx] / deltaY[idx];
                            mag1 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset-1];
                            mag2 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset+1];
                        }
                        else                                // direction 4 (SWW)
                        {
                            alpha = (float)deltaY[idx] / -deltaX[idx];
                            mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx+offset-1];
                            mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx-offset+1];
                        }
                    }
                    else // dx < 0, dy < 0
                    {
                            if((-deltaX[idx] + deltaY[idx]) >= 0)   // direction 5 (NWW)
                            {
                                alpha = (float)deltaY[idx] / deltaX[idx];
                                mag1 = (1-alpha)*mag[idx-1] + alpha*mag[idx-offset-1];
                                mag2 = (1-alpha)*mag[idx+1] + alpha*mag[idx+offset+1];
                            }
                            else                                // direction 6 (NNW)
                            {
                                alpha = (float)deltaX[idx] / deltaY[idx];
                                mag1 = (1-alpha)*mag[idx-offset] + alpha*mag[idx-offset-1];
                                mag2 = (1-alpha)*mag[idx+offset] + alpha*mag[idx+offset+1];
                            }
                    }
                }
                // non-maximal suppression
                // compare mag1, mag2 and mag[t]
                // if mag[t] is smaller than one of the neighbours then suppress it
                if((mag[idx] < mag1) || (mag[idx] < mag2))
                        nms[idx] = SUPPRESSED;
                else
                {
                        nms[idx] = mag[idx];
                }
            } // END OF ELSE (mag != 0)
        } // END OF FOR(j)
    } // END OF FOR(i)
}