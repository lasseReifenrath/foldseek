#include "LoLAlign.h"
#include "Fwbw.h"
#include "Debug.h"
#include "DBReader.h"
#include "DBWriter.h"
#include "IndexReader.h"
#include "SubstitutionMatrix.h"
#include "Matcher.h"
#include "Util.h"
#include "LocalParameters.h"
#include "Coordinate16.h"

#include <iostream>
#include <algorithm>
#include <limits>
#include <new>
#include <cstdlib>
#include <cfloat>
#include <numeric>
#include <cmath>
#include <vector>
#include <fstream>


#ifdef OPENMP
#include <omp.h>
#endif

// The matrices are allocated for the longest sequence. Rows of the current length keep an alignment
// in as few pages and cache lines as possible instead of spreading it over maxSeqLen^2 floats.
static void setRowLength(float **matrix, int rows, int cols) {
    const size_t stride = ((size_t)cols + VECSIZE_FLOAT - 1) / VECSIZE_FLOAT * VECSIZE_FLOAT;
    for (int i = 1; i < rows; ++i) {
        matrix[i] = matrix[0] + i * stride;
    }
}

// Largest value in row[begin, end), or 0 if none is larger
static float rowMax(const float *row, int begin, int end) {
    simd_float vmax = simdf32_set(0.0f);
    int j = begin;
    for (; j + VECSIZE_FLOAT <= end; j += VECSIZE_FLOAT) {
        vmax = simdf32_max(simdf32_loadu(row + j), vmax);
    }
    float max = simdf32_hmax(vmax);
    for (; j < end; ++j) {
        max = std::max(max, row[j]);
    }
    return max;
}

// Last column in row[begin, end) holding value, or begin - 1 if none
static int lastColOf(const float *row, int begin, int end, float value) {
    const simd_float vvalue = simdf32_set(value);
    int j = end;
    for (; j - VECSIZE_FLOAT >= begin; j -= VECSIZE_FLOAT) {
        const unsigned int mask = simdf32_movemask_ps(simdf32_eq(simdf32_loadu(row + j - VECSIZE_FLOAT), vvalue));
        if (mask != 0) {
            return j - VECSIZE_FLOAT + 31 - __builtin_clz(mask);
        }
    }
    for (--j; j >= begin && row[j] != value; --j) {}
    return j;
}

LoLAlign::LoLAlign(unsigned int maxSeqLen, bool computeExactScore, int candidateSeeds, int refineSeeds)
        : xtm(maxSeqLen), ytm(maxSeqLen), xt(maxSeqLen),
          r1(maxSeqLen), r2(maxSeqLen), computeExactScore(computeExactScore)
{
    seedNumber = refineSeeds;
    this->candidateSeeds = candidateSeeds;
    numStartAnchors = std::max(candidateSeeds, 2 * seedNumber);
    StartAnchorNum = candidateSeeds;
    queryX = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    queryY = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    queryZ = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    targetDistRows.assign(maxSeqLen, nullptr);
    queryNbCursor.assign(maxSeqLen, 0);
    firstAnchorAt.assign(maxSeqLen, -1);
    nextAnchorAt.assign(maxSeqLen, -1);
    rowMaxP.assign(maxSeqLen, 0.0f);
    scoreRowIteration.assign(maxSeqLen, -1);
    logSeparation.resize(maxSeqLen);
    for (unsigned int k = 0; k < maxSeqLen; k++) {
        logSeparation[k] = std::log(1.0f + (float)k);
    }
    distancekl = malloc_matrix<float>(maxSeqLen, maxSeqLen);
    lolScoreMatrix = malloc_matrix<float>(maxSeqLen, maxSeqLen);
    hiddenLayer = malloc_matrix<float>(maxSeqLen, 3);
    probabilityMatrix = malloc_matrix<float>(maxSeqLen, maxSeqLen);
    anchorQuery = malloc_matrix<int>(std::max(numStartAnchors, 2*seedNumber), maxSeqLen);
    anchorTarget = malloc_matrix<int>(std::max(numStartAnchors, 2*seedNumber), maxSeqLen);
    lolDist = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    lolSeqDist = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    lolScoreVec = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    lolScoreVecSh = (float*)mem_align(ALIGN_FLOAT, maxSeqLen * sizeof(float));
    
    startAnchorIndex = new int[numStartAnchors];
    startAnchorScores = new float[numStartAnchors];
    anchorLength = new int[numStartAnchors];
    newAnchorLength = new int[numStartAnchors];
    newAnchorQuery = new int[maxSeqLen];
    newAnchorTarget = new int[maxSeqLen];
    gaps = new int[4]{0, 0, 0, 0};


    // initQuery is called with query and target lengths up to maxSeqLen
    finalAnchorQuery = new int[maxSeqLen];
    finalAnchorTarget = new int[maxSeqLen];
    lddtcalc = new LDDTCalculator(maxSeqLen + 1, maxSeqLen + 1);
    //P = malloc_matrix<float>(maxSeqLen, maxSeqLen);
}



LoLAlign::~LoLAlign()
{
    free(queryX);
    free(queryY);
    free(queryZ);
    free(distancekl);
    free(probabilityMatrix);
    free(lolScoreMatrix);
    free(hiddenLayer);
    free(anchorQuery);
    free(anchorTarget);
    delete[] startAnchorIndex;
    delete[] startAnchorScores;
    delete[] anchorLength;
    delete[] newAnchorLength;
    delete[] newAnchorQuery;
    delete[] newAnchorTarget;
    delete[] gaps;
    free(lolDist);
    free(lolSeqDist);
    free(lolScoreVec);
    if (finalAnchorQuery != nullptr) {
        delete[] finalAnchorQuery;
    }
    if (finalAnchorTarget != nullptr) {
        delete[] finalAnchorTarget;
    }
    free(lolScoreVecSh);
    delete lddtcalc;
}

void LoLAlign::calcGap(int* anchorQuery, int* anchorTarget, int* gaps, int queryLen, int targetLen)
{
    gaps[0] = -1;
    int indexQ = gaps[1];
    int indexT = gaps[3];
    while (anchorQuery[indexQ] != 0 || anchorTarget[indexT] != 0) {
        if (anchorQuery[indexQ] != 0 && anchorTarget[indexT] != 0) {
            indexQ++;
            indexT++;
        } else if (anchorQuery[indexQ] == 0) {
            indexQ++;
        } else if (anchorTarget[indexT] == 0) {
            indexT++;
        }
        if (indexQ == queryLen || indexT == targetLen) {
            gaps[0] = -1;
            return;
        }
    }
    gaps[0] = indexQ;
    gaps[2] = indexT;
    while (anchorQuery[indexQ] == 0 || anchorTarget[indexT] == 0) {
        if (anchorQuery[indexQ] == 0 && anchorTarget[indexT] == 0) {
            indexQ++;
            indexT++;
        } else if (anchorQuery[indexQ] == 0) {
            indexQ++;
        } else if (anchorTarget[indexT] == 0) {
            indexT++;
        }
        if (indexQ == queryLen || indexT == targetLen) {
            indexQ = queryLen;
            indexT = targetLen;
            break;
        }
    }
    gaps[1] = indexQ;
    gaps[3] = indexT;
    return;

}
float LoLAlign::maxSubArray(float* nums, int numsSize) {

    float currentMax = nums[0];
    float globalMax = nums[0];

    for (int i = 1; i < numsSize; ++i) {
        currentMax = std::max(nums[i], currentMax + nums[i]);
        globalMax = std::max(globalMax, currentMax);
    }

    return globalMax;
}

void LoLAlign::indexSort(float* nums, int* index, int numsSize) {
    std::sort(index, index + numsSize, [&nums](int i1, int i2) { return nums[i1] < nums[i2]; });
}




Matcher::result_t LoLAlign::align(unsigned int dbKey, float* targetX, float* targetY, float* targetZ,
                                  Sequence& tSeqAA, Sequence& tSeq3Di, int targetLen, SubstitutionMatrix& subMatAA,
                                  FwBwAligner* fwbwaln, int multiDomain)
{
    maxLolMatrixIndex = 0;
    minLolMatrixIndex = queryLen;
    setRowLength(lolScoreMatrix, queryLen, targetLen);
    setRowLength(probabilityMatrix, queryLen, targetLen);
    targetCaX = targetX;
    targetCaY = targetY;
    targetCaZ = targetZ;
    targetCaLen = targetLen;
    targetDistStride = ((size_t)targetLen + VECSIZE_FLOAT - 1) / VECSIZE_FLOAT * VECSIZE_FLOAT;
    targetDistRowsUsed = 0;
    std::fill(targetDistRows.begin(), targetDistRows.begin() + targetLen, (float*)nullptr);


    unsigned char *targetNumAA = tSeqAA.numSequence;
    unsigned char *targetNum3Di = tSeq3Di.numSequence;
    const char *targetSeq = tSeqAA.getSeqData();
    computeForwardScoreMatrix(
            targetNumAA,
            targetNum3Di,
            queryLen,
            targetLen,
            subMatAA,
            lolScoreMatrix);

    int maxIndexX = 0;
    int maxIndexY = 0;

    
    for (int i = 0; i < numStartAnchors; i++) {
        startAnchorIndex[i] = i;
        startAnchorScores[i] = 0;
    }


    for (int startAnchor = 0; startAnchor < numStartAnchors; startAnchor++) {
        anchorLength[startAnchor] = 0;
        newAnchorLength[startAnchor] = 0;
        std::fill(anchorQuery[startAnchor], anchorQuery[startAnchor] + queryLen, 0);
        std::fill(anchorTarget[startAnchor], anchorTarget[startAnchor] + targetLen, 0);

    }


    gaps[0] = 0;
    gaps[1] = queryLen;
    gaps[2] = 0;
    gaps[3] = targetLen;

    fwbwaln->resetParams(startAnchorGo, startAnchorGe, startAnchorT);
    fwbwaln->initScoreMatrix(lolScoreMatrix, gaps, probabilityMatrix, rowMaxP.data());
    fwbwaln->runFwBw<false, 0>();

    // If the posterior overflowed to all-zeros (maxP == 0), the exponentials in the
    // forward-backward blew past float range. Raising the temperature shrinks the score
    // exponents back into range, so re-run the FwBw at a higher temperature until we recover a
    // usable posterior, giving up past temperature 30. This mirrors the refinement phase below.
    // initScoreMatrix must be re-called because it divides the score matrix by the temperature;
    // resetParams recomputes the gap terms exp(go/T), exp(ge/T) for the new temperature.
    while (fwbwaln->getMaxP() == 0 && fwbwaln->getTemperature() <= 30) {
        fwbwaln->resetParams(startAnchorGo, startAnchorGe, fwbwaln->getTemperature() + 2);
        fwbwaln->initScoreMatrix(lolScoreMatrix, gaps, probabilityMatrix, rowMaxP.data());
        fwbwaln->runFwBw<false, 0>();
    }

    // read the posterior directly, getProbability is a virtual call per cell
    float **fwbwP = fwbwaln->getProbabiltyMatrix();
    const float fwbwMaxP = fwbwaln->getMaxP();
    // row maxima of the seed search columns [startAnchorLength, targetLen - startAnchorLength)
    const int seedColBegin = startAnchorLength, seedColEnd = targetLen - startAnchorLength;
    if (fwbwMaxP != 0) {
        for (int i = startAnchorLength; i < queryLen - startAnchorLength; ++i) {
            // the row maximum is inside unless a left-out column reaches it
            float outside = 0.0f;
            for (int j = 0; j < std::min(seedColBegin, targetLen); ++j) {
                outside = std::max(outside, fwbwP[i][j]);
            }
            for (int j = std::max(seedColEnd, 0); j < targetLen; ++j) {
                outside = std::max(outside, fwbwP[i][j]);
            }
            if (!(seedColBegin < seedColEnd && rowMaxP[i] > outside)) {
                rowMaxP[i] = rowMax(fwbwP[i], seedColBegin, seedColEnd);
            }
        }
    }
    for (int startAnchor = 0; startAnchor < candidateSeeds; startAnchor++) {

        // If the temperature retry above could not recover a non-zero posterior, there is no
        // probability signal to seed an anchor from. Scanning the dead posterior would leave
        // maxIndexX/maxIndexY at the sentinel 0 and index out of bounds, so record an empty
        // anchor and skip the scan.
        if (fwbwMaxP == 0) {
            startAnchorScores[startAnchor] = 0;
            anchorLength[startAnchor] = 0;
            newAnchorLength[startAnchor] = 0;
            continue;
        }

        maxIndexX = 0;
        maxIndexY = 0;

        float maxScore = 0;

        if (startAnchor % 2 == 0) {

            for (int i = startAnchorLength; i < queryLen - startAnchorLength; ++i) {
                // ties go to the last column, as with a scalar >= scan
                const float max = rowMaxP[i];
                if (max >= maxScore) {
                    const int col = lastColOf(fwbwP[i], startAnchorLength, targetLen - startAnchorLength, max);
                    if (col >= startAnchorLength) {
                        maxScore = max;
                        maxIndexX = i;
                        maxIndexY = col;
                    }
                }
                if (maxScore >= fwbwMaxP){
                    break;
                }

            }
        }
        else{
            for (int i = queryLen - startAnchorLength -1; i >= startAnchorLength; --i) {
                // the first column a backwards > scan meets is also the last one holding the max
                const float max = rowMaxP[i];
                if (max > maxScore) {
                    const int col = lastColOf(fwbwP[i], startAnchorLength, targetLen - startAnchorLength, max);
                    if (col >= startAnchorLength) {
                        maxScore = max;
                        maxIndexX = i;
                        maxIndexY = col;
                    }
                }
                if (maxScore >= fwbwMaxP){
                    break;
                }
            }
        }





        int startRow = maxIndexX - std::min(maxIndexX, maxIndexY);
        int startCol = maxIndexY - std::min(maxIndexX, maxIndexY);
        int diagLength = std::min(queryLen - startRow, targetLen - startCol);
        for (int i = 0; i < diagLength; i++) {
            lolScoreVec[i] = lolScoreMatrix[startRow + i][startCol + i];
        }
        lolScoreVec[std::min(maxIndexX, maxIndexY)] += 200;
         
        for (int i = -startAnchorLength; i < startAnchorLength; i++) {
            // The neighborhood offset i can push maxIndexX/maxIndexY below 0 (or past the
            // sequence end) when the best seed lies within startAnchorLength of a terminus;
            // skip those out-of-range rows instead of indexing outside the distance matrices.
            if (maxIndexX + i < 0 || maxIndexX + i >= queryLen ||
                maxIndexY + i < 0 || maxIndexY + i >= targetLen) {
                continue;
            }
            // -1 unless residues maxIndexX+i and startRow+j are closer than 20 A
            std::fill(lolDist, lolDist + diagLength, -1.0f);
            std::fill(lolSeqDist, lolSeqDist + diagLength, -1.0f);
            const int rowEnd = queryNbStart[maxIndexX + i + 1];
            const float *targetRow = nullptr;
            for (int k = std::lower_bound(queryNbIdx.begin() + queryNbStart[maxIndexX + i], queryNbIdx.begin() + rowEnd, startRow) - queryNbIdx.begin();
                 k < rowEnd && queryNbIdx[k] < startRow + diagLength; k++) {
                const int j = queryNbIdx[k] - startRow;
                if (targetRow == nullptr) {
                    targetRow = targetDistRow(maxIndexY + i);
                }
                lolDist[j] = std::abs(queryNbDist[k] - targetRow[startCol + j]);
                // sequence separation of residues maxIndexX+i and startRow+j
                const int sep = (maxIndexX + i) - (startRow + j);
                lolSeqDist[j] = std::copysign(1.0f, sep) * logSeparation[std::abs(sep)];
            }

            lolScore(lolDist, lolSeqDist, lolScoreVec, diagLength);
        }
        startAnchorScores[startAnchor] = maxSubArray(lolScoreVec, diagLength);
        alignStartAnchors(anchorQuery[startAnchor], anchorTarget[startAnchor], maxIndexX, maxIndexY,
                          &newAnchorLength[startAnchor], fwbwP, lolScoreMatrix, targetLen);
        // alignStartAnchors zeroed P on the anchor diagonal
        for (int i = std::max(maxIndexX - startAnchorLength, startAnchorLength);
             i <= std::min(maxIndexX + startAnchorLength, queryLen - startAnchorLength - 1); ++i) {
            rowMaxP[i] = rowMax(fwbwP[i], seedColBegin, seedColEnd);
        }
        anchorLength[startAnchor] = newAnchorLength[startAnchor];
    }
    
    for (int i = 0; i < queryLen; ++i) {
        std::memset(lolScoreMatrix[i], 0, targetLen * sizeof(lolScoreMatrix[0][0]));
    }
    
    indexSort(startAnchorScores, startAnchorIndex, candidateSeeds);

    gaps[0] = 0;
    gaps[1] = 0;
    gaps[2] = 0;
    gaps[3] = 0;

    fwbwaln->resetParams(lolGo, lolGe, lolT);
    int startAnchor;

    int refineIter = std::min(seedNumber, candidateSeeds);
    for (int startAnchorIter = 0; startAnchorIter < refineIter; startAnchorIter++) {
        startAnchor = startAnchorIndex[candidateSeeds - startAnchorIter - 1];
        bool addSeq = false;
        // lolScoreMatrix starts from 0 again
        fwbwGapsPrev.clear();
        std::fill(scoreRowIteration.begin(), scoreRowIteration.begin() + queryLen, -1);

        for(int iteration = 0; iteration < 1000; iteration++){
            lolIteration = iteration;
            gaps[0] = 0;
            gaps[1] = 0;
            gaps[2] = 0;
            gaps[3] = 0;


            if (iteration == 0) {
                // seed anchors: one diagonal, so the k-th new query and target positions pair up
                int n = 0, iq = 0, it = 0;
                while (n < newAnchorLength[startAnchor]) {
                    while (iq < queryLen && anchorQuery[startAnchor][iq] != 2) iq++;
                    while (it < targetLen && anchorTarget[startAnchor][it] != 2) it++;
                    if (iq == queryLen || it == targetLen) break;
                    newAnchorQuery[n] = iq++;
                    newAnchorTarget[n] = it++;
                    n++;
                }
                newAnchorLength[startAnchor] = n;
            }
            // later iterations: newAnchorQuery/newAnchorTarget were filled by the anchor selection

            for (int i = 0; i < newAnchorLength[startAnchor]; i++) {
                queryNbCursor[i] = queryNbStart[newAnchorQuery[i]];
            }
            while((gaps[1] < queryLen && gaps[3] < targetLen)){

                calcGap(anchorQuery[startAnchor], anchorTarget[startAnchor], gaps, queryLen, targetLen);
                

                if (gaps[0] != -1){
                    lolMatrix(newAnchorQuery, newAnchorTarget, newAnchorLength[startAnchor], gaps, lolScoreMatrix);
                }else{
                    break;
                }
            }

            for (int i = 0; i < queryLen; i++) {
                if (anchorQuery[startAnchor][i] == 2) {
                    anchorQuery[startAnchor][i] = 1;
                }
            }
            for (int i = 0; i < targetLen; i++) {
                if (anchorTarget[startAnchor][i] == 2) {
                    anchorTarget[startAnchor][i] = 1;
                }
            }
            newAnchorLength[startAnchor] = 0;

            gaps[0] = minLolMatrixIndex;
            gaps[1] = 0;
            gaps[2] = 0;
            gaps[3] = 0;
            float maxP = 0;

            if(iteration == 0){
                maxP = 0.5;
            }
            else{
                maxP = lolMinP;
            }
            // false if gaps were left out, their P are stale
            bool rowMaxValid = true;
            fwbwGapsCur.clear();
            size_t prevGap = 0;
            while (gaps[1] < maxLolMatrixIndex && gaps[3] < targetLen) {
                calcGap(anchorQuery[startAnchor], anchorTarget[startAnchor], gaps, maxLolMatrixIndex, targetLen);
                if(gaps[0] != -1){
                    // same bounds, temperature and score rows as in the previous iteration: P is still there
                    const float temperature = fwbwaln->getTemperature();
                    bool reused = false;
                    float gapMaxP = 0.0f;
                    while (prevGap < fwbwGapsPrev.size() && fwbwGapsPrev[prevGap].rowStart < gaps[0]) {
                        prevGap++;
                    }
                    if (prevGap < fwbwGapsPrev.size()) {
                        const FwbwGap &g = fwbwGapsPrev[prevGap];
                        if (g.rowStart == gaps[0] && g.rowEnd == gaps[1] && g.colStart == gaps[2] && g.colEnd == gaps[3]
                            && g.temperature == temperature) {
                            reused = true;
                            for (int r = gaps[0]; r < gaps[1] && reused; r++) {
                                reused = (scoreRowIteration[r] < iteration);
                            }
                            gapMaxP = g.maxP;
                        }
                    }
                    if (!reused) {
                        fwbwaln->initScoreMatrix(lolScoreMatrix, gaps, probabilityMatrix, rowMaxP.data());
                        fwbwaln->runFwBw<false, 0>();
                        gapMaxP = fwbwaln->getMaxP();
                    }
                    fwbwGapsCur.push_back(FwbwGap{gaps[0], gaps[1], gaps[2], gaps[3], temperature, gapMaxP});
                    maxP = std::max(maxP, gapMaxP);
                    if(gapMaxP == 0){
                        if (fwbwaln->getTemperature() > 30){
                            fwbwaln->resetParams(lolGo, lolGe, lolT);
                            rowMaxValid = false;
                            break;
                        }
                       
                        fwbwaln->resetParams(lolGo, lolGe, fwbwaln->getTemperature() + 2);
                        fwbwGapsCur.clear();
                        prevGap = 0;
                        gaps[0] = 0;
                        gaps[1] = 0;
                        gaps[2] = 0;
                        gaps[3] = 0;
                        continue; 
                        
                    }
                }
                else{
                    break;
                }         
            }
            fwbwGapsPrev.swap(fwbwGapsCur);
            if (fwbwaln->getTemperature() != lolT) {
                fwbwaln->resetParams(lolGo, lolGe, lolT);
            }
            newAnchorLength[startAnchor] = 0;

            gaps[0] = minLolMatrixIndex;
            gaps[1] = 0;
            gaps[2] = 0;
            gaps[3] = 0;


            while (gaps[1] < maxLolMatrixIndex && gaps[3] < targetLen) {
                calcGap(anchorQuery[startAnchor], anchorTarget[startAnchor], gaps, maxLolMatrixIndex, targetLen);
                if (gaps[0] != -1) {
                    for (int i = gaps[0]; i < gaps[1]; i++) {
                        // no P of the row passes the threshold
                        if (rowMaxValid && !(rowMaxP[i] > maxP - 0.1)) {
                            continue;
                        }
                        for (int j = gaps[2]; j < gaps[3]; j++) {
                            if (probabilityMatrix[i][j] > maxP - 0.1) {
                                if (anchorQuery[startAnchor][i] == 0 && anchorTarget[startAnchor][j] == 0) {
                                    anchorQuery[startAnchor][i] = 2;
                                    anchorTarget[startAnchor][j] = 2;
                                    newAnchorQuery[newAnchorLength[startAnchor]] = i;
                                    newAnchorTarget[newAnchorLength[startAnchor]] = j;
                                    anchorLength[startAnchor] += 1;
                                    newAnchorLength[startAnchor] += 1;
                                    break;
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }

            if (newAnchorLength[startAnchor] == 0) {
                if (!addSeq && multiDomain == 1) {
                    addSeq = true;
                    minLolMatrixIndex = 0;
                    maxLolMatrixIndex = queryLen;
                    // every score row changes
                    fwbwGapsPrev.clear();
                    addForwardScoreMatrix(
                        targetNumAA,
                        targetNum3Di,
                        queryLen,
                        targetLen,
                        subMatAA,
                        lolScoreMatrix);
                    int tempAnchorIdx = startAnchorIndex[numStartAnchors - startAnchorIter - 1 - seedNumber];
                    anchorLength[tempAnchorIdx] = anchorLength[startAnchor];
                    for(int i = 0; i<queryLen; i++){
                        if(anchorQuery[startAnchor][i]!= 0){
                            anchorQuery[tempAnchorIdx][i] = 1;
                        }
                        else{
                            anchorQuery[tempAnchorIdx][i] = 0;
                        }
                    }
                    for(int i = 0; i<targetLen; i++){
                        if(anchorTarget[startAnchor][i]!= 0){
                            anchorTarget[tempAnchorIdx][i] = 1;
                        }
                        else{
                            anchorTarget[tempAnchorIdx][i] = 0;
                        }
                    }
                }
                else{
                    break;
                }

            }
        }

        for (int i = 0; i < queryLen; ++i) {
            std::memset(lolScoreMatrix[i], 0, targetLen * sizeof(lolScoreMatrix[0][0]));
        }

    }

    float maxLolScore = std::numeric_limits<float>::min();
    int maxLolIdx = 0;
    int seedIter = seedNumber;
    if(multiDomain){
        seedIter *= 2;
    }

    
    for (int startAnchorIter = 0; startAnchorIter < seedIter; startAnchorIter++){
        
        startAnchor = startAnchorIndex[numStartAnchors - startAnchorIter -1];
        int startAnchorIndex = 0;
        for(int i = 0; i < queryLen; i++){
            if(anchorQuery[startAnchor][i] != 0){
                finalAnchorQuery[startAnchorIndex] = i;
                startAnchorIndex++;
               
            }
        }
        startAnchorIndex = 0;
        for(int i = 0; i < targetLen; i++){
            if(anchorTarget[startAnchor][i] != 0){
                finalAnchorTarget[startAnchorIndex] = i;
                startAnchorIndex++;
            }
        }


        computeDiScore(targetNumAA, targetNum3Di, anchorLength[startAnchor], finalAnchorQuery, finalAnchorTarget, subMatAA, lolScoreVec);

        linkAnchors(anchorLength[startAnchor]);
        for (int i = 0; i < anchorLength[startAnchor]; i++) {
            anchorRowScore(i, true, lolScoreVec);
        }
        unlinkAnchors(anchorLength[startAnchor]);

        float totalLolScore = 0;
        for (int i = 0; i < anchorLength[startAnchor]; i++) {
            totalLolScore += lolScoreVec[i];

        }
        if (totalLolScore > maxLolScore){
            maxLolScore = totalLolScore;
            maxLolIdx = startAnchor;
        }
    }
    float seqId = 0.0;

    int startAnchorIndex = 0;
    for(int i = 0; i < queryLen; i++){
        if(anchorQuery[maxLolIdx][i] != 0){
            finalAnchorQuery[startAnchorIndex] = i;
            startAnchorIndex++;            
        }
    }
    startAnchorIndex = 0;
    for(int i = 0; i < targetLen; i++){
        if(anchorTarget[maxLolIdx][i] != 0){
            finalAnchorTarget[startAnchorIndex] = i;
            startAnchorIndex++;
        }
    }
    
    computeDiScore(targetNumAA, targetNum3Di, anchorLength[maxLolIdx], finalAnchorQuery, finalAnchorTarget, subMatAA, lolScoreVec);
    float maxAA3DiScore = 0;
    for (int i = 0; i < anchorLength[maxLolIdx]; i++) {
        maxAA3DiScore += lolScoreVec[i];
    } 

    for(int i = 0; i < queryLen; i++){
        lolScoreVecSh[i] = 0;
        lolScoreVec[i] = 0;
        
    }
    linkAnchors(anchorLength[maxLolIdx]);
    for (int i = 0; i < anchorLength[maxLolIdx]; i++) {
        anchorRowScore(i, false, lolScoreVecSh);
    }
    for (int i = 0; i < anchorLength[maxLolIdx]; i++) {
        anchorRowScore(i, true, lolScoreVec);
    }
    unlinkAnchors(anchorLength[maxLolIdx]);

    float normalizedLolscoreSelfhit = 0;
    float nanCheck = 0;
    maxLolScore = 0;
    for (int i = 0; i < anchorLength[maxLolIdx]; i++) {
        maxLolScore += lolScoreVec[i];
        if(lolScoreVecSh[i] != 0){
            nanCheck = lolScoreVec[i] / (lolScoreVecSh[i]);
            normalizedLolscoreSelfhit +=  nanCheck == nanCheck ? nanCheck : 0;
        }
        
    }

    backtrace.clear();
    int matches = 0;
    int queryCount = 0;
    int targetCount = 0;
    
    while(matches < anchorLength[maxLolIdx]){
        if(anchorQuery[maxLolIdx][queryCount] != 0 && anchorTarget[maxLolIdx][targetCount] != 0){
            backtrace.append(1, 'M');
            matches++;
            if(querySeq[queryCount] == targetSeq[targetCount]){
                seqId += 1.0;
            }
            queryCount++;
            targetCount++;
        }
        else if(anchorTarget[maxLolIdx][targetCount] == 0){
            backtrace.append(1, 'D');
            targetCount++;
        }
        else if(anchorQuery[maxLolIdx][queryCount] == 0){
            backtrace.append(1, 'I');
            queryCount++;
        }

    }

    Matcher::result_t result = Matcher::result_t();
    result.seqId = seqId / (float)anchorLength[maxLolIdx];
    result.qcov = anchorLength[maxLolIdx] / (float)queryLen;
    result.dbcov = anchorLength[maxLolIdx] / (float)targetLen;
    // eval: sqrt of the qqScore-normalized LoL score, clipped to >=0.
    // Length normalization /(qLen*tLen)^0.25 applies only in single-domain mode
    // (multiDomain==0); in multi-domain mode it is intentionally omitted, matching
    // the pre-existing branching.
    // score: filled in below with the LoLAlign homology probability * 10000.
    float lolEvalRaw = ((maxLolScore + 3.0f * maxAA3DiScore) * normalizedLolscoreSelfhit
                       / (float)anchorLength[maxLolIdx]) / qqScore;
    if (multiDomain == 0) {
        lolEvalRaw /= std::pow(queryLen * targetLen, 0.25);
    }
    result.eval  = std::sqrt(std::max(0.0f, lolEvalRaw));
    result.score = 0;
    result.dbKey = dbKey;
    result.qStartPos = 0;
    result.dbStartPos = 0;
    result.qEndPos = queryLen - 1;
    result.dbEndPos = targetLen - 1;
    result.qLen = queryLen;
    result.dbLen = targetLen;
    result.alnLength = backtrace.length();
    result.backtrace = Matcher::compressAlignment(backtrace);

    
    // trim backtrace find the first 'M'
    size_t firstM = targetLen;
    int qLeftSkip = 0;
    int tLeftSkip = 0;
    for (size_t i = 0; i < backtrace.size(); i++) {
        if (backtrace[i] == 'M') {
            firstM = i;
            break;
        }
        if (backtrace[i] == 'I') {
            qLeftSkip++;
        } else if (backtrace[i] == 'D') {
            tLeftSkip++;
        }
    }

    result.qStartPos  += qLeftSkip;
    result.dbStartPos += tLeftSkip;
    result.qEndPos = result.qStartPos;
    result.dbEndPos = result.dbStartPos;
    for (size_t i = firstM; i < backtrace.size(); i++) {
        switch (backtrace[i]) {
            case 'M': // match consumes 1 base in query + 1 base in target
                result.qEndPos++;
                result.dbEndPos++;
                break;
            case 'I': // insertion consumes 1 base in query
                result.qEndPos++;
                break;
            case 'D': // deletion consumes 1 base in target
                result.dbEndPos++;
                break;
            default:
                // handle unexpected cigar chars if needed
                break;
        }
    }


    result.qEndPos--;
    result.dbEndPos--;
    backtrace.erase(0, firstM);
    result.backtrace = Matcher::compressAlignment(backtrace);
    result.alnLength = (int)result.backtrace.size();

    // ---- LoLAlign homology probability, written into result.score * 10000 -----
    {
        const std::string &aln = backtrace;                  // uncompressed M/I/D, aligned region
        qAln.clear();
        tAln.clear();
        int qpos = result.qStartPos, tpos = result.dbStartPos;
        int alnlen = 0, gapopen = 0, numGaps = 0;
        char prevOp = 0;
        for (size_t i = 0; i < aln.size(); ++i) {
            char op = aln[i];
            ++alnlen;
            if (op == 'M') {
                qAln.push_back(qpos++); tAln.push_back(tpos++);
            } else {
                ++numGaps;
                if (op != prevOp) ++gapopen;                 // gap-run count (= compressed-cigar token)
                if (op == 'I') ++qpos; else if (op == 'D') ++tpos;
            }
            prevOp = op;
        }
        double rawBits = (double)maxLolScore + 3.0 * (double)maxAA3DiScore;   // 'bits' feature (raw LoL score)
        double rawEval = (double)normalizedLolscoreSelfhit;                   // 'evalue' feature (self-hit normalized)
        double lddt = lddtcalc->computeLDDTScore(targetLen, result.qStartPos, result.dbStartPos,
                                                 aln, targetX, targetY, targetZ).avgLddtScore;
        float prob = lolProb.predict(targetLen, targetX, targetY, targetZ,
                                     qAln.data(), tAln.data(), (int)qAln.size(),
                                     rawEval, rawBits, lddt, alnlen, gapopen, numGaps);
        // Homology probability lives in score, scaled by 10000 to survive int truncation.
        result.score = static_cast<int>(prob * 10000.0f + 0.5f);
    }

    return result;

}
void LoLAlign::alignStartAnchors(int* anchorQuery, int* anchorTarget, int maxQuery, int maxTarget,
                                 int* anchorLength, float** fwbwP, float** lolScoreMatrix, int targetLen) {

    for (int i = maxQuery - this->startAnchorLength; i <= maxQuery + this->startAnchorLength; ++i) {
        int maxInd = maxTarget + i - maxQuery;

        // The anchor diagonal can extend past either terminus when the seed lies within
        // startAnchorLength of a sequence end; skip out-of-range cells rather than indexing
        // outside anchorQuery/anchorTarget and the probability/score matrices.
        if (i < 0 || i >= this->queryLen || maxInd < 0 || maxInd >= targetLen) {
            continue;
        }

        anchorQuery[i] = 2;
        anchorTarget[maxInd] = 2;
        fwbwP[i][maxInd] = 0;
        lolScoreMatrix[i][maxInd] = 0;
        *anchorLength = *anchorLength + 1;
    }
}

// row r of the target distance matrix, computed once per pair when first needed
const float* LoLAlign::targetDistRow(int r) {
    if (targetDistRows[r] == nullptr) {
        float *row = distancekl[0] + (size_t)targetDistRowsUsed * targetDistStride;
        targetDistRowsUsed++;
        for (int j = 0; j < targetCaLen; ++j) {
            float dx = targetCaX[r] - targetCaX[j];
            float dy = targetCaY[r] - targetCaY[j];
            float dz = targetCaZ[r] - targetCaZ[j];

            float distSq = dx * dx + dy * dy + dz * dz;

            row[j] = std::sqrt(distSq);
        }
        targetDistRows[r] = row;
    }
    return targetDistRows[r];
}

// the query residues closer than 20 A to each residue (distance > 0) and their distances, as neighbor lists
void LoLAlign::calcQueryNeighbors(const float* x, const float* y, const float* z, int len) {
    const float cutoffDistance = 20.0f;
    const float cutoffSq = cutoffDistance * cutoffDistance;
    queryNbStart.resize(len + 1);
    queryNbIdx.clear();
    queryNbDist.clear();
    for (int i = 0; i < len; ++i) {
        queryNbStart[i] = (int)queryNbIdx.size();
        for (int j = 0; j < len; ++j) {
            float dx = x[i] - x[j];
            float dy = y[i] - y[j];
            float dz = z[i] - z[j];

            float distSq = dx * dx + dy * dy + dz * dz;

            if (!(distSq > cutoffSq)) {
                float dist = std::sqrt(distSq);
                if (dist > 0) {
                    queryNbIdx.push_back(j);
                    queryNbDist.push_back(dist);
                }
            }
        }
    }
    queryNbStart[len] = (int)queryNbIdx.size();
}

// firstAnchorAt[q]: first final anchor at query position q, nextAnchorAt[j]: the one after anchor j
void LoLAlign::linkAnchors(int anchorLen) {
    for (int j = anchorLen - 1; j >= 0; j--) {
        const int q = finalAnchorQuery[j];
        nextAnchorAt[j] = firstAnchorAt[q];
        firstAnchorAt[q] = j;
    }
}

void LoLAlign::unlinkAnchors(int anchorLen) {
    for (int j = 0; j < anchorLen; j++) {
        firstAnchorAt[finalAnchorQuery[j]] = -1;
        nextAnchorAt[j] = -1;
    }
}

// lolScore of anchor i against the anchors closer than 20 A in the query, the others add nothing
void LoLAlign::anchorRowScore(int i, bool withTargetDist, float* score) {
    const int qi = finalAnchorQuery[i];
    const float *targetRow = nullptr;
    for (int k = queryNbStart[qi]; k < queryNbStart[qi + 1]; k++) {
        for (int j = firstAnchorAt[queryNbIdx[k]]; j >= 0; j = nextAnchorAt[j]) {
            if (withTargetDist && targetRow == nullptr) {
                targetRow = targetDistRow(finalAnchorTarget[i]);
            }
            const float dist = withTargetDist ? std::abs(queryNbDist[k] - targetRow[finalAnchorTarget[j]]) : 0.0f;
            const int sep = finalAnchorQuery[i] - finalAnchorQuery[j];
            const float dSeq = std::copysign(1.0f, sep) * logSeparation[std::abs(sep)];
            if (dist >= 0) {
                float s = score[j];
                for (int m = 0; m < 3; ++m) {
                    float h = dSeq * w1[0][m];
                    h += dist * w1[1][m];
                    h += b1[m];
                    h = std::max(0.0f, h);
                    s += h * w2[m];
                }
                s += b2;
                score[j] = s;
            }
        }
    }
}

void LoLAlign::computeDiScore(
        unsigned char* targetNumAA,
        unsigned char* targetNum3Di,
        int anchorLen,
        int* finalAnchorQuery,
        int* finalAnchorTarget,
        SubstitutionMatrix& subMatAA,
        float* scoreForward)
{


    for (int i = 0; i < anchorLen; ++i) {
        int q = finalAnchorQuery[i];
        int t = finalAnchorTarget[i];
        scoreForward[i] = static_cast<float>((subMatAA.subMatrix[queryNumAA[q]][targetNumAA[t]] * 1.4)
                                             + (scoringMatrix3Di[queryNum3Di[q]][targetNum3Di[t]] * 2.1));
    }
}

// forward score tables of the query, filled by the loops they replace. Out of line: inside initQuery, GCC inlines
// computeDiScore differently and fuses its multiply-adds differently, which changes qqScore in the last bits
__attribute__((noinline))
void LoLAlign::initForwardTables(const SubstitutionMatrix& subMatAA) {
    const int alphabetAA = subMatAA.alphabetSize;
    fwdCodeCount = alphabetAA * ALPHABET_SIZE_3DI;
    std::vector<int> rowOfCode(fwdCodeCount, -1);
    std::vector<unsigned char> rowAA, row3Di;
    fwdQueryRow.resize(queryLen);
    fwdTable.clear();
    fwdAddTable.clear();
    bool tableValid = true;
    for (int i = 0; i < queryLen; i++) {
        if (queryNumAA[i] >= alphabetAA || queryNum3Di[i] >= ALPHABET_SIZE_3DI) {
            tableValid = false;
            break;
        }
        const int code = queryNumAA[i] * ALPHABET_SIZE_3DI + queryNum3Di[i];
        if (rowOfCode[code] == -1) {
            rowOfCode[code] = (int)rowAA.size();
            rowAA.push_back(queryNumAA[i]);
            row3Di.push_back(queryNum3Di[i]);
        }
        fwdQueryRow[i] = rowOfCode[code];
    }
    if (tableValid) {
        fwdCodeAA.resize(fwdCodeCount);
        fwdCode3Di.resize(fwdCodeCount);
        for (int c = 0; c < fwdCodeCount; c++) {
            fwdCodeAA[c] = c / ALPHABET_SIZE_3DI;
            fwdCode3Di[c] = c % ALPHABET_SIZE_3DI;
        }
        const int rows = (int)rowAA.size();
        fwdTable.assign((size_t)rows * fwdCodeCount, 0.0f);
        fwdAddTable.assign((size_t)rows * fwdCodeCount, 0.0f);
        std::vector<float*> tableRows(rows), addTableRows(rows);
        for (int r = 0; r < rows; r++) {
            tableRows[r] = fwdTable.data() + (size_t)r * fwdCodeCount;
            addTableRows[r] = fwdAddTable.data() + (size_t)r * fwdCodeCount;
        }
        forwardScores(rowAA.data(), row3Di.data(), rows, fwdCodeAA.data(), fwdCode3Di.data(), fwdCodeCount, subMatAA,
                      tableRows.data());
        addForwardScores(rowAA.data(), row3Di.data(), rows, fwdCodeAA.data(), fwdCode3Di.data(), fwdCodeCount, subMatAA,
                         addTableRows.data());
    }
}

void LoLAlign::initQuery(float* x, float* y, float* z, Sequence& qSeqAA, Sequence& qSeq3Di,
                         int queryLen, SubstitutionMatrix& subMatAA, int maxTLen, int multiDomain)
{
    memcpy(queryX, x, sizeof(float) * queryLen);
    memcpy(queryY, y, sizeof(float) * queryLen);
    memcpy(queryZ, z, sizeof(float) * queryLen);

    lddtcalc->initQuery(queryLen, queryX, queryY, queryZ);
    lolProb.prepareQuery(queryLen, queryX, queryY, queryZ);

    this->queryLen = queryLen;
    this->querySeq = qSeqAA.getSeqData();
    this->query3DiSeq = qSeq3Di.getSeqData();
    queryNumAA = qSeqAA.numSequence;
    queryNum3Di = qSeq3Di.numSequence;

    initForwardTables(subMatAA);
    Coordinates queryCaCords;
    queryCaCords.x = queryX;
    queryCaCords.y = queryY;
    queryCaCords.z = queryZ;

    maxAnchorLen = std::max(queryLen, maxTLen);
    if(queryLen < 10){
        startAnchorLength = 0;
    }
    else{
        startAnchorLength = 3;
    }
    calcQueryNeighbors(queryX, queryY, queryZ, queryLen);
    for (int i = 0; i < queryLen; i++) {
        finalAnchorQuery[i] = i;
    }


    computeDiScore(queryNumAA, queryNum3Di, queryLen, finalAnchorQuery, finalAnchorQuery, subMatAA, lolScoreVec);
    float diScore = 0;
    qqScore = 0;
    for (int i = 0; i < queryLen; i++) {
        diScore += lolScoreVec[i];
        lolScoreVec[i] = 0;
    }

    for (int i = 0; i < queryLen; i++) {
        // -1 unless residues i and j are closer than 20 A
        std::fill(lolDist, lolDist + queryLen, -1.0f);
        std::fill(lolSeqDist, lolSeqDist + queryLen, -1.0f);
        for (int k = queryNbStart[i]; k < queryNbStart[i + 1]; k++) {
            const int j = queryNbIdx[k];
            lolDist[j] = 0;
            lolSeqDist[j] = std::copysign(1.0f, (i-j)) * logSeparation[std::abs(i-j)];
        }
        lolScore(lolDist, lolSeqDist, lolScoreVec, queryLen);
    }



    for (int i = 0; i < queryLen; i++) {
        qqScore += lolScoreVec[i];
    }
    qqScore = (qqScore + 3 * diScore);
    if (multiDomain == 0) {
        qqScore = qqScore / std::pow(queryLen * queryLen, 0.25);
    }
    
    return;
}

void LoLAlign::lolMatrix(const int* newQuery, const int* newTarget, int anchorLength, int* gaps, float** lolScoreMatrix)
{
    int gap0Start = gaps[0];
    int gap0End = gaps[1];
    int gap1Start = gaps[2];
    int gap1End = gaps[3];
    float seqDist = 0.0;

    for (int i = 0; i < anchorLength; i++) {
        const int anchorQ = newQuery[i];
        const int anchorT = newTarget[i];
        

        // query residues closer than 20 A to the anchor in the gap rows
        const int rowEnd = queryNbStart[anchorQ + 1];
        int k = queryNbCursor[i];
        if (k < rowEnd && queryNbIdx[k] < gap0Start) {
            k = std::lower_bound(queryNbIdx.begin() + k, queryNbIdx.begin() + rowEnd, gap0Start) - queryNbIdx.begin();
        }
        for (; k < rowEnd && queryNbIdx[k] < gap0End; k++) {
            const int j = queryNbIdx[k];
            const float dq = queryNbDist[k];
            minLolMatrixIndex = std::min(minLolMatrixIndex, j);
            maxLolMatrixIndex = std::max(maxLolMatrixIndex, j + 1);
            scoreRowIteration[j] = lolIteration;
            seqDist = std::copysign(1.0f, (anchorQ - j)) * logSeparation[std::abs(anchorQ - j)];
            lolScore(targetDistRow(anchorT) + gap1Start, dq, seqDist, lolScoreMatrix[j], gap1End - gap1Start, gap1Start);
        }
        queryNbCursor[i] = k;
    }
    return;
}


// dq, dSeq: query distance and sequence separation to the anchor, dt: target distances to the anchor
void LoLAlign::lolScore(const float* dt, float dq, float dSeq, float* score, int length, int start)
{
    simd_float zero = simdf32_setzero();
    const simd_float signBit = simdf32_set(-0.0f);
    const simd_float dqVec = simdf32_set(dq);

    simd_float seq0 = simdf32_mul(simdf32_set(dSeq), simdf32_set(w1[0][0]));
    simd_float seq1 = simdf32_mul(simdf32_set(dSeq), simdf32_set(w1[0][1]));
    simd_float seq2 = simdf32_mul(simdf32_set(dSeq), simdf32_set(w1[0][2]));
    int i = 0;
    for (; i <= length - VECSIZE_FLOAT; i += VECSIZE_FLOAT) {
        // |dq - dt|
        simd_float dDistVec = simdf32_andnot(signBit, simdf32_sub(dqVec, simdf32_loadu(&dt[i])));

        simd_float hl0 = simdf32_add(seq0, simdf32_fmadd(dDistVec, simdf32_set(w1[1][0]), simdf32_set(b1[0])));
        simd_float hl1 = simdf32_add(seq1, simdf32_fmadd(dDistVec, simdf32_set(w1[1][1]), simdf32_set(b1[1])));
        simd_float hl2 = simdf32_add(seq2, simdf32_fmadd(dDistVec, simdf32_set(w1[1][2]), simdf32_set(b1[2])));

        hl0 = simdf32_max(hl0, zero);
        hl1 = simdf32_max(hl1, zero);
        hl2 = simdf32_max(hl2, zero);

        simd_float scoreVec = simdf32_loadu(&score[i + start]);
        scoreVec = simdf32_fmadd(hl0, simdf32_set(w2[0]), scoreVec);
        scoreVec = simdf32_fmadd(hl1, simdf32_set(w2[1]), scoreVec);
        scoreVec = simdf32_fmadd(hl2, simdf32_set(w2[2]), scoreVec);
        scoreVec = simdf32_add(scoreVec, simdf32_set(b2));

        simdf32_storeu(&score[i + start], scoreVec);
    }

    // the hidden layer goes through memory: then GCC fuses only dist * w1[1][k] into the addition, as it always did
    for (; i < length; ++i) {
        const float dist = std::abs(dq - dt[i]);
        for (int k = 0; k < 3; ++k) {
            hiddenLayer[i][k] = dSeq * w1[0][k];
            hiddenLayer[i][k] += dist * w1[1][k];
            hiddenLayer[i][k] += b1[k];
            hiddenLayer[i][k] = std::max(0.0f, hiddenLayer[i][k]);
            score[i + start] += hiddenLayer[i][k] * w2[k];
        }
        score[i + start] += b2;
    }
}



void LoLAlign::lolScore(const float* dist, const float* dSeq, float* score, int length)
{
    // the hidden layer goes through memory, see the other lolScore
    for (int i = 0; i < length; ++i) {
        if (dist[i] >= 0) {
            for (int k = 0; k < 3; ++k) {
                hiddenLayer[i][k] = dSeq[i] * w1[0][k];
                hiddenLayer[i][k] += dist[i] * w1[1][k];
                hiddenLayer[i][k] += b1[k];
                hiddenLayer[i][k] = std::max(0.0f, hiddenLayer[i][k]);
                score[i] += hiddenLayer[i][k] * w2[k];
            }
            score[i] += b2;
        }
    }
}


void LoLAlign::computeForwardScoreMatrix(
        unsigned char* targetNumAA,
        unsigned char* targetNum3Di,
        int queryLen,
        int targetLen,
        SubstitutionMatrix& subMatAA,
        float** scoreForward)
{
    if (mapTargetCodes(targetNumAA, targetNum3Di, targetLen)) {
        const int *code = fwdTargetCode.data();
        for (int i = 0; i < queryLen; ++i) {
            const float *row = fwdTable.data() + (size_t)fwdQueryRow[i] * fwdCodeCount;
            float *out = scoreForward[i];
            for (int j = 0; j < targetLen; ++j) {
                out[j] = row[code[j]];
            }
        }
        return;
    }
    forwardScores(queryNumAA, queryNum3Di, queryLen, targetNumAA, targetNum3Di, targetLen, subMatAA, scoreForward);
}

// out of line, so that the tables of initForwardTables come from the same code as the fallback calls
__attribute__((noinline))
void LoLAlign::forwardScores(const unsigned char* qAA, const unsigned char* q3Di, int qLen, const unsigned char* tAA,
                             const unsigned char* t3Di, int tLen, const SubstitutionMatrix& subMatAA, float** scoreForward) {
    for (int i = 0; i < qLen; ++i) {
        for (int j = 0; j < tLen; ++j) {
            scoreForward[i][j] = static_cast<float>((subMatAA.subMatrix[qAA[i]][tAA[j]] * 1.4) + (scoringMatrix3Di[q3Di[i]][t3Di[j]] * 2.1));
        }
    }
}

__attribute__((noinline))
void LoLAlign::addForwardScores(const unsigned char* qAA, const unsigned char* q3Di, int qLen, const unsigned char* tAA,
                                const unsigned char* t3Di, int tLen, const SubstitutionMatrix& subMatAA, float** scoreForward) {
    for (int i = 0; i < qLen; ++i) {
        for (int j = 0; j < tLen; ++j) {
            scoreForward[i][j] += static_cast<float>((subMatAA.subMatrix[qAA[i]][tAA[j]] * 1.4) + (scoringMatrix3Di[q3Di[i]][t3Di[j]] * 2.1));
        }
    }
}

// target pair codes, the columns of the tables; false if there is no table or a letter is outside it
bool LoLAlign::mapTargetCodes(const unsigned char* targetNumAA, const unsigned char* targetNum3Di, int targetLen) {
    if (fwdTable.empty()) {
        return false;
    }
    fwdTargetCode.resize(targetLen);
    for (int j = 0; j < targetLen; ++j) {
        if (targetNumAA[j] * ALPHABET_SIZE_3DI >= fwdCodeCount || targetNum3Di[j] >= ALPHABET_SIZE_3DI) {
            return false;
        }
        fwdTargetCode[j] = targetNumAA[j] * ALPHABET_SIZE_3DI + targetNum3Di[j];
    }
    return true;
}

void LoLAlign::addForwardScoreMatrix(
    unsigned char* targetNumAA,
    unsigned char* targetNum3Di,
    int queryLen,
    int targetLen,
    SubstitutionMatrix& subMatAA,
    float** scoreForward)
{
    if (mapTargetCodes(targetNumAA, targetNum3Di, targetLen)) {
        const int *code = fwdTargetCode.data();
        for (int i = 0; i < queryLen; ++i) {
            const float *row = fwdAddTable.data() + (size_t)fwdQueryRow[i] * fwdCodeCount;
            float *out = scoreForward[i];
            for (int j = 0; j < targetLen; ++j) {
                out[j] += row[code[j]];
            }
        }
        return;
    }
    addForwardScores(queryNumAA, queryNum3Di, queryLen, targetNumAA, targetNum3Di, targetLen, subMatAA, scoreForward);
}

