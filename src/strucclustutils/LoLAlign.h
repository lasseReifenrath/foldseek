#ifndef FOLDSEEK_LOLALIGN_H
#define FOLDSEEK_LOLALIGN_H

#include "SubstitutionMatrix.h"
#include "IndexReader.h"
#include "DBReader.h"
#include "Fwbw.h"

#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <vector>

#include "Matcher.h"
#include "tmalign/Coordinates.h"
#include "LDDT.h"
#include "LoLAlignProbability.h"

class LoLAlign {
public:
    LoLAlign(unsigned int maxSeqLen, bool exact, int candidateSeeds, int refineSeeds);
    ~LoLAlign();

    void computeForwardScoreMatrix(
        unsigned char * targetNumAA,
        unsigned char *targetNum3Di,
        int queryLen,
        int targetLen,
        SubstitutionMatrix &subMatAA,
        float** scoreForward
    );
    void addForwardScoreMatrix(
        unsigned char * targetNumAA,
        unsigned char *targetNum3Di,
        int queryLen,
        int targetLen,
        SubstitutionMatrix &subMatAA,
        float** scoreForward
    );
    void calcGap(int* anchorQuery, int* anchorTarget, int* gaps, int queryLen, int targetLen);

    void initQuery(float* x, float* y, float* z, Sequence& qSeqAA, Sequence& qSeq3Di, int queryLen, SubstitutionMatrix& subMatAA, int maxTLen, int multiDomain);
    
    Matcher::result_t align(unsigned int dbKey, float* targetX, float* targetY, float* targetZ,
                            Sequence& tSeqAA, Sequence& tSeq3Di, int targetLen, SubstitutionMatrix& subMatAA, FwBwAligner* fwbwaln, int multiDomain);
    float maxSubArray(float* nums, int numsSize);

    void alignStartAnchors(int* anchorQuery, int* anchorTarget, int maxQuery, int maxTarget, int* anchorLength, float** fwbwP, float** lolScoreMatrix, int targetLen);
    void indexSort(float* nums, int* index, int numsSize);


    void lolFwbw(float** scoreForward, float** probabilityMatrix,
        size_t queryLen, size_t targetLen, size_t assignTargetLen,
        float go, float ge, float T, int length, int blocks, int* gaps);

    void rescaleBlocks(float **matrix, float **scale, size_t rows, size_t length, size_t blocks, size_t targetLen);
    void forwardBackwardSaveBlockMaxLocal(float** S, float** zInit,
                                            float T, float go, float ge,
                                            size_t rows, size_t start, size_t end, size_t memcpyCols, size_t targetLen,
                                            float** zm, float* zmax, float** zmBlock, float* zeBlock, float* zfBlock, int* gaps);
    void calcQueryNeighbors(const float* x, const float* y, const float* z, int len);
    void linkAnchors(int anchorLen);
    void unlinkAnchors(int anchorLen);
    void anchorRowScore(int i, bool withTargetDist, float* score);
    float calcDiscore(int* anchorQuery, int* anchorTarget, int anchorLength);
    void lolMatrix(const int* newQuery, const int* newTarget, int anchorLength, int* gaps, float** lolScoreMatrix);
    const float* targetDistRow(int r);
    void lolScore(const float* dist, const float* dSeq, float* score, int length);
    void setStartAnchorLength(int length) { startAnchorLength = length; }
    

    void lolScore(const float* dt, float dq, float dSeq, float* score, int length, int start);
    void computeDiScore(
        unsigned char* targetNumAA,
        unsigned char* targetNum3Di,
        int anchorLen,
        int* finalAnchorQuery,
        int* finalAnchorTarget,
        SubstitutionMatrix& subMatAA,
        float* scoreForward);

private:
    std::string backtrace;
    float* queryX;
    float* queryY;
    float* queryZ;
    // CA coordinates of the current target; row r of its distance matrix is computed when first needed in a pair
    // and packed into distancekl (targetDistRows[r], targetDistStride floats apart)
    const float *targetCaX, *targetCaY, *targetCaZ;
    int targetCaLen;
    size_t targetDistStride;
    int targetDistRowsUsed;
    std::vector<float*> targetDistRows;
    float** scoreForward;
    float** probabilityMatrix;
    // query residues closer than 20 A to residue i and their distances:
    // queryNbIdx/queryNbDist[queryNbStart[i], queryNbStart[i + 1])
    std::vector<int> queryNbStart;
    std::vector<int> queryNbIdx;
    std::vector<float> queryNbDist;
    // lolMatrix: position of each new anchor in its neighbor list
    std::vector<int> queryNbCursor;
    // final anchors at each query position as linked lists, see linkAnchors
    std::vector<int> firstAnchorAt;
    std::vector<int> nextAnchorAt;
    // refinement FwBw gaps of the previous iteration and the iteration that last changed each score row
    struct FwbwGap { int rowStart, rowEnd, colStart, colEnd; float temperature, maxP; };
    std::vector<FwbwGap> fwbwGapsPrev, fwbwGapsCur;
    std::vector<int> scoreRowIteration;
    int lolIteration = 0;
    // log(1 + k) for sequence separation k
    std::vector<float> logSeparation;
    // largest P of each row
    std::vector<float> rowMaxP;
    float** distancekl;
    float** lolScoreMatrix;
    int** anchorQuery;
    int** anchorTarget;
    int numStartAnchors = 10;
    int StartAnchorNum = 10;
    int candidateSeeds = 10;
    int seedNumber = 3;
    float startAnchorGo = -6.0;
    float startAnchorGe = -3.0;
    float startAnchorT = 2.0;
    int startAnchorLength = 3;
    float lolGo = -1.5;
    float lolGe = -0.0;
    float lolMinP = 0.7;
    float lolT = 4;
    // lolScore: hidden layer of the scalar parts
    float** hiddenLayer;
    int* startAnchorIndex;
    float* startAnchorScores;
    int* anchorLength;
    int* newAnchorLength;
    // (query, target) positions of the anchors added last (status 2) for the seed being refined,
    // in query order; consumed by lolMatrix
    int* newAnchorQuery;
    int* newAnchorTarget;
    int* gaps;
    float* lolDist;
    float* lolSeqDist;
    float* lolScoreVec;
    int* finalAnchorQuery;
    int* finalAnchorTarget;
    // aligned query/target positions, reused across alignments like backtrace
    std::vector<int> qAln, tAln;
    float qqScore;
    int maxAnchorLen;

    unsigned char* queryNumAA;
    unsigned char* queryNum3Di;
    static const int ALPHABET_SIZE_3DI = 20;
    // 1.4 * AA + 2.1 * 3Di scores of computeForwardScoreMatrix and addForwardScoreMatrix: one row per (AA, 3Di)
    // letter pair of the query, column AA * ALPHABET_SIZE_3DI + 3Di of the target pair; empty if a letter is
    // outside the matrices
    int fwdCodeCount;
    std::vector<float> fwdTable, fwdAddTable;
    std::vector<int> fwdQueryRow;
    std::vector<int> fwdTargetCode;
    std::vector<unsigned char> fwdCodeAA, fwdCode3Di;
    bool mapTargetCodes(const unsigned char* targetNumAA, const unsigned char* targetNum3Di, int targetLen);
    void initForwardTables(const SubstitutionMatrix& subMatAA);
    void forwardScores(const unsigned char* qAA, const unsigned char* q3Di, int qLen, const unsigned char* tAA,
                       const unsigned char* t3Di, int tLen, const SubstitutionMatrix& subMatAA, float** scoreForward);
    void addForwardScores(const unsigned char* qAA, const unsigned char* q3Di, int qLen, const unsigned char* tAA,
                          const unsigned char* t3Di, int tLen, const SubstitutionMatrix& subMatAA, float** scoreForward);
    float* lolScoreVecSh;

    LDDTCalculator*      lddtcalc;   // foldseek LDDT (one of the LoLAlignProbability features)
    LoLAlignProbability  lolProb;    // homology probability classifier (written into result.score * 10000)

    float w1[2][3] = {
        {-1.3584513e-04,7.6149112e-01,-8.1348085e-01 },
        {9.9329501e-01 , 5.7029408e-01, 6.0702705e-01}
    };
    float b1[3] = {0.7043129 , 0.374659  , 0.39905924};

    float w2[3] = {-0.776632  ,  0.61055756, 0.5823986};
    float b2 = 0.38799961;

    // Load weights and biases into SIMD registers
   
    int minLolMatrixIndex;
    int maxLolMatrixIndex;

    int queryLen;
    const char* querySeq;
    const char* query3DiSeq;
    std::string seqM, seqxA, seqyA;// for output alignment

    Coordinates xtm, ytm, xt, r1, r2;
    bool computeExactScore;
    int* invmap;
    const float scoringMatrix3Di[ALPHABET_SIZE_3DI][ALPHABET_SIZE_3DI] = {
        // A   B   C   D   E   F   G   H   I   J   K   L   M   N   O   P   Q   R   S   T
        {10, -1,  1,  7,  6,  2,  2, -4,  1, -1, -6, -2, -1,  4, -1, -5, -1, -5,  3,  2}, // A
        {-1,  7, -4, -5, -3, -2, -1, -10, -10,  1, -11,  1, -2,  2,  1,  1, -6,  1, -5, -5}, // B
        {1, -4,  0, -3, -1,  0,  1, -4, -5, -6, -5, -4, -3, -3, -2, -6, -2, -6, -2, -2}, // C
        {7, -5, -3, 15,  3, -2,  1, -8, -5, -4, -12, -5, -5,  1, -3, -7, -5, -10, -2,  3}, // D
        {6, -3, -1,  3, 10,  1,  1, -2,  4, -1, -5, -3, -2,  4, -2, -6,  0, -6,  7,  0}, // E
        {2, -2,  0, -2,  1, 10,  7,  3, -3, -5,  3,  1, -2, -1,  5, -1,  7, -5, -1,  3}, // F
        {2, -1,  1,  1,  1,  7, 10,  0, -4, -4, -1,  3, -1,  0,  4, -1,  3, -4, -1,  8}, // G
        {-4, -10, -4, -8, -2,  3,  0, 11, -1, -10, 11, -5, -6, -3,  0, -7,  9, -11, -2, -3}, // H
        {1, -10, -5, -5,  4, -3, -4, -1, 13, -9, -4, -9, -7, -2, -6, -12, -2, -13,  9, -3}, // I
        {-1,  1, -6, -4, -1, -5, -4, -10, -9,  6, -13, -2, -3,  3, -2, -3, -7,  0, -6, -6}, // J
        {-6, -11, -5, -12, -5,  3, -1, 11, -4, -13, 15, -6, -8, -7, -1, -8,  7, -14, -2, -4}, // K
        {-2,  1, -4, -5, -3,  1,  3, -5, -9, -2, -6,  8, -1, -1,  4,  4, -2,  0, -6, -1}, // L
        {-1, -2, -3, -5, -2, -2, -1, -6, -7, -3, -8, -1,  1, -1, -1, -3, -4, -2, -5, -5}, // M
        {4,  2, -3,  1,  4, -1,  0, -3, -2,  3, -7, -1, -1,  7,  0, -2, -3, -1,  1, -1}, // N
        {-1,  1, -2, -3, -2,  5,  4,  0, -6, -2, -1,  4, -1,  0,  8,  4,  3, -1, -3,  1}, // O
        {-5,  1, -6, -7, -6, -1, -1, -7, -12, -3, -8,  4, -3, -2,  4,  6, -4,  0, -9, -5}, // P
        {-1, -6, -2, -5,  0,  7,  3,  9, -2, -7,  7, -2, -4, -3,  3, -4, 11, -8, -2,  0}, // Q
        {-5,  1, -6, -10, -6, -5, -4, -11, -13,  0, -14,  0, -2, -1, -1,  0, -8,  2, -9, -9}, // R
        {3, -5, -2, -2,  7, -1, -1, -2,  9, -6, -2, -6, -5,  1, -3, -9, -2, -9, 11, -2}, // S
        {2, -5, -2,  3,  0,  3,  8, -3, -3, -6, -4, -1, -5, -1,  1, -5,  0, -9, -2, 14}  // T
    };
};

#endif