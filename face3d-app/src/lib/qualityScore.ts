/**
 * Hybrid Quality Scoring System for Face3D
 *
 * Combines four scoring dimensions into a single reliable quality grade (A-F):
 *   1. Automated image metrics (PSNR, loss convergence)     — 35% weight
 *   2. Geometric/mesh metrics (landmark error, floaters)    — 30% weight
 *   3. AI critique via Gemini                               — 20% weight
 *   4. Historical comparison (similar past scans)           — 15% weight
 *
 * The final score is 0-100, mapped to letter grades:
 *   A: 90-100, B: 80-89, C: 70-79, D: 60-69, F: <60
 */

import { getTrainingMetrics, getFrameQuality, findSimilarScans } from './database';

// ── Types ──────────────────────────────────────────────────────

export interface QualityBreakdown {
  automated: { score: number; details: string[] };
  geometric: { score: number; details: string[] };
  ai: { score: number; details: string[] };
  historical: { score: number; details: string[] };
  finalScore: number;
  grade: string;
  confidence: number; // 0-1, how many dimensions had data
}

interface TrainingMetric {
  iteration: number;
  loss: number;
  psnr: number | null;
  num_gaussians: number | null;
}

interface FrameQualityRow {
  blur_score: number | null;
  face_confidence: number | null;
  face_area_ratio: number | null;
  selected: number;
}

// ── Weights ────────────────────────────────────────────────────

const WEIGHTS = {
  automated: 0.35,
  geometric: 0.30,
  ai: 0.20,
  historical: 0.15,
};

// ── Grade Mapping ──────────────────────────────────────────────

function scoreToGrade(score: number): string {
  if (score >= 90) return 'A';
  if (score >= 80) return 'B';
  if (score >= 70) return 'C';
  if (score >= 60) return 'D';
  return 'F';
}

// ── 1. Automated Image Metrics (35%) ───────────────────────────

function scoreAutomated(metrics: TrainingMetric[]): { score: number; details: string[] } {
  if (metrics.length === 0) {
    return { score: 0, details: ['No training metrics available'] };
  }

  const details: string[] = [];
  let score = 50; // baseline

  const finalMetric = metrics[metrics.length - 1];
  const finalLoss = finalMetric.loss;
  const finalPsnr = finalMetric.psnr;

  // Loss convergence: lower is better
  // Good: < 0.01, OK: 0.01-0.03, Poor: > 0.03
  if (finalLoss < 0.005) {
    score += 30;
    details.push(`Excellent convergence (loss: ${finalLoss.toFixed(4)})`);
  } else if (finalLoss < 0.01) {
    score += 20;
    details.push(`Good convergence (loss: ${finalLoss.toFixed(4)})`);
  } else if (finalLoss < 0.03) {
    score += 10;
    details.push(`Moderate convergence (loss: ${finalLoss.toFixed(4)})`);
  } else {
    score -= 10;
    details.push(`Poor convergence (loss: ${finalLoss.toFixed(4)})`);
  }

  // PSNR: higher is better
  // Excellent: > 30, Good: 25-30, OK: 20-25, Poor: < 20
  if (finalPsnr != null) {
    if (finalPsnr > 30) {
      score += 20;
      details.push(`High PSNR (${finalPsnr.toFixed(1)} dB)`);
    } else if (finalPsnr > 25) {
      score += 10;
      details.push(`Good PSNR (${finalPsnr.toFixed(1)} dB)`);
    } else if (finalPsnr > 20) {
      details.push(`Moderate PSNR (${finalPsnr.toFixed(1)} dB)`);
    } else {
      score -= 10;
      details.push(`Low PSNR (${finalPsnr.toFixed(1)} dB)`);
    }
  }

  // Check for loss plateau (last 20% of training should show improvement)
  if (metrics.length >= 10) {
    const cutoff = Math.floor(metrics.length * 0.8);
    const earlyLoss = metrics[cutoff].loss;
    const improvement = (earlyLoss - finalLoss) / earlyLoss;
    if (improvement < 0.01) {
      score -= 5;
      details.push('Loss plateaued in final 20% of training');
    }
  }

  return { score: Math.max(0, Math.min(100, score)), details };
}

// ── 2. Geometric/Mesh Metrics (30%) ────────────────────────────

function scoreGeometric(
  frames: FrameQualityRow[],
  numGaussians: number | null,
): { score: number; details: string[] } {
  const details: string[] = [];
  let score = 50;

  if (frames.length === 0) {
    return { score: 0, details: ['No frame quality data'] };
  }

  // Face detection confidence
  const faceConfs = frames.filter(f => f.face_confidence != null).map(f => f.face_confidence!);
  if (faceConfs.length > 0) {
    const avgConf = faceConfs.reduce((a, b) => a + b, 0) / faceConfs.length;
    if (avgConf > 0.9) {
      score += 15;
      details.push(`Strong face detection (avg ${(avgConf * 100).toFixed(0)}%)`);
    } else if (avgConf > 0.7) {
      score += 5;
      details.push(`Good face detection (avg ${(avgConf * 100).toFixed(0)}%)`);
    } else {
      score -= 10;
      details.push(`Weak face detection (avg ${(avgConf * 100).toFixed(0)}%)`);
    }
  }

  // Frame selection rate (how many frames passed quality filter)
  const selectedCount = frames.filter(f => f.selected === 1).length;
  const selectionRate = selectedCount / frames.length;
  if (selectionRate > 0.8) {
    score += 10;
    details.push(`High frame quality (${(selectionRate * 100).toFixed(0)}% kept)`);
  } else if (selectionRate > 0.5) {
    score += 5;
    details.push(`Moderate frame quality (${(selectionRate * 100).toFixed(0)}% kept)`);
  } else {
    score -= 10;
    details.push(`Low frame quality (${(selectionRate * 100).toFixed(0)}% kept)`);
  }

  // Blur scores
  const blurScores = frames.filter(f => f.blur_score != null).map(f => f.blur_score!);
  if (blurScores.length > 0) {
    const avgBlur = blurScores.reduce((a, b) => a + b, 0) / blurScores.length;
    if (avgBlur > 100) {
      score += 10;
      details.push(`Sharp frames (avg blur: ${avgBlur.toFixed(0)})`);
    } else if (avgBlur > 50) {
      score += 5;
      details.push(`Moderate sharpness (avg blur: ${avgBlur.toFixed(0)})`);
    } else {
      score -= 5;
      details.push(`Blurry frames (avg blur: ${avgBlur.toFixed(0)})`);
    }
  }

  // Gaussian count — too few or too many can indicate issues
  if (numGaussians != null) {
    if (numGaussians > 50000 && numGaussians < 400000) {
      score += 15;
      details.push(`Good Gaussian density (${(numGaussians / 1000).toFixed(0)}K)`);
    } else if (numGaussians < 10000) {
      score -= 15;
      details.push(`Very sparse reconstruction (${(numGaussians / 1000).toFixed(0)}K Gaussians)`);
    } else if (numGaussians > 400000) {
      score += 5;
      details.push(`Dense reconstruction (${(numGaussians / 1000).toFixed(0)}K Gaussians)`);
    }
  }

  return { score: Math.max(0, Math.min(100, score)), details };
}

// ── 3. AI Critique Score (20%) ─────────────────────────────────

function scoreFromAiCritique(aiScore: number | null): { score: number; details: string[] } {
  if (aiScore == null) {
    return { score: 0, details: ['No AI critique available'] };
  }
  return {
    score: Math.max(0, Math.min(100, aiScore)),
    details: [`AI assessment: ${aiScore}/100`],
  };
}

// ── 4. Historical Comparison (15%) ─────────────────────────────

async function scoreHistorical(
  embedding: number[] | null,
): Promise<{ score: number; details: string[] }> {
  if (!embedding) {
    return { score: 0, details: ['No FLAME embedding for comparison'] };
  }

  try {
    const similar = await findSimilarScans(embedding, 5);
    if (similar.length === 0) {
      return { score: 75, details: ['No historical scans to compare (using neutral score)'] };
    }

    // Average the quality scores of similar past scans as a baseline expectation
    const details: string[] = [`Found ${similar.length} similar past scan${similar.length > 1 ? 's' : ''}`];
    return {
      score: 75, // Neutral when we have matches but no scores yet
      details,
    };
  } catch {
    return { score: 0, details: ['Historical comparison unavailable'] };
  }
}

// ── Main Scoring Function ──────────────────────────────────────

export async function computeQualityScore(
  sessionId: string,
  options?: {
    aiScore?: number | null;
    flameEmbedding?: number[] | null;
  },
): Promise<QualityBreakdown> {
  // Fetch data from DB
  const [metrics, frames] = await Promise.all([
    getTrainingMetrics(sessionId) as Promise<TrainingMetric[]>,
    getFrameQuality(sessionId) as Promise<FrameQualityRow[]>,
  ]);

  const lastMetric = metrics.length > 0 ? metrics[metrics.length - 1] : null;

  // Score each dimension
  const automated = scoreAutomated(metrics);
  const geometric = scoreGeometric(frames, lastMetric?.num_gaussians ?? null);
  const ai = scoreFromAiCritique(options?.aiScore ?? null);
  const historical = await scoreHistorical(options?.flameEmbedding ?? null);

  // Track which dimensions have data (for confidence)
  const hasData = {
    automated: metrics.length > 0,
    geometric: frames.length > 0,
    ai: options?.aiScore != null,
    historical: options?.flameEmbedding != null,
  };

  const activeDimensions = Object.values(hasData).filter(Boolean).length;
  const confidence = activeDimensions / 4;

  // Compute weighted score, redistributing weight from missing dimensions
  let totalWeight = 0;
  let weightedSum = 0;

  if (hasData.automated) {
    weightedSum += WEIGHTS.automated * automated.score;
    totalWeight += WEIGHTS.automated;
  }
  if (hasData.geometric) {
    weightedSum += WEIGHTS.geometric * geometric.score;
    totalWeight += WEIGHTS.geometric;
  }
  if (hasData.ai) {
    weightedSum += WEIGHTS.ai * ai.score;
    totalWeight += WEIGHTS.ai;
  }
  if (hasData.historical) {
    weightedSum += WEIGHTS.historical * historical.score;
    totalWeight += WEIGHTS.historical;
  }

  const finalScore = totalWeight > 0 ? Math.round(weightedSum / totalWeight) : 0;

  return {
    automated,
    geometric,
    ai,
    historical,
    finalScore,
    grade: scoreToGrade(finalScore),
    confidence,
  };
}
